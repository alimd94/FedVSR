
import torch
from os import path as osp
from torch.amp import GradScaler
from torch.cuda.amp import autocast
import archs  # noqa F401
import model  # noqa F401
from basicsr.data import build_dataset
from basicsr.models import build_model

from basicsr.utils.dist_util import get_dist_info
from basicsr.utils.options import ordered_yaml
from basicsr.utils import set_random_seed

from collections import OrderedDict
from typing import List
import numpy as np
import torch
import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.server import ServerConfig
from flwr.common import ndarrays_to_parameters, Context
from basicsr.utils import (make_exp_dirs, mkdir_and_rename)
from flwr.common import (
    parameters_to_ndarrays,
)
import os.path
import math
import argparse
import random
import numpy as np
from collections import OrderedDict
import torch
import yaml
from dataset import load_datasets
import os
from recurrent_mix_precision_train import load_resume_state
from functools import reduce
from archs.iart_arch import IART

from pytorch_wavelets import DWTForward

DEVICE = torch.device("cuda:0")  # Try "cuda" to train on GPU
torch.backends.cudnn.benchmark = True


seed = 123
if seed is None:
    seed = random.randint(1, 10000)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

flwr.common.logger.configure(identifier="myFlowerExperiment", filename="log.txt")

def parse_options(root_path, is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default="options/IART_REDS_N16_600K.yml")
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none', help='job launcher')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--force_yml', nargs='+', default=None, help='Force to update yml files. Examples: train:ema_decay=0.999')
    args = parser.parse_args()

    # parse yml to dict
    with open(args.opt, mode='r') as f:
        opt = yaml.load(f, Loader=ordered_yaml()[0])

    # distributed settings
    if args.launcher == 'none':
        opt['dist'] = False
        
    opt['rank'], opt['world_size'] = get_dist_info()

    # random seed

    set_random_seed(seed + opt['rank'])

    opt['auto_resume'] = args.auto_resume
    opt['is_train'] = is_train

    # debug setting
    if args.debug and not opt['name'].startswith('debug'):
        opt['name'] = 'debug_' + opt['name']

    if opt['num_gpu'] == 'auto':
        opt['num_gpu'] = torch.cuda.device_count()

    # datasets
    for phase, dataset in opt['datasets'].items():
        # for multiple datasets, e.g., val_1, val_2; test_1, test_2
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        if 'scale' in opt:
            dataset['scale'] = opt['scale']
        if dataset.get('dataroot_gt') is not None:
            dataset['dataroot_gt'] = osp.expanduser(dataset['dataroot_gt'])
        if dataset.get('dataroot_lq') is not None:
            dataset['dataroot_lq'] = osp.expanduser(dataset['dataroot_lq'])

    # paths
    for key, val in opt['path'].items():
        if (val is not None) and ('resume_state' in key or 'pretrain_network' in key):
            opt['path'][key] = osp.expanduser(val)

    if is_train:
        experiments_root = opt['path'].get('experiments_root')
        if experiments_root is None:
            experiments_root = osp.join(root_path, 'experiments')
        experiments_root = osp.join(experiments_root, opt['name'])

        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = osp.join(experiments_root, 'models')
        opt['path']['training_states'] = osp.join(experiments_root, 'training_states')
        opt['path']['log'] = experiments_root
        opt['path']['visualization'] = osp.join(experiments_root, 'visualization')

        # change some options for debug mode
        if 'debug' in opt['name']:
            if 'val' in opt:
                opt['val']['val_freq'] = 8
            opt['logger']['print_freq'] = 1
            opt['logger']['save_checkpoint_freq'] = 8
    else:  # test
        results_root = opt['path'].get('results_root')
        if results_root is None:
            results_root = osp.join(root_path, 'results')
        results_root = osp.join(results_root, opt['name'])

        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root
        opt['path']['visualization'] = osp.join(results_root, 'visualization')

    return opt, args

root_path = osp.abspath(osp.join(__file__, osp.pardir))
opt, args = parse_options(root_path, is_train=True)
opt['root_path'] = root_path
opt['auto_resume'] = True

resume_state = load_resume_state(opt)
    # mkdir for experiments and logger
if resume_state is None:
    make_exp_dirs(opt)
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and opt['rank'] == 0:
        mkdir_and_rename(osp.join(opt['root_path'], 'tb_logger', opt['name']))

NUM_PARTITIONS = 40
BATCH_SIZE = 3
RND = 1
NUM_ROUNDS = 100
NUM_PARTICIPATION = 4
SAVE_DIR = "IART_FedVSR"


class HighFreqLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-9):
        super(HighFreqLoss, self).__init__()
        self.eps = eps
        self.dwt = DWTForward(J=1, wave='db1', mode='zero')

    def forward(self, x, y,alpha=0.5,penalty=0.1):
      
        b,t, c, h, w = x.size()
        x = x.reshape(-1, c, h, w)
        y = y.reshape(-1, c, h, w)

        _, Yh_x = self.dwt(x)
        _, Yh_y = self.dwt(y)
        # print("Yh requires_grad:", Yh_x[0].requires_grad)
        diff = Yh_x[0] - Yh_y[0]
        loss_hf = torch.mean(torch.sqrt((diff * diff) + self.eps))

        alpha*loss_hf
        return loss

HiFreLoss = HighFreqLoss()

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.net_g.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    # print(type(parameters),"TYPEP")
    params_dict = zip(net.net_g.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.net_g.load_state_dict(state_dict, strict=True)

params = get_parameters(build_model(opt))

def train_with_newLoss(net, trainloader,scaler, epochs):
    current_step = 0 
    loss = 0.0
    for epoch in range(epochs):
        for i, train_data in enumerate(trainloader):
            current_step += 1
            net.update_learning_rate(current_step, warmup_iter=opt['train'].get('warmup_iter', -1))
            net.feed_data(train_data)
            if net.fix_flow_iter:
                if current_step == 1:
                    for name, param in net.net_g.named_parameters():
                        if 'spynet' in name or 'deform' in name:
                            param.requires_grad_(False)
                elif current_step == net.fix_flow_iter:
                    net.net_g.requires_grad_(True)
            net.optimizer_g.zero_grad()
            with autocast():
                net.output = net.net_g(net.lq)
                l_total = 0
                loss_dict = OrderedDict()
                if net.cri_pix:
                    l_pix = net.cri_pix(net.output, net.gt)+HiFreLoss(net.output,net.gt)
                    l_total += l_pix
                    loss_dict['l_pix'] = l_pix
                if net.cri_perceptual:
                    l_percep, l_style = net.cri_perceptual(net.output, net.gt)
                    if l_percep is not None:
                        l_total += l_percep
                        loss_dict['l_percep'] = l_percep
                    if l_style is not None:
                        l_total += l_style
                        loss_dict['l_style'] = l_style
                loss += l_total.item()
                scaler.scale(l_total).backward()
                scaler.step(net.optimizer_g)
                scaler.update()

                net.log_dict = net.reduce_loss_dict(loss_dict)
            if net.ema_decay > 0:
                net.model_ema(decay=net.ema_decay)            

    return loss/current_step
    
class FlowerClient(NumPyClient):
    def __init__(self, partition_id, opt, trainloader):
        import archs  # noqa F401
        import model  # noqa F401
        self.partition_id = partition_id
        self.net = build_model(opt)
        self.scaler = GradScaler()
        self.trainloader = trainloader[int(self.partition_id)]

    def get_parameters(self,):
        return get_parameters(self.net)

    def fit(self, parameters,config):
        set_parameters(self.net, parameters)

        loss = train_with_control_variates(self.net, self.trainloader,self.scaler,1)

        return get_parameters(self.net), len(self.trainloader), {'loss': loss}, 

    def evaluate(self, parameters, config):
        return .1, 1, {"accuracy": 0}

def client_fn(context: Context) -> Client:
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = build_dataset(dataset_opt)
            break
    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    trainloaders, _, _ = load_datasets(train_set, train_set,
         num_clients=num_partitions, batch_size=BATCH_SIZE,balance=True,iid=True,#alpha=0.1
    )

    return FlowerClient(partition_id, opt, trainloaders).to_client()

client = ClientApp(client_fn=client_fn)

def save_model_weights(round_number, parameters,opt=opt):
    net = build_model(opt)
    params_dict = zip(net.net_g.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.net_g.load_state_dict(state_dict, strict=True)
    net.save(1,round_number)

def aggregate_with_lowloss(results,weight_results, adaptive_step=1.0):
    """Compute weighted average with inverse loss weighting and adaptive step."""
    # Extract losses and calculate inverse losses
    losses = [fit_res.metrics["loss"] for _, fit_res in results]
    inverse_losses = [1 / loss for loss in losses]

    # Apply the adaptive step to the inverse losses
    adjusted_inverse_losses = [inv_loss ** adaptive_step for inv_loss in inverse_losses]

    # Normalize adjusted inverse losses to sum to 1
    total_adjusted_inverse_loss = sum(adjusted_inverse_losses)
    normalized_adjusted_inverse_losses = [adj_inv_loss / total_adjusted_inverse_loss for adj_inv_loss in adjusted_inverse_losses]

    # Create a list of weights, each multiplied by the related normalized adjusted inverse loss
    weighted_weights = [
        [layer * weight for layer in weights] for (weights, _), weight in zip(weight_results, normalized_adjusted_inverse_losses)
    ]

    # Calculate the weighted average of the weights
    weights_prime = [
        reduce(np.add, layer_updates)
        for layer_updates in zip(*weighted_weights)
    ]

    return weights_prime


class FedVSRStrategy(flwr.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs) 
        self.base_work_dir = SAVE_DIR
        global RND
        self.round = RND
        # self.global_model = None
    
    def aggregate_fit(self,server_round,results,failures):

        if not results:
            return None, {}

        if not self.accept_failures and failures:
            return None, {}
        
        weight_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for client, fit_res in results
        ]
        if self.round == 0:
            adaptive_step = 1.0
        else:
            global NUM_ROUNDS
            adaptive_step = 1.0 - (self.round / NUM_ROUNDS)

        self.round += 1
        
        weights_avg = aggregate_with_lowloss(results,weight_results,adaptive_step)

        weights_avg = ndarrays_to_parameters(weights_avg)
        
        glb_dir = self.base_work_dir
        os.makedirs(glb_dir, exist_ok=True)
        
        if weights_avg is not None:
            if type(weights_avg) == type([]):
                save_model_weights(server_round, weights_avg)
            else:
                weights_avg: List[np.ndarray] = flwr.common.parameters_to_ndarrays(weights_avg)
                save_model_weights(server_round, weights_avg)
            
            weights_avg = ndarrays_to_parameters(weights_avg)

        global RND
        RND += 1

        return weights_avg, {}
   

if DEVICE.type == "cuda":
    backend_config = {"client_resources": {"num_gpus": 1,"num_cpus": 1},}

flwr.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_PARTITIONS,
        config=ServerConfig(num_rounds=NUM_ROUNDS,round_timeout=3600*24),
        client_resources={"num_gpus": 1,"num_cpus": 1},
        strategy=SCAFFOLDStrategy(
        fraction_fit=0.001,          
        min_fit_clients=NUM_PARTICIPATION,         
        min_available_clients=NUM_PARTICIPATION,
        fraction_evaluate = 0.1,
        initial_parameters=ndarrays_to_parameters(params))
)


