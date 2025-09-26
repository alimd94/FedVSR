import datetime
import logging
import math
import time
import torch
from os import path as osp
from torch.amp import GradScaler
from torch.cuda.amp import autocast
import archs  # noqa F401
import model  # noqa F401
from basicsr.data import build_dataloader, build_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import build_model

from basicsr.utils.dist_util import get_dist_info
from basicsr.utils.options import ordered_yaml
from basicsr.utils import set_random_seed

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.server import ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.common import ndarrays_to_parameters, NDArrays, Scalar, Context
from basicsr.utils import (AvgTimer, MessageLogger, check_resume, get_env_info, get_root_logger, get_time_str,
                           init_tb_logger, init_wandb_logger, make_exp_dirs, mkdir_and_rename, scandir)
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
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from recurrent_mix_precision_train import load_resume_state
import copy
from functools import reduce


DEVICE = torch.device("cuda:0")  # Try "cuda" to train on GPU
torch.backends.cudnn.benchmark = True
# print(f"Training on {DEVICE}")
# print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")

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

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.net_g.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.net_g.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.net_g.load_state_dict(state_dict, strict=True)

params = get_parameters(build_model(opt))
c_global = build_model(opt)

total_delta = copy.deepcopy(c_global.net_g.state_dict())
for j,key in enumerate(total_delta.keys()):

    total_delta[key] = torch.zeros_like(total_delta[key], dtype=torch.float32)

def train_with_control_variates(net, trainloader,scaler, epochs,c_local,c_global):
    current_step = 0 

    c_global_para = c_global.net_g.state_dict()
    c_local_para = c_local.net_g.state_dict()
    global_model_para = copy.deepcopy(net.net_g.state_dict())

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
                    l_pix = net.cri_pix(net.output, net.gt)
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
                scaler.scale(l_total).backward()
                scaler.step(net.optimizer_g)
                scaler.update()

                net.log_dict = net.reduce_loss_dict(loss_dict)
            if net.ema_decay > 0:
                net.model_ema(decay=net.ema_decay)            
            net_para = net.net_g.state_dict()
            for key in net_para:
                net_para[key] = net_para[key] - opt['train']['optim_g']['lr'] * (c_global_para[key] - c_local_para[key])
            net.net_g.load_state_dict(net_para)

    c_new_para = c_local.net_g.state_dict()
    c_delta_para = copy.deepcopy(c_local.net_g.state_dict())
    net_para = net.net_g.state_dict()
    for key in net_para:
        c_new_para[key] = c_new_para[key] - c_global_para[key] + (global_model_para[key] - net_para[key]) / (current_step * opt['train']['optim_g']['lr'])
        c_delta_para[key] = c_new_para[key] - c_local_para[key]

    c_local.net_g.load_state_dict(c_new_para)

    return c_delta_para,c_local
    
class FlowerClient(NumPyClient):
    def __init__(self, partition_id, opt, trainloader,c_global=c_global,total_delta=total_delta):
        import archs  # noqa F401
        import model  # noqa F401
        self.partition_id = partition_id
        self.net = build_model(opt)
        self.scaler = GradScaler()
        self.trainloader = trainloader[int(self.partition_id)]
        self.c_local = copy.deepcopy(self.net)
        self.c_global = c_global
        self.total_delta = total_delta

    def get_parameters(self,):
        return get_parameters(self.net)

    def fit(self, parameters,config):
        set_parameters(self.net, parameters)

        c_delta_para, c_local = train_with_control_variates(self.net, self.trainloader,self.scaler,1,self.c_local,self.c_global)
        self.c_local = c_local

        for j,key in enumerate(self.total_delta):
            self.total_delta[key] += c_delta_para[key]

        return get_parameters(self.net), len(self.trainloader), {}, 

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
    net.save1(round_number)

class SCAFFOLDStrategy(flwr.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs) 
        self.c_global= build_model(opt)
        self.total_delta = total_delta
    
    def aggregate_fit(self,server_round,results,failures):

        if not results:
            return None, {}

        if not self.accept_failures and failures:
            return None, {}
        
        weight_results = [
            (flwr.common.parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for client, fit_res in results
        ]
        weights_avg = aggregate(weight_results) 
        
        weights_avg = ndarrays_to_parameters(weights_avg)
                
        if weights_avg is not None:
            if type(weights_avg) == type([]):
                save_model_weights(server_round, weights_avg,)
            else:
                weights_avg: List[np.ndarray] = flwr.common.parameters_to_ndarrays(weights_avg)
                save_model_weights(server_round, weights_avg,)
            
            weights_avg = ndarrays_to_parameters(weights_avg)

        for key in self.total_delta:
            self.total_delta[key] /= NUM_PARTITIONS

        c_global_para = self.c_global.net_g.state_dict()

        for key in c_global_para:
            
            if c_global_para[key].type() == 'torch.LongTensor':
                c_global_para[key] += self.total_delta[key].type(torch.LongTensor)
            elif c_global_para[key].type() == 'torch.cuda.LongTensor':
                c_global_para[key] += self.total_delta[key].type(torch.cuda.LongTensor)
            else:
                c_global_para[key] += self.total_delta[key]

        self.c_global.net_g.load_state_dict(c_global_para)
                
        return weights_avg, {}

def aggregate(results):
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime

if DEVICE.type == "cuda":
    backend_config = {"client_resources": {"num_gpus": 1,"num_cpus": 1},}

flwr.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_PARTITIONS,
        config=ServerConfig(num_rounds=100,round_timeout=3600*24),
        client_resources={"num_gpus": 1,"num_cpus": 1},
        strategy=SCAFFOLDStrategy(
        fraction_fit=0.001,          
        min_fit_clients=4,         
        min_available_clients=4,
        fraction_evaluate = 0.1,
        initial_parameters=ndarrays_to_parameters(params))
)


