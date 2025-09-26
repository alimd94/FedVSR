
import torch
from os import path as osp
from torch.amp import GradScaler
from torch.cuda.amp import autocast
import archs  # noqa F401
import model  # noqa F401
from basicsr.data import build_dataset
from basicsr.models import build_model
import torch.nn as nn

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
import pywt
from torch.autograd import Function

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


class DWTFunction_3D(Function):
    @staticmethod
    def forward(ctx, input,
                matrix_Low_0, matrix_Low_1, matrix_Low_2,
                matrix_High_0, matrix_High_1, matrix_High_2):
        ctx.save_for_backward(matrix_Low_0, matrix_Low_1, matrix_Low_2,
                              matrix_High_0, matrix_High_1, matrix_High_2)
        L = torch.matmul(matrix_Low_0, input)
        H = torch.matmul(matrix_High_0, input)
        LL = torch.matmul(L, matrix_Low_1).transpose(dim0=2, dim1=3)
        LH = torch.matmul(L, matrix_High_1).transpose(dim0=2, dim1=3)
        HL = torch.matmul(H, matrix_Low_1).transpose(dim0=2, dim1=3)
        HH = torch.matmul(H, matrix_High_1).transpose(dim0=2, dim1=3)
        LLL = torch.matmul(matrix_Low_2, LL).transpose(dim0=2, dim1=3)
        LLH = torch.matmul(matrix_Low_2, LH).transpose(dim0=2, dim1=3)
        LHL = torch.matmul(matrix_Low_2, HL).transpose(dim0=2, dim1=3)
        LHH = torch.matmul(matrix_Low_2, HH).transpose(dim0=2, dim1=3)
        HLL = torch.matmul(matrix_High_2, LL).transpose(dim0=2, dim1=3)
        HLH = torch.matmul(matrix_High_2, LH).transpose(dim0=2, dim1=3)
        HHL = torch.matmul(matrix_High_2, HL).transpose(dim0=2, dim1=3)
        HHH = torch.matmul(matrix_High_2, HH).transpose(dim0=2, dim1=3)
        return LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH

    @staticmethod
    def backward(ctx, grad_LLL, grad_LLH, grad_LHL, grad_LHH,
                 grad_HLL, grad_HLH, grad_HHL, grad_HHH):
        matrix_Low_0, matrix_Low_1, matrix_Low_2, matrix_High_0, matrix_High_1, matrix_High_2 = ctx.saved_variables
        grad_LL = torch.add(torch.matmul(matrix_Low_2.t(), grad_LLL.transpose(dim0=2, dim1=3)), torch.matmul(
            matrix_High_2.t(), grad_HLL.transpose(dim0=2, dim1=3))).transpose(dim0=2, dim1=3)
        grad_LH = torch.add(torch.matmul(matrix_Low_2.t(), grad_LLH.transpose(dim0=2, dim1=3)), torch.matmul(
            matrix_High_2.t(), grad_HLH.transpose(dim0=2, dim1=3))).transpose(dim0=2, dim1=3)
        grad_HL = torch.add(torch.matmul(matrix_Low_2.t(), grad_LHL.transpose(dim0=2, dim1=3)), torch.matmul(
            matrix_High_2.t(), grad_HHL.transpose(dim0=2, dim1=3))).transpose(dim0=2, dim1=3)
        grad_HH = torch.add(torch.matmul(matrix_Low_2.t(), grad_LHH.transpose(dim0=2, dim1=3)), torch.matmul(
            matrix_High_2.t(), grad_HHH.transpose(dim0=2, dim1=3))).transpose(dim0=2, dim1=3)
        grad_L = torch.add(torch.matmul(grad_LL, matrix_Low_1.t()),
                           torch.matmul(grad_LH, matrix_High_1.t()))
        grad_H = torch.add(torch.matmul(grad_HL, matrix_Low_1.t()),
                           torch.matmul(grad_HH, matrix_High_1.t()))
        grad_input = torch.add(torch.matmul(
            matrix_Low_0.t(), grad_L), torch.matmul(matrix_High_0.t(), grad_H))
        return grad_input, None, None, None, None, None, None, None, None

class DWT_3D(nn.Module):
    """
    input: the 3D data to be decomposed -- (N, C, D, H, W)
    output: lfc -- (N, C, D/2, H/2, W/2)
            hfc_llh -- (N, C, D/2, H/2, W/2)
            hfc_lhl -- (N, C, D/2, H/2, W/2)
            hfc_lhh -- (N, C, D/2, H/2, W/2)
            hfc_hll -- (N, C, D/2, H/2, W/2)
            hfc_hlh -- (N, C, D/2, H/2, W/2)
            hfc_hhl -- (N, C, D/2, H/2, W/2)
            hfc_hhh -- (N, C, D/2, H/2, W/2)
    """

    def __init__(self, wavename):
        """
        3D discrete wavelet transform (DWT) for 3D data decomposition
        :param wavename: pywt.wavelist(); in the paper, 'chx.y' denotes 'biorx.y'.
        """
        super(DWT_3D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.rec_lo
        self.band_high = wavelet.rec_hi
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        """
        generating the matrices: \mathcal{L}, \mathcal{H}
        :return: self.matrix_low = \mathcal{L}, self.matrix_high = \mathcal{H}
        """
        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros((L, L1 + self.band_length - 2))
        matrix_g = np.zeros((L1 - L, L1 + self.band_length - 2))
        end = None if self.band_length_half == 1 else (
            - self.band_length_half + 1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index + j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(
            self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(
            self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]
        matrix_h_2 = matrix_h[0:(math.floor(
            self.input_depth / 2)), 0:(self.input_depth + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index + j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(self.input_height - math.floor(
            self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(self.input_width - math.floor(
            self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]
        matrix_g_2 = matrix_g[0:(self.input_depth - math.floor(
            self.input_depth / 2)), 0:(self.input_depth + self.band_length - 2)]

        matrix_h_0 = matrix_h_0[:, (self.band_length_half - 1):end]
        matrix_h_1 = matrix_h_1[:, (self.band_length_half - 1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_h_2 = matrix_h_2[:, (self.band_length_half - 1):end]

        matrix_g_0 = matrix_g_0[:, (self.band_length_half - 1):end]
        matrix_g_1 = matrix_g_1[:, (self.band_length_half - 1):end]
        matrix_g_1 = np.transpose(matrix_g_1)
        matrix_g_2 = matrix_g_2[:, (self.band_length_half - 1):end]
        if torch.cuda.is_available():
            self.matrix_low_0 = torch.Tensor(matrix_h_0).cuda()
            self.matrix_low_1 = torch.Tensor(matrix_h_1).cuda()
            self.matrix_low_2 = torch.Tensor(matrix_h_2).cuda()
            self.matrix_high_0 = torch.Tensor(matrix_g_0).cuda()
            self.matrix_high_1 = torch.Tensor(matrix_g_1).cuda()
            self.matrix_high_2 = torch.Tensor(matrix_g_2).cuda()
        else:
            self.matrix_low_0 = torch.Tensor(matrix_h_0)
            self.matrix_low_1 = torch.Tensor(matrix_h_1)
            self.matrix_low_2 = torch.Tensor(matrix_h_2)
            self.matrix_high_0 = torch.Tensor(matrix_g_0)
            self.matrix_high_1 = torch.Tensor(matrix_g_1)
            self.matrix_high_2 = torch.Tensor(matrix_g_2)

    def forward(self, input):
        """
        :param input: the 3D data to be decomposed
        :return: the eight components of the input data, one low-frequency and seven high-frequency components
        """
        assert len(input.size()) == 5
        self.input_depth = input.size()[-3]
        self.input_height = input.size()[-2]
        self.input_width = input.size()[-1]
        self.get_matrix()
        return DWTFunction_3D.apply(input, self.matrix_low_0, self.matrix_low_1, self.matrix_low_2,
                                    self.matrix_high_0, self.matrix_high_1, self.matrix_high_2)

class HighFreqLoss(nn.Module):

    def __init__(self, eps=1e-9):
        super(HighFreqLoss, self).__init__()
        self.eps = eps
        self.dwt = DWT_3D('haar')

    def forward(self, x, y,alpha=1,penalty=0.1):
      
        b,t, c, h, w = x.size()
        x = torch.transpose(x, 1, 2)
        y = torch.transpose(y, 1, 2)

        dwt_x = self.dwt(x)
        dwt_x_cont = torch.cat(dwt_x[1:], dim=1)
        dwt_y = self.dwt(y)
        dwt_y_cont = torch.cat(dwt_y[1:], dim=1)
        diff = dwt_y_cont - dwt_x_cont
        loss_hf = torch.mean(torch.sqrt((diff * diff) + self.eps))

        return alpha*loss_hf

HiFreLoss = HighFreqLoss()

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.net_g.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
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

        loss = train_with_newLoss(self.net, self.trainloader,self.scaler,1)

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

def aggregate_with_hellinger_distance(results, weight_results, adaptive_step=1.0):
    losses = [fit_res.metrics["loss"] for _, fit_res in results]
    print("Client losses:", losses)
    n_clients = len(losses)
    
    # Uniform distribution
    uniform_weights = np.array([1.0/n_clients] * n_clients)
    
    # Loss-based distribution  
    inverse_losses = np.array([1/loss for loss in losses])
    adjusted_inverse_losses = inverse_losses ** adaptive_step
    loss_weights = adjusted_inverse_losses / np.sum(adjusted_inverse_losses)
    
    # Hellinger distance
    hellinger_dist = np.sqrt(0.5 * np.sum((np.sqrt(uniform_weights) - np.sqrt(loss_weights))**2))
    
    # Hellinger distance is bounded [0,1]
    # 0 = identical distributions, 1 = maximally different
    mixing_coeff = hellinger_dist
    
    # Apply soft threshold to avoid switching on tiny differences
    threshold = 0.05
    if hellinger_dist < threshold:
        mixing_coeff = 0.0
    else:
        mixing_coeff = (hellinger_dist - threshold) / (1 - threshold)
    
    # Final weights
    final_weights = (1 - mixing_coeff) * uniform_weights + mixing_coeff * loss_weights
    
    # Apply aggregation
    weighted_weights = [
        [layer * weight for layer in weights] 
        for (weights, _), weight in zip(weight_results, final_weights)
    ]
    
    weights_prime = [
        reduce(np.add, layer_updates)
        for layer_updates in zip(*weighted_weights)
    ]
    
    return weights_prime, mixing_coeff, hellinger_dist


class FedVSRStrategy(flwr.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs) 
        self.base_work_dir = SAVE_DIR
        global RND
        self.round = RND
    
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
        
        weights_avg,_,_ = aggregate_with_hellinger_distance(results,weight_results,adaptive_step)

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
        strategy=FedVSRStrategy(
        fraction_fit=0.1,          
        min_fit_clients=NUM_PARTICIPATION,         
        min_available_clients=NUM_PARTICIPATION,
        fraction_evaluate = 0.1,
        initial_parameters=ndarrays_to_parameters(params))
)


