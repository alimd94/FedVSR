from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
from flwr.common import ndarrays_to_parameters, NDArrays, Scalar, Context
from flwr_datasets import FederatedDataset
from collections import defaultdict
import sys
import os.path
import math
import argparse
import time
import random
import cv2
import numpy as np
from collections import OrderedDict
import logging
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
# from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from data.select_dataset import define_Dataset
from models.select_model import define_Model
from models.model_vrt import ModelVRT as M
from dataset import load_datasets
import os
from logging import INFO, DEBUG
from flwr.common.logger import log
import copy
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from typing import Optional, Union
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
)
import pickle
from utils.utils_regularizers import regularizer_orth, regularizer_clip
from math import exp

from functools import reduce
# import mmcv
import re
from flwr.common import (
    FitRes,
    NDArrays,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
)
import pywt
from torch.autograd import Function


DEVICE = torch.device("cuda:0")  


flwr.common.logger.configure(identifier="myFlowerExperiment", filename="log.txt")

# json_path='options/vrt/002_train_vrt_videosr_bi_reds_16frames.json'
json_path='options/rvrt/001_train_rvrt_videosr_bi_reds_30frames.json'


parser = argparse.ArgumentParser()
parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')

opt = option.parse(parser.parse_args().opt, is_train=True)

util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G',
                                                        pretrained_path=opt['path']['pretrained_netG'])
init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E',
                                                        pretrained_path=opt['path']['pretrained_netE'])
opt['path']['pretrained_netG'] = init_path_G
opt['path']['pretrained_netE'] = init_path_E
init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'],
                                                                            net_type='optimizerG')
opt['path']['pretrained_optimizerG'] = init_path_optimizerG
current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)

option.save(opt)

opt = option.dict_to_nonedict(opt)

NUM_PARTITIONS = 40
BATCH_SIZE = 1
RND = 1


seed = 123
if seed is None:
    seed = random.randint(1, 10000)
print('Random seed: {}'.format(seed))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


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
    return [val.cpu().numpy() for _, val in net.netG.state_dict().items()]

def get_parameters1(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.items()]


def set_parameters(net, parameters: List[np.ndarray]):
    # print(type(parameters),"TYPEP")
    params_dict = zip(net.netG.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.netG.load_state_dict(state_dict, strict=True)

params = get_parameters(define_Model(opt))

def train_with_newLoss(net, trainloader, epochs,client_id):
    current_step = 0 
    loss = 0.0
    for epoch in range(epochs):
        for i, train_data in enumerate(trainloader):
            current_step += 1
            net.update_learning_rate(current_step)
            net.feed_data(train_data)

            net.G_optimizer.zero_grad()
            net.netG_forward()
            G_loss = net.G_lossfn(net.E, net.H) + HiFreLoss(net.E, net.H)

            G_loss.backward()

            G_optimizer_clipgrad = net.opt_train['G_optimizer_clipgrad'] if net.opt_train['G_optimizer_clipgrad'] else 0
            if G_optimizer_clipgrad > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=net.opt_train['G_optimizer_clipgrad'], norm_type=2)

            net.G_optimizer.step()

            G_regularizer_orthstep = net.opt_train['G_regularizer_orthstep'] if net.opt_train['G_regularizer_orthstep'] else 0
            if G_regularizer_orthstep > 0 and current_step % G_regularizer_orthstep == 0 and current_step % net.opt['train']['checkpoint_save'] != 0:
                net.netG.apply(regularizer_orth)
            G_regularizer_clipstep = net.opt_train['G_regularizer_clipstep'] if net.opt_train['G_regularizer_clipstep'] else 0
            if G_regularizer_clipstep > 0 and current_step % G_regularizer_clipstep == 0 and current_step % net.opt['train']['checkpoint_save'] != 0:
                net.netG.apply(regularizer_clip)

            net.log_dict['G_loss'] = G_loss.item()
            loss += net.log_dict['G_loss']

            if net.opt_train['E_decay'] > 0:
                net.update_E(net.opt_train['E_decay'])
    return loss / current_step
            

class FlowerClient(NumPyClient):
    def __init__(self, partition_id, opt, trainloader):
        self.partition_id = partition_id
        self.net = define_Model(opt)
        self.net.init_train()
        self.trainloader = trainloader[int(self.partition_id)]
        # self.fisher_collector = FISHER_COLLECTOR

    def get_parameters(self,):
        return get_parameters(self.net)

    def fit(self, parameters,config):
        set_parameters(self.net, parameters)

        loss = train_with_newLoss(self.net, self.trainloader,1,self.partition_id)
        
        return get_parameters(self.net), len(self.trainloader), {'loss': loss}, 
    

    def evaluate(self, parameters, config):
        return .1, 1, {"accuracy": 0}
    

def client_fn(context: Context) -> Client:
    dataset_opt = opt['datasets']['train']
    train_set = define_Dataset(dataset_opt)
    train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))

    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    trainloaders, _, _ = load_datasets(train_set, train_set,
         num_clients=num_partitions, batch_size=BATCH_SIZE,balance=True,iid=True,#alpha=0.1,name='kinetics'
    )

    # trainloader,  = load_datasets(partition_id, num_partitions)
    return FlowerClient(partition_id, opt, trainloaders).to_client()

client = ClientApp(client_fn=client_fn)


def save_model_weights(round_number, parameters,save_dir,opt=opt):
    net = define_Model(opt)
    params_dict = zip(net.netG.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.netG.load_state_dict(state_dict, strict=True)
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"round_{round_number}_weights.pth")
    torch.save(net.netG.state_dict(), file_path)


class FedVSRStrategy(flwr.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs) 
        self.base_work_dir = "rvrt_kinetics_ours"
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
            global TOTAL_RND
            adaptive_step = 1.0 - (self.round / TOTAL_RND)

        self.round += 1
        
        weights_avg,_,_ = aggregate_with_hellinger_distance(results,weight_results,adaptive_step)

        weights_avg = ndarrays_to_parameters(weights_avg)
        
        glb_dir = self.base_work_dir
        os.makedirs(glb_dir, exist_ok=True)
        
        if weights_avg is not None:
            if type(weights_avg) == type([]):
                save_model_weights(server_round, weights_avg, glb_dir)
            else:
                weights_avg: List[np.ndarray] = flwr.common.parameters_to_ndarrays(weights_avg)
                save_model_weights(server_round, weights_avg, glb_dir)
            
            weights_avg = ndarrays_to_parameters(weights_avg)

        global RND
        RND += 1

        return weights_avg, {}


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

def aggregate_with_fisher(results):
    """Compute weighted average by summing Fisher information and dividing model weights by this sum."""
    global RND
    all_fisher = []
    directory = f"fisher_round_{RND}"

    for filename in os.listdir(directory):
        if filename.endswith(".pkl"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "rb") as f:
                all_fisher.append(pickle.load(f))

    if not all_fisher:
        raise ValueError("No Fisher information collected.")
    

    fisher_sum = [np.zeros_like(layer) for layer in all_fisher[0]]

    for fisher_list in all_fisher:
        for i, fisher_layer in enumerate(fisher_list):
            fisher_sum[i] += np.asarray(fisher_layer) 
    
    weight_results = [parameters_to_ndarrays(fit_res.parameters)
        for _, fit_res in results
    ]

    if len(weight_results[0]) != len(fisher_sum):
            raise ValueError(
                f"Dimension mismatch: weights have {len(weight_results[0])} layers "
                f"but Fisher information has {len(fisher_sum)} layers"
            )
    
    summed_weights = [np.zeros_like(layer) for layer in weight_results[0]]
    for model_weights in weight_results:
        for i, layer_weights in enumerate(model_weights):
            summed_weights[i] += np.asarray(layer_weights)

    weights_prime = [
        layer_weights / fisher_sum[i]   # Prevent division by zero
        for i, layer_weights in enumerate(summed_weights)
    ]
    return weights_prime

if DEVICE.type == "cuda":
    backend_config = {"client_resources": {"num_gpus": 1,"num_cpus": 1},}

flwr.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_PARTITIONS,
        config=ServerConfig(num_rounds=40,round_timeout=3600*24),
        client_resources={"num_gpus": 1,"num_cpus": 1},
        strategy=FedVSRStrategy(
        fraction_fit=0.001,          
        min_fit_clients=4,         
        min_available_clients=4,
        fraction_evaluate = 0.1,
        initial_parameters=ndarrays_to_parameters(params))
)
