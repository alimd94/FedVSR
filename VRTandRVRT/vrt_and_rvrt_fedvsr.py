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
from flwr.server.strategy import FedAvg, FedAdagrad
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
from pytorch_wavelets import DWTForward


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
        # fisher = self._compute_empirical_fisher()

        # for name, param in self.net.netG.state_dict().items():
        #     if name in fisher:
        #         # fisher_info = fisher[name]
        #         fisher_info = fisher[name].to(param.data.dtype)
        #         param.data.mul_(fisher_info)
        
        # global RND
        # round_dir = f"fisher_round_{RND}"
        # os.makedirs(round_dir, exist_ok=True)  # Ensure directory exists

        # file_path = os.path.join(round_dir, f"fisher_info_{self.partition_id}.pkl")
        # with open(file_path, "wb") as f:
        #     pickle.dump(get_parameters1(fisher), f) 
        
        return get_parameters(self.net), len(self.trainloader), {'loss': loss}, 
    

    def evaluate(self, parameters, config):
        return .1, 1, {"accuracy": 0}
    

    def _compute_fisher_single_batch(self,train_data):

        self.net.G_optimizer.zero_grad()
        self.net.feed_data(train_data)
        
        self.net.netG_forward()
        loss = self.net.G_lossfn(self.net.E , self.net.H)
        # loss = loss.clone()
        loss.backward()
        
        fisher_dict = {name: torch.zeros_like(param) + 1e-12 for name, param in self.net.netG.state_dict().items()}
        for name, param in self.net.netG.named_parameters():
            if param.grad is not None:
                fisher_dict[name] += param.grad.pow(2).detach()
                
        return fisher_dict
    
    def _compute_empirical_fisher(self):
        
        self.net.netG.eval()
        accumulated_fisher = {}
        n_processed = 0
        
        for _, train_data in enumerate(self.trainloader):
            batch_fisher = self._compute_fisher_single_batch(train_data)

            # Accumulate Fisher values
            for name, value in batch_fisher.items():
                if name not in accumulated_fisher:
                    accumulated_fisher[name] = value
                else:
                    accumulated_fisher[name] += value
            
            n_processed += 1
        
        # Average over batches
        for name in accumulated_fisher:
            accumulated_fisher[name] /= n_processed
        
        self.net.netG.train()

        return accumulated_fisher

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
        
        weights_avg = aggregate_with_lowloss(results,weight_results,adaptive_step)

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
        strategy=SCAFFOLDStrategy(
        fraction_fit=0.001,          
        min_fit_clients=4,         
        min_available_clients=4,
        fraction_evaluate = 0.1,
        initial_parameters=ndarrays_to_parameters(params))
)
