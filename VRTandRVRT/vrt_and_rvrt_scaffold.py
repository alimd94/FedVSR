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
import re
from flwr.common import (
    FitRes,
    NDArrays,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
)

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
SAVE_DIR = "RVRT_SCAFFOLD" # "VRT_SCAFFOLD"
NUM_ROUNDS = 100
NUM_PARTICIPATION = 4


seed = 123
if seed is None:
    seed = random.randint(1, 10000)
print('Random seed: {}'.format(seed))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.netG.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.netG.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.netG.load_state_dict(state_dict, strict=True)

params = get_parameters(define_Model(opt))
c_global = define_Model(opt)
c_global.init_train()

total_delta = copy.deepcopy(c_global.netG.state_dict())
for j,key in enumerate(total_delta.keys()):

    total_delta[key] = torch.zeros_like(total_delta[key], dtype=torch.float32)


def train(net, trainloader, epochs: int):
    current_step = 0 
    loss = 0.0
    for epoch in range(epochs):
        for i, train_data in enumerate(trainloader):
            
            current_step += 1

            net.update_learning_rate(current_step)

            net.feed_data(train_data)

            net.optimize_parameters(current_step)
            loss += net.log_dict['G_loss']
            
    return loss

def train_with_control_variates(net, trainloader, epochs,c_local,c_global):
    current_step = 0 

    c_global_para = c_global.netG.state_dict()
    c_local_para = c_local.netG.state_dict()
    global_model_para = copy.deepcopy(net.netG.state_dict())

    for epoch in range(epochs):
        for i, train_data in enumerate(trainloader):
            current_step += 1
            net.update_learning_rate(current_step)
            net.feed_data(train_data)

            net.G_optimizer.zero_grad()
            net.netG_forward()
            G_loss = net.G_lossfn_weight * net.G_lossfn(net.E, net.H)

            G_loss.backward()

            G_optimizer_clipgrad = net.opt_train['G_optimizer_clipgrad'] if net.opt_train['G_optimizer_clipgrad'] else 0
            if G_optimizer_clipgrad > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=net.opt_train['G_optimizer_clipgrad'], norm_type=2)

            net.G_optimizer.step()

            net_para = net.netG.state_dict()
            for key in net_para:
                net_para[key] = net_para[key] - net.opt_train['G_optimizer_lr'] * (c_global_para[key] - c_local_para[key])

            net.netG.load_state_dict(net_para)


            G_regularizer_orthstep = net.opt_train['G_regularizer_orthstep'] if net.opt_train['G_regularizer_orthstep'] else 0
            if G_regularizer_orthstep > 0 and current_step % G_regularizer_orthstep == 0 and current_step % net.opt['train']['checkpoint_save'] != 0:
                net.netG.apply(regularizer_orth)
            G_regularizer_clipstep = net.opt_train['G_regularizer_clipstep'] if net.opt_train['G_regularizer_clipstep'] else 0
            if G_regularizer_clipstep > 0 and current_step % G_regularizer_clipstep == 0 and current_step % net.opt['train']['checkpoint_save'] != 0:
                net.netG.apply(regularizer_clip)

            net.log_dict['G_loss'] = G_loss.item()

            if net.opt_train['E_decay'] > 0:
                net.update_E(net.opt_train['E_decay'])

    
    c_new_para = c_local.netG.state_dict()
    c_delta_para = copy.deepcopy(c_local.netG.state_dict())
    net_para = net.netG.state_dict()
    for key in net_para:
        c_new_para[key] = c_new_para[key] - c_global_para[key] + (global_model_para[key] - net_para[key]) / (current_step * net.opt_train['G_optimizer_lr'])
        c_delta_para[key] = c_new_para[key] - c_local_para[key]

    c_local.netG.load_state_dict(c_new_para)

    return c_delta_para,c_local


class FlowerClient(NumPyClient):
    def __init__(self, partition_id, opt, trainloader,c_global=c_global,total_delta=total_delta):
        self.partition_id = partition_id
        self.net = define_Model(opt)
        self.net.init_train()
        self.trainloader = trainloader[int(self.partition_id)]
        self.c_local = copy.deepcopy(self.net)
        self.c_global = c_global
        self.total_delta = total_delta

    def get_parameters(self,):
        return get_parameters(self.net)

    def fit(self, parameters,config):
        set_parameters(self.net, parameters)

        c_delta_para, c_local =train_with_control_variates(self.net, self.trainloader,1,self.c_local,self.c_global)
        self.c_local = c_local

        for j,key in enumerate(self.total_delta):
            self.total_delta[key] += c_delta_para[key]


        return get_parameters(self.net), len(self.trainloader), {}, 
    

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
         num_clients=num_partitions, batch_size=BATCH_SIZE,balance=True,iid=True,#alpha=0.1
    )

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


class SCAFFOLDStrategy(flwr.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs) 

        self.c_global= define_Model(opt)
        self.c_global.init_train()
        self.total_delta = total_delta
        self.base_work_dir = SAVE_DIR
    
    def aggregate_fit(self,server_round,results,failures):

        if not results:
            return None, {}

        if not self.accept_failures and failures:
            return None, {}
        
        weight_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for client, fit_res in results
        ]
        weights_avg = aggregate(weight_results) 
        
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


        for key in self.total_delta:
            self.total_delta[key] /= NUM_PARTITIONS

        c_global_para = self.c_global.netG.state_dict()

        for key in c_global_para:
            
            if c_global_para[key].type() == 'torch.LongTensor':
                c_global_para[key] += self.total_delta[key].type(torch.LongTensor)
            elif c_global_para[key].type() == 'torch.cuda.LongTensor':
                c_global_para[key] += self.total_delta[key].type(torch.cuda.LongTensor)
            else:
                c_global_para[key] += self.total_delta[key]

        self.c_global.netG.load_state_dict(c_global_para)
                
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
        config=ServerConfig(num_rounds=NUM_ROUNDS,round_timeout=3600*24),
        client_resources={"num_gpus": 1,"num_cpus": 1},
        strategy=SCAFFOLDStrategy(
        fraction_fit=0.001,          
        min_fit_clients=NUM_PARTICIPATION,         
        min_available_clients=NUM_PARTICIPATION,
        fraction_evaluate = 0.1,
        initial_parameters=ndarrays_to_parameters(params))
)


