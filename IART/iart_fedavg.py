import datetime
import logging
import math
import time
import torch
from os import path as osp
from torch.amp import GradScaler

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
from logging import INFO, DEBUG
from flwr.common.logger import log

from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from typing import Optional, Union
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
)
from recurrent_mix_precision_train import load_resume_state


DEVICE = torch.device("cuda:0")  # Try "cuda" to train on GPU
torch.backends.cudnn.benchmark = True


seed = 123
if seed is None:
    seed = random.randint(1, 10000)
print('Random seed: {}'.format(seed))
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
    # print(type(parameters),"TYPEP")
    params_dict = zip(net.net_g.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.net_g.load_state_dict(state_dict, strict=True)

params = get_parameters(build_model(opt))


def train(net, trainloader,scaler, epochs: int):
    current_step = 0 
    loss = 0.0
    for epoch in range(epochs):
        for i, train_data in enumerate(trainloader):
            current_step += 1
            net.update_learning_rate(current_step, warmup_iter=opt['train'].get('warmup_iter', -1))
            # training
            net.feed_data(train_data)
            net.optimize_parameters(scaler, current_step)


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
        train(self.net, self.trainloader,self.scaler, epochs=1)
        return get_parameters(self.net), len(self.trainloader), {}#, metrics

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

    # trainloader,  = load_datasets(partition_id, num_partitions)
    return FlowerClient(partition_id, opt, trainloaders,).to_client()


# Create the ClientApp
client = ClientApp(client_fn=client_fn)


def save_model_weights(round_number, parameters,opt=opt):
    net = build_model(opt)
    params_dict = zip(net.net_g.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.net_g.load_state_dict(state_dict, strict=True)
    net.save(1,round_number)

class SaveModelStrategy(flwr.server.strategy.FedAvg):

    def aggregate_fit(self, rnd, results, failures):
        res = super().aggregate_fit(rnd, results, failures)

        if not results:
            return None, {}

        if not self.accept_failures and failures:
            return None, {}
        
        if res:
            aggregated_parameters, aggregated_metrics = res

            if aggregated_parameters is not None:
                aggregated_ndarrays: List[np.ndarray] = flwr.common.parameters_to_ndarrays(aggregated_parameters)
                save_model_weights(rnd, aggregated_ndarrays)

        return aggregated_parameters,aggregated_metrics


def server_fn(context: Context) -> ServerAppComponents:
    strategy = SaveModelStrategy(
        # min_fit_clients=NUM_PARTITIONS,
        # min_available_clients=4,
        fraction_fit=0.1,          
        min_fit_clients=4,         
        min_available_clients=4,
        fraction_evaluate = 0.1,
        # proximal_mu = 0.01,
        initial_parameters=ndarrays_to_parameters(
            params
        ), 

)
    config = ServerConfig(num_rounds=100,round_timeout=3600*24)
    return ServerAppComponents(strategy=strategy, config=config)


if DEVICE.type == "cuda":
    backend_config = {"client_resources": {"num_gpus": 1,"num_cpus": 1},}

flwr.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_PARTITIONS,
        config=ServerConfig(num_rounds=100,round_timeout=3600*24),
        client_resources={"num_gpus": 1,"num_cpus": 1},
        strategy=SaveModelStrategy(
        # min_fit_clients=NUM_PARTITIONS,
        # min_available_clients=4,
        fraction_fit=0.001,          
        min_fit_clients=4,         
        min_available_clients=4,
        fraction_evaluate = 0.1,
        # proximal_mu = -0.9,
        initial_parameters=ndarrays_to_parameters(params),)
)
