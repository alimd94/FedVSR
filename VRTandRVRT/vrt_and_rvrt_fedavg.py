from collections import OrderedDict
from typing import List
import numpy as np
import torch
import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.server import ServerConfig
from flwr.common import ndarrays_to_parameters, NDArrays, Scalar, Context
import os.path
import math
import argparse
import random
import numpy as np
from collections import OrderedDict
import torch
from utils import utils_image as util
from utils import utils_option as option
from data.select_dataset import define_Dataset
from models.select_model import define_Model
from models.model_vrt import ModelVRT as M
from dataset import load_datasets
import os
from functools import reduce
from flwr.common import (
    NDArrays,
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
BATCH_SIZE = 4
NUM_ROUNDS = 100
NUM_PARTICIPATION = 4
SAVE_DIR = "RVRT_FedAVG"

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

def train(net, trainloader, epochs: int):
    current_step = 0 
    loss = 0.0
    for epoch in range(epochs):
        for i, train_data in enumerate(trainloader):
            
            current_step += 1

            net.update_learning_rate(current_step)

            net.feed_data(train_data,False)

            net.optimize_parameters(current_step)
 

            
class FlowerClient(NumPyClient):
    def __init__(self, partition_id, opt, trainloader):
        self.partition_id = partition_id
        self.net = define_Model(opt)
        self.net.init_train()
        self.trainloader = trainloader[int(self.partition_id)]

    def get_parameters(self,):
        return get_parameters(self.net)

    def fit(self, parameters,config):
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=1)

        return get_parameters(self.net), len(self.trainloader), {}#, metrics
    

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

    return FlowerClient(partition_id, opt, trainloaders,).to_client()


# Create the ClientApp
client = ClientApp(client_fn=client_fn)



def save_model_weights(round_number, parameters,save_dir,opt=opt):
    net = define_Model(opt)

    params_dict = zip(net.netG.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.netG.load_state_dict(state_dict, strict=True)

    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"round_{round_number}_weights.pth")
    torch.save(net.netG.state_dict(), file_path)


class SaveModelStrategy(flwr.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs) 
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

        weights_avg = aggregate_avg(weight_results)
            
        
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

        return weights_avg, {}


def aggregate_avg(results):
    """Compute weighted average."""
    num_examples_total = sum([num_examples for _, num_examples in results])
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
        strategy=SaveModelStrategy(
        # min_fit_clients=NUM_PARTITIONS,
        # min_available_clients=4,
        fraction_fit=0.001,          
        min_fit_clients=NUM_PARTICIPATION,         
        min_available_clients=NUM_PARTICIPATION,
        fraction_evaluate = 0.1,
        # proximal_mu = np.inf,
        initial_parameters=ndarrays_to_parameters(params),)
)
