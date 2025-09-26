# As the original paper did not provide an implementation, we reproduced the method ourselves and added it as a baseline in our evaluation.

# FedSR: Federated Learning for Image Super-Resolution via detail-assisted contrastive learning,
# Knowledge-Based Systems,
# Volume 309,
# 2025,
# 112778,
# ISSN 0950-7051,
# https://doi.org/10.1016/j.knosys.2024.112778.
# (https://www.sciencedirect.com/science/article/pii/S0950705124014126)


from collections import OrderedDict
from typing import List, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.server import ServerConfig
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, NDArrays, Scalar, Context
import os.path
import math
import argparse
import random
from collections import OrderedDict
from utils import utils_image as util
from utils import utils_option as option
from data.select_dataset import define_Dataset
from models.select_model import define_Model
from models.model_vrt import ModelVRT as M
from dataset import load_datasets
import os
from functools import reduce
from flwr.common import NDArrays, parameters_to_ndarrays

import time
import torch.cuda
torch.cuda.empty_cache()
DEVICE = torch.device("cuda:0")  
flwr.common.logger.configure(identifier="myFlowerExperiment", filename="log.txt")
json_path='options/vrt/002_train_vrt_videosr_bi_reds_16frames.json'
# json_path='options/rvrt/001_train_rvrt_videosr_bi_reds_30frames.json'
parser = argparse.ArgumentParser()
parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
opt = option.parse(parser.parse_args().opt, is_train=True)
util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))
init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G',pretrained_path=opt['path']['pretrained_netG'])
init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E',pretrained_path=opt['path']['pretrained_netE'])
opt['path']['pretrained_netG'] = init_path_G
opt['path']['pretrained_netE'] = init_path_E
init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'],net_type='optimizerG')
opt['path']['pretrained_optimizerG'] = init_path_optimizerG
current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)
option.save(opt)
opt = option.dict_to_nonedict(opt)
NUM_PARTITIONS = 40 #112 rounds to meet the total number of partitions 120, 40
BATCH_SIZE = 3
RND = 0
TOTAL_RND = 100 #around 40 is extra but in case
seed = 123
if seed is None:
    seed = random.randint(1, 10000)
print('Random seed: {}'.format(seed))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Projection head for contrastive learning
class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# Modified VRT model to return shallow features
class VRTWithShallowFeatures(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.shallow_feature_extractor = base_model.netG.conv_first if hasattr(base_model.netG, 'conv_first') else None
    
    def forward(self, x):
        return self.base_model(x)
    
    def get_shallow_features(self, x):
        if self.shallow_feature_extractor is None:
            return None
        shallow_feat = self.shallow_feature_extractor(x)
        shallow_feat = F.adaptive_avg_pool3d(shallow_feat, (1, 1, 1))
        return torch.flatten(shallow_feat, 1)

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.netG.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.netG.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.netG.load_state_dict(state_dict, strict=True)

params = get_parameters(define_Model(opt))

def train_with_contrastive(net, trainloader, epochs, client_id, global_params: List[np.ndarray], 
                           prev_local_params: List[np.ndarray], projection_head, tau: float = 0.07, lambda_cl: float = 0.1):
    current_step = 0 
    loss = 0.0
    timee = 0
    start_time = time.time()
    
    # Create temporary models for global and previous local
    global_model = define_Model(opt)
    prev_local_model = define_Model(opt)
    
    set_parameters(global_model, global_params)
    set_parameters(prev_local_model, prev_local_params)
    
    wrapped_global = VRTWithShallowFeatures(global_model).to(DEVICE)
    wrapped_prev = VRTWithShallowFeatures(prev_local_model).to(DEVICE)
    wrapped_net = VRTWithShallowFeatures(net).to(DEVICE)
    
    wrapped_global.eval()
    wrapped_prev.eval()
    
    for epoch in range(epochs):
        for i, train_data in enumerate(trainloader):
            start_time = time.time()
            current_step += 1
            net.update_learning_rate(current_step)
            net.feed_data(train_data)

            net.G_optimizer.zero_grad()
            net.netG_forward()
            G_loss = net.G_lossfn(net.E, net.H) #+ net.G_lossfn2(net.E, net.H)
            
            # Get shallow features
            L = train_data['L']  # Assuming 'L' is the low-res input
            with torch.no_grad():
                shallow_global = wrapped_global.get_shallow_features(L)
                shallow_prev = wrapped_prev.get_shallow_features(L)
            shallow_current = wrapped_net.get_shallow_features(L)
            
            if shallow_current is not None and shallow_global is not None and shallow_prev is not None:
                # Project features
                v = projection_head(shallow_current)
                v_plus = projection_head(shallow_global)
                v_minus = projection_head(shallow_prev)
                
                # Normalize
                v = F.normalize(v, p=2, dim=1)
                v_plus = F.normalize(v_plus, p=2, dim=1)
                v_minus = F.normalize(v_minus, p=2, dim=1)
                
                # Contrastive loss
                pos_sim = F.cosine_similarity(v, v_plus, dim=1)
                neg_sim = F.cosine_similarity(v, v_minus, dim=1)
                
                numerator = torch.exp(pos_sim / tau)
                denominator = numerator + torch.exp(neg_sim / tau)
                loss_cl = -torch.log(numerator / denominator).mean()
                
                # Add to total loss
                G_loss += lambda_cl * loss_cl

            G_loss.backward()

            G_optimizer_clipgrad = net.opt_train['G_optimizer_clipgrad'] if net.opt_train['G_optimizer_clipgrad'] else 0
            if G_optimizer_clipgrad > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=net.opt_train['G_optimizer_clipgrad'], norm_type=2)

            net.G_optimizer.step()

            G_regularizer_orthstep = net.opt_train['G_regularizer_orthstep'] if net.opt_train['G_regularizer_orthstep'] else 0
            if G_regularizer_orthstep > 0 and current_step % G_regularizer_orthstep == 0 and current_step % net.opt['train']['checkpoint_save'] != 0:
                net.netG.apply(G_regularizer_orthstep)
            G_regularizer_clipstep = net.opt_train['G_regularizer_clipstep'] if net.opt_train['G_regularizer_clipstep'] else 0
            if G_regularizer_clipstep > 0 and current_step % G_regularizer_clipstep == 0 and current_step % net.opt['train']['checkpoint_save'] != 0:
                net.netG.apply(G_regularizer_clipstep)

            # self.log_dict['G_loss'] = G_loss.item()/self.E.size()[0]  # if `reduction='sum'`
            net.log_dict['G_loss'] = G_loss.item()
            loss += net.log_dict['G_loss']

            if net.opt_train['E_decay'] > 0:
                net.update_E(net.opt_train['E_decay'])
            end_time = time.time()
            timee += (end_time - start_time)

    return loss 

class FlowerClient(NumPyClient):
    def __init__(self, partition_id, opt, trainloader):
        self.partition_id = partition_id
        self.net = define_Model(opt)
        self.net.init_train()
        self.trainloader = trainloader[int(self.partition_id)]
        shallow_feat_dim = 64  # Adjust based on actual dim of conv_first output after flattening
        self.projection_head = ProjectionHead(shallow_feat_dim).to(DEVICE)
        self.prev_parameters = get_parameters(self.net)
    
    def get_parameters(self,):
        return get_parameters(self.net)
    
    def fit(self, parameters, config):
        set_parameters(self.net, parameters)

        global_params = parameters  # Current parameters are the global ones at start of round
        loss = train_with_contrastive(
            self.net, 
            self.trainloader, 
            epochs=1, 
            client_id=self.partition_id,
            global_params=global_params,
            prev_local_params=self.prev_parameters,
            projection_head=self.projection_head,
            tau=config.get("tau", 0.07),
            lambda_cl=config.get("lambda_cl", 0.1)
        )
        updated_params = get_parameters(self.net)
        self.prev_parameters = updated_params  # Update prev to current after training
        
        return updated_params, len(self.trainloader), {'loss': loss}
    
    def evaluate(self, parameters, config):
        return .1, 1, {"accuracy": 0}

def client_fn(context: Context) -> Client:
    dataset_opt = opt['datasets']['train']
    train_set = define_Dataset(dataset_opt)
    train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))

    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    trainloaders = load_datasets(train_set,
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

class Strategy(flwr.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs) 
        self.base_work_dir = "fedsr"
        global RND
        self.round = RND
        self.previous_global_params = None
    
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
            weights_avg = self.aggregate_avg(weight_results)
        else:
            weights_avg = self.hierarchical_aggregation(weight_results)
        
        
        self.previous_global_params = weights_avg
        
        self.round += 1

        weights_avg_params = ndarrays_to_parameters(weights_avg)
        
        glb_dir = self.base_work_dir
        os.makedirs(glb_dir, exist_ok=True)
        
        if weights_avg_params is not None and server_round % 1 == 0:
            weights_avg_nd = parameters_to_ndarrays(weights_avg_params)
            save_model_weights(server_round, weights_avg_nd, glb_dir)
            
        global RND
        RND += 1

        return weights_avg_params, {}
    
    def aggregate_avg(self, results):
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
    
    def hierarchical_aggregation(self, weight_results):
        """Hierarchical aggregation based on layer-wise similarity"""
        # Convert to numpy arrays
        client_weights = [weights for weights, _ in weight_results]
        client_examples = [num_examples for _, num_examples in weight_results]
        
        # Initialize aggregated weights
        aggregated_weights = []
        
        # For each layer, compute similarity-based aggregation
        for layer_idx in range(len(client_weights[0])):
            layer_weights = []
            layer_similarities = []
            
            # Get current layer weights from all clients
            for client_idx in range(len(client_weights)):
                layer_weights.append(client_weights[client_idx][layer_idx])
            
            # Compute similarity with previous global model
            for client_w in layer_weights:
                if self.previous_global_params[layer_idx].size > 0 and client_w.size > 0:
                    # Flatten and compute cosine similarity
                    global_flat = self.previous_global_params[layer_idx].flatten()
                    client_flat = client_w.flatten()
                    
                    # Handle potential size mismatch (though should be same)
                    min_size = min(len(global_flat), len(client_flat))
                    if len(global_flat) != len(client_flat):
                        print(f"Warning: Layer {layer_idx} size mismatch: {len(global_flat)} vs {len(client_flat)}")
                    global_flat = global_flat[:min_size]
                    client_flat = client_flat[:min_size]
                    
                    similarity = np.dot(global_flat, client_flat) / (
                        np.linalg.norm(global_flat) * np.linalg.norm(client_flat) + 1e-8)
                    layer_similarities.append(similarity)
                else:
                    layer_similarities.append(0.0)
            
            # Convert similarities to weights using softmax
            similarities = np.array(layer_similarities)
            exp_similarities = np.exp(similarities - np.max(similarities))
            weights = exp_similarities / np.sum(exp_similarities)
            
            # Weighted average of layer parameters
            aggregated_layer = np.zeros_like(layer_weights[0], dtype=np.float64)
            for client_idx in range(len(layer_weights)):
                aggregated_layer += weights[client_idx] * layer_weights[client_idx]
            
            aggregated_weights.append(aggregated_layer)
        
        return aggregated_weights

if DEVICE.type == "cuda":
    backend_config = {"client_resources": {"num_gpus": 1,"num_cpus": 1},}

flwr.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_PARTITIONS,
        config=ServerConfig(num_rounds=TOTAL_RND,round_timeout=3600*24*7),
        client_resources={"num_gpus": 1,"num_cpus": 1},
        strategy=Strategy(
        fraction_fit=0.1,          
        min_fit_clients=4,         
        min_available_clients=4,
        fraction_evaluate = 0.1,
        initial_parameters=ndarrays_to_parameters(params))
)
