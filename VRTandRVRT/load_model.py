import argparse
from collections import OrderedDict
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist
from models.select_model import define_Model
# from testt import set_parameters


def load(path):
    json_path='options/vrt/002_train_vrt_videosr_bi_reds_16frames.json'
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--dist', default=False)

    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist

    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    if opt['rank'] == 0:
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


    opt = option.dict_to_nonedict(opt)

    model = define_Model(opt)
    pretrained_model = torch.load(path)
    model.netG.load_state_dict(pretrained_model['params'] if 'params' in pretrained_model.keys() else pretrained_model, strict=True)
    # model.netG.load_state_dict(state_dict, strict=True)
    return model
