# -*- coding: utf-8 -*-
# @Time   : 2022/3/24
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn
import argparse
import logging
from logging import getLogger

from recbole.data import construct_transform
from recbole.utils import init_logger, init_seed, set_color, get_flops
from recbole_debias.config import Config
from recbole_debias.data import create_dataset, data_preparation
from recbole_debias.utils import get_model, get_trainer


def run_recbole_debias(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True):
    r""" A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """
    # configurations initialization
    config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()

    # dataset filtering
    dataset = create_dataset(config)

    print(dataset.inter_feat.head())
    # dataset splitting
    #train_data, valid_data, test_data = data_preparation(config, dataset)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='MACR', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='ml-100k', help='name of datasets')
    parser.add_argument('--config_files', '-c', type=str, default=None, help='config files')

    args, _ = parser.parse_known_args()


    config_file_list = args.config_files.strip().split(' ') if args.config_files else None


    run_recbole_debias(model=args.model, dataset=args.dataset, config_file_list=config_file_list)
