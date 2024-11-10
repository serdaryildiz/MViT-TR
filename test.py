import argparse
import random

import numpy
import torch
from torch.backends import cudnn

from Model.trainer import Trainer
from train_logging import TBLog, get_logger, over_write_args_from_file
from train_utils import getDatasets, getModel, getOptimizer, getScheduler


def main_worker(args):
    # random seed has to be set for the synchronization of labeled data sampling in each process.
    assert args.seed is not None
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    numpy.random.seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    # set logger
    logger_level = "INFO"
    logger = get_logger(args.save_name, None, logger_level)
    logger.warning(f"USE GPU: {args.gpu} for training")
    logger.info(args)
    # init model
    trainer = Trainer(args=args, tb_logger=None, logger=logger)

    trainDataset, valDataset = getDatasets(args)
    trainer.setDatasets(trainDataset=trainDataset, evalDataset=valDataset)

    # get model
    model = getModel(args)

    # set model
    trainer.setModel(model)

    trainer.model.load_state_dict(torch.load(args.ckpt)["model"])

    logger.info("Test with best checkpoint")
    eval_dict = trainer.evaluate(evalDataset=valDataset)
    logger.info(f"Test Results : {eval_dict}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scene Text Recognition Study!')
    parser.add_argument('--config', type=str, default='./config/mvt_tr_real.yaml')
    parser.add_argument('--ckpt', type=str, default='./experiments/real_train/model_best.pth')
    args = parser.parse_args()
    over_write_args_from_file(args, args.config)
    main_worker(args)
