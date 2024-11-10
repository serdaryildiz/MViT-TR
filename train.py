import argparse
import os
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
    # cudnn.deterministic = True
    # cudnn.benchmark = True

    save_path = os.path.join(args.save_dir, args.save_name)
    args.save_path = save_path
    if os.path.exists(save_path) and args.overwrite and args.resume == False:
        import shutil
        shutil.rmtree(save_path)
    if os.path.exists(save_path) and not args.overwrite:
        raise Exception('already existing model: {}'.format(save_path))
    if args.resume:
        if args.load_path is None:
            raise Exception('Resume of training requires load_path')
        if os.path.abspath(save_path) == os.path.abspath(args.load_path) and not args.overwrite:
            raise Exception('Saving & Loading paths are same')

    # set logger
    tb_logger = TBLog(save_path, 'tensorboard', True)
    logger_level = "INFO"
    logger = get_logger(args.save_name, save_path, logger_level)
    logger.warning(f"USE GPU: {args.gpu} for training")
    logger.info(args)
    # init model
    trainer = Trainer(args=args, tb_logger=tb_logger, logger=logger)

    trainDataset, valDataset = getDatasets(args)
    trainer.setDatasets(trainDataset=trainDataset, evalDataset=valDataset)

    # get model
    model = getModel(args)

    # set optimizer and scheduler
    optimizer = getOptimizer(args, model)
    if args.scheduler is not None:
        scheduler = getScheduler(args, optimizer=optimizer)
    else:
        scheduler = None
    trainer.set_optimizer(optimizer=optimizer, scheduler=scheduler)

    # set model
    trainer.setModel(model)

    # if pretrain
    if args.pretrain is not None:
        trainer.model.load_state_dict(torch.load(args.pretrain)["model"], strict= False)

    # If args.resume
    if args.resume:
        trainer.load_model(args.load_path)

    for e in range(args.epoch):
        trainer.train()

    trainer.save_model('last.pth', save_path)

    if os.path.exists(os.path.join(args.save_path, "model_best.pth")):
        logger.info("Test with best checkpoint")
        trainer.load_model(args.save_path, "model_best.pth")
        eval_dict = trainer.evaluate(evalDataset=valDataset)
        logger.info(f"Test Results : {eval_dict}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scene Text Recognition Study!')
    parser.add_argument('--config', type=str, default='./config/mvt_tr_real.yaml')
    args = parser.parse_args()
    over_write_args_from_file(args, args.config)
    main_worker(args)
