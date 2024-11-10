import math

import torch
from torch.optim.lr_scheduler import LambdaLR

from Model import TTR
from dataset.TurkishSceneTextDataset import TurkishSceneTextDataset
from dataset.strit import STRIT
from dataset.syntheticTurkishStyleText import SyntheticTurkishStyleText


def getDatasets(args):
    if args.dataset["train"]["name"] == 'SyntheticTurkishStyleText':
        trainDataset = SyntheticTurkishStyleText(args.dataset["train"])
    elif args.dataset["train"]["name"] == 'TurkishSceneTextDataset':
        trainDataset = TurkishSceneTextDataset(args.dataset["train"], train=True)
    elif args.dataset["train"]["name"] == 'Eng':
        raise NotImplemented()
        # TODO : add codes for English dataset training
    else:
        raise Exception("Unknown Train Dataset!")

    if args.dataset["val"]["name"] == 'strit':
        valDataset = STRIT(args.dataset["val"])
    elif args.dataset["val"]["name"] == 'TurkishSceneTextDataset':
        valDataset = TurkishSceneTextDataset(args.dataset["val"], train=False)
    elif args.dataset["val"]["name"] == 'Eng':
        raise NotImplemented()
        # TODO : add codes for English dataset training
    else:
        raise Exception("Unknown Val Dataset!")

    return trainDataset, valDataset


def getModel(args):
    if args.model["name"].lower() == 'ttr':
        model = TTR(args.model)
    else:
        raise Exception(f"Unknown model name : {args.model}")
    return model


def getOptimizer(args, model):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if 'bn' in name or 'bias' in name:
            no_decay.append(param)
        else:
            decay.append(param)

    per_param_args = [{'params': decay, 'weight_decay': args.optimizer["weight_decay"]},
                      {'params': no_decay, 'weight_decay': 0.0}]

    if args.optimizer["name"] == 'SGD':
        optimizer = torch.optim.SGD(per_param_args, lr=args.optimizer["lr"], momentum=args.optimizer["momentum"],
                                    weight_decay=args.optimizer["weight_decay"], nesterov=True)
    elif args.optimizer["name"] == 'AdamW':
        lr = args.optimizer["lr"]
        optimizer = torch.optim.AdamW(per_param_args, lr=lr, weight_decay=args.optimizer["weight_decay"])
    elif args.optimizer["name"] == "Adam":
        optimizer = torch.optim.Adam(per_param_args, lr=args.optimizer["lr"], weight_decay=args.optimizer["weight_decay"])
    else:
        raise Exception(f"Unknown optimizer name! : {args.optimizer}")
    return optimizer


def getScheduler(args, optimizer):
    if args.scheduler is None:
        return None
    else:
        if args.scheduler["name"].lower() == "cosine":
            return get_cosine_schedule_with_warmup(optimizer,
                                                   args.iter * args.epoch,
                                                   num_warmup_steps=args.scheduler["num_warmup_steps"])
        elif args.scheduler["name"].lower() == "onecyclelr":
            lr = args.optimizer["lr"]
            return torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, args.iter,
                                                       pct_start=0.0075, cycle_momentum=False)
            pass
        else:
            raise Exception


def get_cosine_schedule_with_warmup(optimizer,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    num_warmup_steps=0,
                                    last_epoch=-1):
    """
    Get cosine scheduler (LambdaLR).
    if warmup is needed, set num_warmup_steps (int) > 0.
    """

    def _lr_lambda(current_step):
        """
        _lr_lambda returns a multiplicative factor given an integer parameter epochs.
        Decaying criteria: last_epoch
        """

        if current_step < num_warmup_steps:
            _lr = float(current_step) / float(max(1, num_warmup_steps))
        else:
            num_cos_steps = float(current_step - num_warmup_steps)
            num_cos_steps = num_cos_steps / float(max(1, num_training_steps - num_warmup_steps))
            _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
        return _lr

    return LambdaLR(optimizer, _lr_lambda, last_epoch)
