import os

import torch
import tqdm
from torch import nn
from torch.nn import functional as F

from torch.utils.data import DataLoader, RandomSampler, Dataset

from metrics import getAcc
from torch.cuda.amp import autocast, GradScaler


class Trainer:

    def __init__(self, args, tb_logger, logger):
        self.args = args

        self.gpu = torch.device(args.gpu)
        self.model = None
        self.it = 0
        self.best_eval_acc, self.best_it = 0.0, 0

        # init dataset
        self.trainDataset = None
        self.trainDataloader = None
        self.evalDataset = None
        self.evalDataloader = None

        # optimizer and scheduler
        self.scheduler = None
        self.optimizer = None

        # loss
        self.loss_fn = None
        self.weight = None
        self.setLoss(args.loss)
        self.ignore_index = args.model["letter_size"]

        # gradient clipping
        if args.clip_grad is not None:
            self.clip_grad = True
            self.clip_value = args.clip_grad
        else:
            self.clip_grad = False

        if hasattr(args, "label_smoothing") and args.label_smoothing is not None:
            self.label_smoothing = float(args.label_smoothing)
        else:
            self.label_smoothing = 0.0

        # logging
        if tb_logger is not None:
            self.tb_log = tb_logger
        self.print_fn = print if logger is None else logger.info

        return

    def train(self):
        """
            Train The Model
        """
        self.model.train()

        # for gpu profiling
        start_batch = torch.cuda.Event(enable_timing=True)
        end_batch = torch.cuda.Event(enable_timing=True)
        start_run = torch.cuda.Event(enable_timing=True)
        end_run = torch.cuda.Event(enable_timing=True)

        scaler = GradScaler()

        start_batch.record()
        # eval for once
        if self.args.resume:
            eval_dict = self.evaluate()
            print(eval_dict)

        tbar = tqdm.tqdm(total=len(self.trainDataloader), colour='BLUE')

        for samples, targets, _ in self.trainDataloader:
            tbar.update(1)
            self.it += 1

            end_batch.record()
            torch.cuda.synchronize()
            start_run.record()

            samples, targets = samples.to(self.gpu), targets.to(self.gpu).long()

            with autocast():
                logits = self.model(samples)
                loss = F.cross_entropy(logits.flatten(end_dim=1), targets.flatten(),
                                       ignore_index=self.ignore_index,
                                       label_smoothing=self.label_smoothing)

            scaler.scale(loss).backward()

            if self.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value)

                scaler.step(self.optimizer)
                scaler.update()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.model.zero_grad()

            end_run.record()
            torch.cuda.synchronize()

            # tensorboard_dict update
            tb_dict = {}
            tb_dict['train/loss'] = loss.detach().cpu().item()

            tb_dict['lr'] = self.optimizer.param_groups[0]['lr']
            tb_dict['GPU/prefecth_time'] = start_batch.elapsed_time(end_batch) / 1000.
            tb_dict['GPU/run_time'] = start_run.elapsed_time(end_run) / 1000.

            if self.it % self.args.num_eval_iter == 0:
                eval_dict = self.evaluate()
                tb_dict.update(eval_dict)
                save_path = self.args.save_path
                if tb_dict['Word/Acc'] > self.best_eval_acc:
                    self.best_eval_acc = tb_dict['Word/Acc']
                    self.best_it = self.it

                self.print_fn(
                    f"\n {self.it} iteration, {tb_dict}, \n BEST_EVAL_ACC: {self.best_eval_acc}, at {self.best_it} iters")
                self.print_fn(
                    f" {self.it} iteration, ACC: {tb_dict['Word/Acc']}\n")
                if self.it == self.best_it:
                    self.save_model('model_best.pth', save_path)

            if self.tb_log is not None:
                self.tb_log.update(tb_dict, self.it)
            del tb_dict
            start_batch.record()

        eval_dict = self.evaluate()
        eval_dict.update({'eval/best_acc': self.best_eval_acc, 'eval/best_it': self.best_it})
        return eval_dict

    @torch.no_grad()
    def evaluate(self, model: nn.Module = None, evalDataset: Dataset = None):
        self.print_fn("\n Evaluation!!!")

        if model is None:
            model = self.model
        if evalDataset is not None:
            evalDataloader = DataLoader(evalDataset, self.args.eval_batch_size, shuffle=False, num_workers=0)
        else:
            evalDataloader = self.evalDataloader

        eval_dict = {}

        model.eval()

        preds_arr = None
        targets_arr = None
        lengths_arr = None
        for samples, targets, lengths in evalDataloader:
            samples, targets = samples.to(self.gpu), targets.to(self.gpu)

            outputs = model(samples)

            preds = torch.max(outputs, dim=2)[1]

            if preds_arr is None:
                preds_arr = preds.detach().cpu()
                targets_arr = targets.detach().cpu()
                lengths_arr = lengths.detach().cpu()
            else:
                preds_arr = torch.concat((preds_arr, preds.detach().cpu()))
                targets_arr = torch.concat((targets_arr, targets.detach().cpu()))
                lengths_arr = torch.concat((lengths_arr, lengths.detach().cpu()))

        wordAcc, charAcc = getAcc(preds_arr, targets_arr, lengths_arr)
        eval_dict.update({"Word/Acc": wordAcc,
                          "Char/Acc": charAcc})
        model.train()
        return eval_dict

    def save_model(self, save_name, save_path):
        save_filename = os.path.join(save_path, save_name)
        self.model.eval()
        save_dict = {"model": self.model.state_dict(),
                     'optimizer': self.optimizer.state_dict(),
                     'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None,
                     'it': self.it}
        torch.save(save_dict, save_filename)
        self.model.train()
        self.print_fn(f"model saved: {save_filename}\n")

    def save_baseLearner(self, save_name, save_path, trainIndexes):
        save_filename = os.path.join(save_path, save_name)
        self.model.eval()
        save_dict = {"model": self.model.state_dict(),
                     'optimizer': self.optimizer.state_dict(),
                     'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None,
                     'trainIndexes': trainIndexes,
                     'it': self.it}
        torch.save(save_dict, save_filename)
        self.model.train()
        self.print_fn(f"model saved: {save_filename}\n")

    def load_model(self, load_dir, load_name):
        """
            load saved model a
        :param load_dir: directory of loading model
        :param load_name: model name
        """
        load_path = os.path.join(load_dir, load_name)
        checkpoint = torch.load(load_path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint['scheduler'] is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.it = checkpoint['it']
        self.print_fn(f'model loaded from {load_path}')

    def set_optimizer(self, optimizer, scheduler=None):
        """
            set optimizer and scheduler
        :param optimizer: optimizer
        :param scheduler: scheduler
        """
        self.optimizer = optimizer
        self.scheduler = scheduler

    def setModel(self, model):
        """
            set model
        :param model: model
        """
        self.model = model.cuda(self.gpu)

    def setDatasets(self, trainDataset, evalDataset):
        """
            set train and evaluation datasets and dataloaders
        :param trainDataset: train dataset
        :param evalDataset: evaluation dataset
        """
        self.print_fn(f"\n Num Train Labeled Sample : {len(trainDataset)}\n Num Val Sample : {len(evalDataset)}")
        self.trainDataset = trainDataset
        self.evalDataset = evalDataset

        self.trainDataloader = DataLoader(trainDataset, batch_size=self.args.batch_size,
                                          sampler=RandomSampler(data_source=trainDataset,
                                                                replacement=True,
                                                                num_samples=self.args.iter * self.args.batch_size),
                                          num_workers=self.args.num_workers, drop_last=True, pin_memory=True)

        self.evalDataloader = DataLoader(evalDataset, self.args.eval_batch_size, shuffle=False, num_workers=0,
                                         pin_memory=True)

    def setLoss(self, loss_function: dict):
        """
            set loss function
        :param loss_function: loss function arguments
        """
        if loss_function["name"] == 'CrossEntropyLoss':
            self.loss_fn = nn.CrossEntropyLoss(label_smoothing=loss_function["label_smoothing"]).cuda(self.gpu)
        else:
            raise Exception(f"Unknown Loss Function : {loss_function}")
