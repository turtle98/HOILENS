"""
Utilities

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import torch
import numpy as np
import scipy.io as sio
#import wandb
from tqdm import tqdm
from collections import defaultdict

from utils.hico_text_label import hico_unseen_index
import utils.ddp as ddp
import pocket
from pocket.core import DistributedLearningEngine
from pocket.utils import DetectionAPMeter, BoxPairAssociation
import datetime
import time

from torch.cuda import amp
from pocket.ops import relocate_to_cuda

class CustomisedDLE(DistributedLearningEngine):
    def __init__(self, net, dataloader, max_norm=0, num_classes=117,test_loader=None,args=None, **kwargs):
        super().__init__(net, None, dataloader, **kwargs)
        self.net = net
        self.max_norm = max_norm
        self.num_classes = num_classes
        self.train_loader = dataloader
        self.test_loader = test_loader
        self.best_unseen = -1
        self.best_seen = -1
        self.args = args
        if self.args.amp:
            self.scaler = amp.GradScaler(enabled=True)

        self.epoch_start_time = None
        self.last_logged_epoch = 0

    def _on_end_iteration(self):
        # Print stats in the master process
        if self._verbal and self._state.iteration % self._print_interval == 0:
            self._print_statistics()

    def _on_start_iteration(self):
        if self._state.epoch != self.last_logged_epoch:
            self.epoch_start_time = time.time()
            self.last_logged_epoch = self._state.epoch
        self._state.iteration += 1
        self._state.inputs = relocate_to_cuda(self._state.inputs,ignore=True, non_blocking=True)
        self._state.targets = relocate_to_cuda(self._state.targets,ignore=True, non_blocking=True)

    def _print_statistics(self):
        running_loss = self._state.running_loss.mean()
        t_data = self._state.t_data.sum() / self._world_size
        t_iter = self._state.t_iteration.sum() / self._world_size

        t_iter_mean = self._state.t_iteration.mean()
        t_data_mean = self._state.t_data.mean()

        it_sec = t_iter_mean + t_data_mean

        mse_loss = None
        if hasattr(self._state, "inputs") and hasattr(self._state, "targets"):
            try:
                with torch.no_grad():
                    out = self.net(*self._state.inputs, targets=self._state.targets)
                    if isinstance(out, dict) and "mse_loss" in out:
                        mse_loss = out["mse_loss"].item()
            except Exception:
                mse_loss = None

        # Print stats in the master process
        if self._rank == 0:
            num_iter = len(self._train_loader)
            n_d = len(str(num_iter))
            current_iter = self._state.iteration - num_iter * (self._state.epoch - 1)

            msg = (
                f"Epoch [{self._state.epoch}/{self.epochs}], "
                f"Iter. [{str(current_iter).zfill(n_d)}/{num_iter}], "
                f"Loss: {running_loss:.4f}"
            )
            if mse_loss is not None:
                msg += f", MSE: {mse_loss:.4f}"

            msg += (
                f", Time[Data/Iter./Remain.]: "
                f"[{t_data:.2f}s/{t_iter:.2f}s/"
                f"{datetime.timedelta(seconds=(num_iter - current_iter) * it_sec)}]"
            )
            print(msg)

            log_path = os.path.join(self.args.output_dir, "train_log.txt")
            with open(log_path, "a") as f:
                f.write(msg + "\n")

        self._state.t_iteration.reset()
        self._state.t_data.reset()
        self._state.running_loss.reset()

    def _on_each_iteration(self):
        self._state.net.train()
        with amp.autocast(enabled=self.args.amp):
            loss_dict = self._state.net(
                *self._state.inputs, targets=self._state.targets)
        if loss_dict['mse_loss'].isnan():
            raise ValueError(f"The MSE loss is NaN for rank {self._rank}")

        if self.args.amp:
            self._state.loss = sum(loss for loss in loss_dict.values())
            self._state.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(self._state.loss).backward()
            self.scaler.step(self._state.optimizer)
            self.scaler.update()
        else:
            self._state.loss = sum(loss for loss in loss_dict.values())
            self._state.optimizer.zero_grad(set_to_none=True)
            self._state.loss.backward()
            if self.max_norm > 0:
                torch.nn.utils.clip_grad_norm_(self._state.net.parameters(), self.max_norm)
            self._state.optimizer.step()

    def _on_end_epoch(self):
        # if self._rank == 0:
        #     self.save_checkpoint()
        if self._state.lr_scheduler is not None:
            self._state.lr_scheduler.step()
        if self.epoch_start_time is not None:
            epoch_duration = time.time() - self.epoch_start_time
            if self._rank == 0:  # only log on master
                print(f"Epoch {self._state.epoch} training time: {epoch_duration:.2f} seconds")
                self.epoch_start_time = None  # reset for next epoch
                log_file_path = os.path.join(self.args.output_dir, "eval_log.txt")
                with open(log_file_path, "a") as f:
                    f.write(f"Training time : {epoch_duration }\n")
                    f.write("\n")
                self.save_checkpoint()

