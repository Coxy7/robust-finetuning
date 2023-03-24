from __future__ import annotations
import os
from typing import Any, Union
from typing_extensions import Literal
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from data.base import BaseDataset
from utils.experiman import ExperiMan
from utils.misc import Accuracy, AverageMeter, MovingAverageMeter, ScalerMeter, PerClassMeter


class inference_mode(torch.no_grad):

    def __init__(self, models=None):
        super().__init__()
        if models is None:
            models = []
        elif not isinstance(models, list):
            models = [models]
        self.models = models
        self.training = [False for _ in models]

    def __enter__(self):
        super().__enter__()
        for i, model in enumerate(self.models):
            self.training[i] = model.training
            model.eval()
    
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        for model, training in zip(self.models, self.training):
            model.train(training)
        super().__exit__(exc_type, exc_value, traceback)


class LoopConfig():

    def __init__(
        self,
        name: str,
        dataset: BaseDataset,
        dataloader: DataLoader,
        training: bool,
        n_iterations: int,
        n_phases: int = 1,
        n_logical_steps: Union[list[int], int] = 1,
        n_computation_steps: Union[list[int], int] = 1,
        run_every_n_epochs: int = 1,
        run_at_checkpoint: bool = True,
        run_at_last_epoch: bool = True,
        for_best_meter: bool = False,
    ):
        self.name = name
        self.dataset = dataset
        self.dataloader = dataloader
        self.training = training
        self.n_iterations = n_iterations
        self.n_phases = n_phases
        if type(n_logical_steps) is int:
            n_logical_steps = [n_logical_steps]
        self.n_logical_steps = n_logical_steps
        assert n_phases == len(n_logical_steps)
        if type(n_computation_steps) is int:
            n_computation_steps = [n_computation_steps]
        self.n_computation_steps = n_computation_steps
        assert n_phases == len(n_computation_steps)
        self.run_every_n_epochs = run_every_n_epochs
        self.run_at_checkpoint = run_at_checkpoint
        self.run_at_last_epoch = run_at_last_epoch or run_at_checkpoint
        self.for_best_meter = for_best_meter

    def __str__(self):
        configs = self.__dict__.copy()
        name = configs.pop('name')
        configs.pop('dataset')
        configs.pop('dataloader')
        arg_strings = [f"{k}={v}" for k, v in configs.items()]
        return f"[{name}] ({', '.join(arg_strings)})"


class OptimizerWithSchedule():

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        schedule_step: Literal['epoch', 'iter'],
    ):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.schedule_step = schedule_step

    def state_dict(self):
        return {
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'schedule_step': self.schedule_step,
        }

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])
    
    def step(self, *args, **kwargs):
        self.optimizer.step(*args, **kwargs)
    
    def zero_grad(self, *args, **kwargs):
        self.optimizer.zero_grad(*args, **kwargs)

    def scheduler_step(self, *args, **kwargs):
        self.scheduler.step(*args, **kwargs)
    
    def get_learning_rates(self):
        return self.scheduler.get_last_lr()


class BaseTrainer():
    """
    Hierarchy:
        1. Epoch: consists of multiple training / testing / updating loops
        2. Loop: loops (through a dataset) with repeated iterations 
        3. Iteration: training iteration may consist of multiple phases with
            different objectives
        4. Phase: consists of one or more repeated (accumulated) steps
        5. Logical step: update the models once (after one or more computation
            steps for accumulated gradients)
        6. Computation step: handles a batch of data; update the meters
            (train: forward & backward; test: forward)
    """

    def __init__(
        self,
        manager: ExperiMan,
        models: dict[str, nn.Module],
        criterions: dict[str, nn.Module],
        n_epochs: int,
        loop_configs: list[LoopConfig],
        optimizers: dict[str, OptimizerWithSchedule],
        log_period: int,
        ckpt_period: int,
        device: torch.device,
        save_init_ckpt: bool = False,
        resume_ckpt: dict = None,
    ):
        self.manager = manager
        self.is_master = manager.is_master()
        self.is_distributed = manager.is_distributed()
        self.logger = manager.get_logger()
        self.last_log_iter_id = -1 
        self.tqdms = [None for _ in loop_configs]
        self.models = models
        self.criterions = criterions
        self.n_epochs = n_epochs
        self.loop_configs = loop_configs
        self.data_counters = [0 for _ in loop_configs]
        self.data_iters = [self._get_data_iter(i) for i in range(len(loop_configs))]
        self.optimizers = optimizers
        self.log_period = log_period
        self.ckpt_period = ckpt_period or n_epochs + 1
        self.device = device
        self.save_init_ckpt = save_init_ckpt
        self.meters = [{} for _ in loop_configs]
        self.meters_info = [{} for _ in loop_configs]
        self.loop_meters = self.meters[0]
        self.loop_meters_info = self.meters_info[0]
        self.meter_for_best_checkpoint = None
        self.best_value = None
        self.start_epoch = 0
        self.iter_count = 0
        if resume_ckpt:
            self.resume_from_checkpoint(resume_ckpt)
        self._default_checkpoint_name = 'ckpt-last'
    
    def master_only(func):
        def wrapper_master_only(self, *args, **kwargs):
            if self.is_master:
                return func(self, *args, **kwargs)
            else:
                return None
        return wrapper_master_only

    def _setup_tqdms(self, loop_id):
        n_iters = self.loop_configs[loop_id].n_iterations
        t = tqdm(total=n_iters, leave=False, dynamic_ncols=True,
                 disable=(not self.is_master))
        t.clear()
        self.tqdms[loop_id] = t

    @master_only
    def _manager_log_metric(self, epoch_id, loop_id):
        split_name = self.loop_configs[loop_id].name
        for name, meter in self.loop_meters.items():
            self.manager.log_metric(name, meter.get_value(),
                                    self.iter_count, epoch_id,
                                    split=split_name)

    def _get_data_iter(self, loop_id):
        loader = self.loop_configs[loop_id].dataloader
        if hasattr(loader, 'sampler'):
            if hasattr(loader.sampler, 'set_epoch'):
                loader.sampler.set_epoch(self.data_counters[loop_id])
        self.data_counters[loop_id] += 1
        return iter(loader)

    def _next_data_batch(self, loop_id):
        try:
            batch = next(self.data_iters[loop_id])
        except StopIteration:
            self.data_iters[loop_id] = self._get_data_iter(loop_id)
            batch = next(self.data_iters[loop_id])
        return batch

    def _should_run_loop(self, epoch_id, loop_id):
        config = self.loop_configs[loop_id]
        cnt = epoch_id + 1
        period = config.run_every_n_epochs
        periodic = (period and cnt % period == 0)
        at_ckpt = (config.run_at_checkpoint and cnt % self.ckpt_period == 0)
        at_last = (config.run_at_last_epoch and cnt == self.n_epochs)
        not_empty = (config.n_iterations > 0)
        return not_empty and (periodic or at_ckpt or at_last)

    def get_data_batch(self, loop_id, phase_id):
        """Return a batch of data for the phase."""
        raise NotImplementedError

    def get_active_optimizers(self, loop_id, phase_id):
        """Return the optimizers active for the phase."""
        raise NotImplementedError

    def get_checkpoint(self, epoch_id):
        """Return a checkpoint object to be saved."""
        checkpoint = {
            'epoch': epoch_id,
            'best_value': self.best_value,
            '_data_counters': self.data_counters,
        }
        for name, model in self.models.items():
            bare_model = model.module if hasattr(model, 'module') else model
            checkpoint[name] = bare_model.state_dict()
        for name, optimizer in self.optimizers.items():
            checkpoint[name] = optimizer.state_dict()
        return checkpoint

    def resume_from_checkpoint(self, checkpoint):
        """Resume training from a checkpoint object."""
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_value = checkpoint['best_value']
        self.data_counters = checkpoint['_data_counters']
        for name, model in self.models.items():
            bare_model = model.module if hasattr(model, 'module') else model
            bare_model.load_state_dict(checkpoint[name])
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(checkpoint[name])
        for config in self.loop_configs:
            if config.training:
                self.iter_count += config.n_iterations * self.start_epoch
        return checkpoint

    def toggle_model_mode(self, epoch_id, loop_id):
        """Toggle train/eval mode of models."""
        raise NotImplementedError
        
    def update_meters(self):
        """Update meters before logging."""

    def do_step(self, epoch_id, loop_id, iter_id, phase_id, data_batch):
        """
        Typical procedure:
        1. Forward for predictions and losses;
        2. Backward for gradients (if needed);
        3. Update (avg/sum) meters.
        """
        raise NotImplementedError

    def add_meter(self, name, abbr=None, loop_id=None, meter_type='avg',
                  fstr_format=None, reset_every_epoch=None,
                  omit_from_results=False):
        if loop_id is None:
            loop_id = list(range(len(self.loop_configs)))
        elif type(loop_id) is int:
            loop_id = [loop_id]
        assert meter_type in ('avg', 'scaler', 'per_class_avg')
        for id in loop_id:
            if reset_every_epoch is None:
                reset = not self.loop_configs[id].training
            else:
                reset = reset_every_epoch
            self.meters_info[id][name] = {
                'abbr': abbr if abbr is not None else name,
                'type': meter_type,
                'format': fstr_format,
                'reset_every_epoch': reset,
                'omit_from_results': omit_from_results,
            }

    def set_meter_for_best_checkpoint(self, loop_id, name, maximum=True):
        self.meter_for_best_checkpoint = (loop_id, name, maximum)

    def setup_loop_meters(self, loop_id):
        training = self.loop_configs[loop_id].training
        self.loop_meters = self.meters[loop_id]
        self.loop_meters_info = self.meters_info[loop_id]
        for name, info in self.loop_meters_info.items():
            if name not in self.loop_meters:
                meter_type = info['type']
                if meter_type == 'avg':
                    if training:
                        meter = MovingAverageMeter()
                    else:
                        meter = AverageMeter()
                elif meter_type == 'per_class_avg':
                    if training:
                        meter = PerClassMeter(MovingAverageMeter)
                    else:
                        meter = PerClassMeter(AverageMeter)
                elif meter_type == 'scaler':
                    meter = ScalerMeter()
                else:
                    raise NotImplementedError()
                self.loop_meters[name] = meter
            elif info['reset_every_epoch']:
                self.loop_meters[name].reset()
                
    def is_best_checkpoint(self, epoch_id):
        is_best = False
        if self.meter_for_best_checkpoint is not None:
            loop_id, name, maximum = self.meter_for_best_checkpoint
            if self._should_run_loop(epoch_id, loop_id):
                value = self.meters[loop_id][name].get_value()
                if self.best_value is None:
                    self.best_value = value
                    is_best = True
                else:
                    delta = value - self.best_value
                    sign = 1 if maximum else -1
                    if delta * sign > 0:
                        self.best_value = value
                        is_best = True
        return is_best

    @master_only
    def save_checkpoint(self, epoch_id, checkpoint_names=None):
        checkpoint = self.get_checkpoint(epoch_id)
        if checkpoint_names is None:
            filenames = [f'ckpt-{epoch_id}.pt']
        else:
            filenames = [f'{name}.pt' for name in checkpoint_names]
        for name in filenames:
            model_path = os.path.join(self.manager.get_checkpoint_dir(), name)
            torch.save(checkpoint, model_path)
            if name[:-3] != self._default_checkpoint_name \
                    or epoch_id == self.n_epochs - 1:
                self.logger.info(f'Checkpoint saved to: {model_path}')

    @master_only
    def save_results(self, filename='results'):
        metrics = []
        for i, loop_config in enumerate(self.loop_configs):
            split = loop_config.name
            for name, meter in self.meters[i].items():
                if not self.meters_info[i][name]['omit_from_results']:
                    metric = dict(split=split, name=name,
                                  value=meter.get_value())
                    metrics.append(metric)
        self.manager.save_metrics(metrics, filename=filename)

    def update_lr(self, step):
        for optimizer in self.optimizers.values():
            if optimizer.schedule_step == step:
                optimizer.scheduler_step()

    def do_iter(self, epoch_id, loop_id, iter_id):
        config = self.loop_configs[loop_id]
        for phase_id in range(config.n_phases):
            for _ in range(config.n_logical_steps[phase_id]):
                optimizers = self.get_active_optimizers(loop_id, phase_id)
                for _ in range(config.n_computation_steps[phase_id]):
                    data_batch = self.get_data_batch(loop_id, phase_id)
                    self.do_step(epoch_id, loop_id, iter_id, phase_id, data_batch)
                for optimizer in optimizers:
                    optimizer.step()
                    optimizer.zero_grad()
        if config.training:
            self.iter_count += 1

    def log_iter(self, epoch_id, loop_id, iter_id):
        self.update_meters()
        display_items = [""]
        for name, info in self.loop_meters_info.items():
            fmt = info['format']
            if fmt is not None:
                value = self.loop_meters[name].get_value()
                display_items.append(f"{info['abbr']} {value:{fmt}}")
        msg = '|'.join(display_items)
        loop_name = self.loop_configs[loop_id].name
        self.tqdms[loop_id].set_postfix_str(f"[{loop_name}] {msg}")
        self.tqdms[loop_id].update(iter_id - self.last_log_iter_id)
        self.last_log_iter_id = iter_id
        if self.loop_configs[loop_id].training:
            self._manager_log_metric(epoch_id, loop_id)
    
    def do_loop(self, epoch_id, loop_id):
        config = self.loop_configs[loop_id]
        self.toggle_model_mode(epoch_id, loop_id)
        self.setup_loop_meters(loop_id)
        for iter_id in range(config.n_iterations):
            self.do_iter(epoch_id, loop_id, iter_id)
            if (iter_id + 1) % self.log_period == 0 \
                    or iter_id == config.n_iterations - 1:
                self.log_iter(epoch_id, loop_id, iter_id)
            if config.training:
                self.update_lr('iter')

    def log_loop(self, epoch_id, loop_id):
        config = self.loop_configs[loop_id]
        self.last_log_iter_id = -1
        bar = self.tqdms[loop_id]
        elapsed_time = bar.format_interval(bar.format_dict['elapsed'])
        bar.close()
        self.update_meters()
        display_items = []
        for name, info in self.loop_meters_info.items():
            meter = self.loop_meters[name]
            if self.is_distributed:
                meter.sync(self.device)
            fmt = info['format']
            if fmt is not None:
                value = meter.get_value()
                display_items.append(f"{info['abbr']} {value:{fmt}}")
        msg = '|'.join(display_items)
        self.logger.info(f"elapsed: {elapsed_time} [{config.name}] {msg}")
        if not config.training:
            self._manager_log_metric(epoch_id, loop_id)

    def do_epoch(self, epoch_id):
        for loop_id, loop_config in enumerate(self.loop_configs):
            if self._should_run_loop(epoch_id, loop_id):
                self._setup_tqdms(loop_id)
                with torch.set_grad_enabled(loop_config.training):
                    self.do_loop(epoch_id, loop_id)
                self.log_loop(epoch_id, loop_id)
    
    def log_epoch(self, epoch_id):
        lrs = [optimizer.get_learning_rates()[0]
               for optimizer in self.optimizers.values()]
        lrs = "|".join([f"{lr:.5f}" for lr in lrs])
        self.logger.info(f'Epoch: {epoch_id}/{self.n_epochs} lr: {lrs}')
    
    def log_loop_configs(self):
        n_loops = len(self.loop_configs)
        self.logger.info(f"Configs of {n_loops} loops:")
        for i, config in enumerate(self.loop_configs):
            self.logger.info(f"{i}: {str(config)}")

    def train(self):
        self.log_loop_configs()
        if self.save_init_ckpt and self.start_epoch == 0:
            self.save_checkpoint(-1, ['ckpt-init'])
        for epoch_id in range(self.start_epoch, self.n_epochs):
            self.log_epoch(epoch_id)
            self.do_epoch(epoch_id)
            self.update_lr('epoch')
            checkpoint_names = [self._default_checkpoint_name]
            if (epoch_id + 1) % self.ckpt_period == 0:
                checkpoint_names.append(f'ckpt-{epoch_id}')
            if self.is_best_checkpoint(epoch_id):
                checkpoint_names.append('ckpt-best')
                self.save_results('results-best')
            self.save_checkpoint(epoch_id, checkpoint_names)
        self.save_results()

    def test(self):
        self.log_loop_configs()
        self.do_epoch(0)
        self.save_results()


class ClassificationTrainer(BaseTrainer):

    def __init__(self, *args, num_classes=None, ignored_classes=None, **kwargs):
        super().__init__(*args, **kwargs)
        if ignored_classes is None:
            self.ignored_classes = []
            for config in self.loop_configs: 
                if hasattr(config.dataset, 'ignored_classes'):
                    self.ignored_classes.append(config.dataset.ignored_classes)
                else:
                    self.ignored_classes.append([])
        elif ignored_classes and isinstance(ignored_classes[0], list):
            assert len(ignored_classes) == len(self.loop_configs)
            self.ignored_classes = ignored_classes
        else:
            self.ignored_classes = [ignored_classes for _ in self.loop_configs]
        self.accuracies = []
        for ic in self.ignored_classes:
            accuracy = Accuracy(num_classes=num_classes,
                                ignored_classes=ic, reduction='sum')
            self.accuracies.append(accuracy)
        self.loop_accuracy = self.accuracies[0]
    
    def setup_loop_meters(self, loop_id):
        super().setup_loop_meters(loop_id)
        self.loop_accuracy = self.accuracies[loop_id]

    def _update_acc_meter(self, meter_name, outputs, labels):
        accuracy = self.loop_accuracy
        if self.loop_meters_info[meter_name]['type'] == 'per_class_avg':
            self.loop_meters[meter_name].update(
                accuracy(outputs, labels, reduction='none'), labels)
        else:
            n_correct, n = accuracy(outputs, labels)
            self.loop_meters[meter_name].update(n_correct, n)
