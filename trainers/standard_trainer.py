from __future__ import annotations
from typing import Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.base import BaseDataset
from trainers.base import ClassificationTrainer, LoopConfig, OptimizerWithSchedule, inference_mode
from utils.experiman import ExperiMan


class StandardLoopConfig(LoopConfig):

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
        super().__init__(
            name, dataset, dataloader, training, n_iterations,
            n_phases, n_logical_steps, n_computation_steps, run_every_n_epochs,
            run_at_checkpoint, run_at_last_epoch, for_best_meter)


class StandardTrainer(ClassificationTrainer):

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
        num_classes: int = None,
        ignored_classes: Union[list[list], list] = None,
        keep_eval_mode: bool = False,
        acc_per_class: bool = False,
    ):
        self.opt = manager.get_opt()
        super().__init__(
            manager=manager,
            models=models,
            criterions=criterions,
            n_epochs=n_epochs,
            loop_configs=loop_configs,
            optimizers=optimizers,
            log_period=log_period,
            ckpt_period=ckpt_period,
            device=device,
            save_init_ckpt=save_init_ckpt,
            resume_ckpt=resume_ckpt,
            num_classes=num_classes,
            ignored_classes=ignored_classes,
        )
        self.keep_eval_mode = keep_eval_mode
        self.acc_per_class = acc_per_class
        self.setup_meters()
    
    def setup_meters(self):
        def loops_satisfy(criterion):
            return [i for i, c in enumerate(self.loop_configs) if criterion(c)]
        self.add_meter('learning_rate', 'lr',
                       meter_type='scaler', omit_from_results=True)
        self.add_meter('loss', 'L',
                       loop_id=loops_satisfy(lambda c: c.training),
                       fstr_format='6.3f')
        acc_meter_type = 'per_class_avg' if self.acc_per_class else 'avg'
        self.add_meter('acc', 'Acc',
                       meter_type=acc_meter_type, fstr_format='5.2f')
        loop_for_best_meter = loops_satisfy(lambda c: c.for_best_meter)
        if loop_for_best_meter:
            loop_id = loop_for_best_meter[0]
            config = self.loop_configs[loop_id]
            best_meter = 'acc'
            self.set_meter_for_best_checkpoint(
                loop_id=loop_id, name=best_meter, maximum=True)

    def get_data_batch(self, loop_id, phase_id):
        batch = self._next_data_batch(loop_id)
        return [t.to(self.device) for t in batch]

    def get_active_optimizers(self, loop_id, phase_id):
        if self.loop_configs[loop_id].training:
            return [self.optimizers['optimizer']]
        else:
            return []

    def get_checkpoint(self, epoch_id):
        checkpoint = super().get_checkpoint(epoch_id)
        if self._should_run_loop(epoch_id=epoch_id, loop_id=3):
            meters = self.meters[3]
            if 'acc' in meters:
                checkpoint['test_acc'] = meters['acc'].get_value()
            if self.acc_per_class:
                if 'acc' in meters:
                    checkpoint['test_acc_per_class'] = \
                        meters['acc'].get_value(per_class_avg=False)
        return checkpoint

    def toggle_model_mode(self, epoch_id, loop_id):
        model = self.models['model']
        training = self.loop_configs[loop_id].training
        model.train(training and not self.keep_eval_mode)
        bare_model = model.module if hasattr(model, 'module') else model 
        if bare_model.use_dataset_preprocess:
            preprocess_fn = self.loop_configs[loop_id].dataset.preprocess
            bare_model.set_preprocess(preprocess_fn)

    def update_meters(self):
        if self.optimizers:
            lr = self.optimizers['optimizer'].get_learning_rates()[0]
            self.loop_meters['learning_rate'].update(lr)

    def do_step(self, epoch_id, loop_id, iter_id, phase_id, data_batch):
        config = self.loop_configs[loop_id]
        if config.training:
            self.do_step_train(epoch_id, data_batch, config,
                               n_accum_steps=config.n_computation_steps[phase_id])
        else:
            self.do_step_test(data_batch, config)

    def do_step_train(self, epoch_id, data_batch, config, n_accum_steps):
        model = self.models['model']
        criterion_cls = self.criterions['classification']
        images, labels = data_batch

        logits = model(images)
        loss = criterion_cls(logits, labels)

        loss /= n_accum_steps
        loss.backward()
        if self.opt.grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), self.opt.grad_clip)

        self.loop_meters['loss'].update(loss)
        self._update_acc_meter('acc', logits, labels)

    def do_step_test(self, data_batch, config):
        model = self.models['model']
        images, labels = data_batch
        self._update_acc_meter('acc', model(images), labels)
