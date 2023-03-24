import os
import torch
import torch.nn as nn
from trainers import StandardTrainer
from trainers.base import inference_mode


class DistillTrainer(StandardTrainer):

    def __init__(self, *args, teacher=None, masking=None, **kwargs):
        super().__init__(*args, **kwargs)
        assert teacher is not None
        self.teacher = teacher
        self.masking = masking
        self.masking = masking
        if self.opt.distill_mode != 'none':
            self.add_meter('loss_distill', 'Ld', loop_id=0, fstr_format='6.3f')
        self.add_meter('loss_task', 'Lt', loop_id=0, fstr_format='6.3f')
        if self.opt.save_mask:
            self.masks_to_save = []

    def get_data_batch(self, loop_id, phase_id):
        batch = self._next_data_batch(loop_id)
        if self.opt.save_mask:
            batch = list(batch[0]) + [batch[1]]
        return [t.to(self.device) for t in batch]

    def save_mask(self, epoch_id, idx, mask):
        N, L = mask.size()
        p = torch.arange(0, L, dtype=torch.int64, device=mask.device)
        p = p.unsqueeze(0).repeat(N, 1)
        mask_int = (mask * 2 ** p).sum(1)
        self.masks_to_save.append(torch.stack([idx, mask_int], 1).cpu())

    def do_epoch(self, epoch_id):
        super().do_epoch(epoch_id)
        if self.opt.save_mask:
            masks = torch.cat(self.masks_to_save, 0)
            path = os.path.join(self.manager.get_checkpoint_dir(),
                                f'mask-{epoch_id}-{self.manager._rank}.pt')
            with open(path, 'w'):
                torch.save(masks, path)
            self.masks_to_save = []

    def do_step_train(self, epoch_id, data_batch, config, n_accum_steps):
        model = self.models['model']
        criterion_cls = self.criterions['classification']
        if 'knowledge_distillation' in self.criterions:
            criterion_kd = self.criterions['knowledge_distillation']
        if 'feature_distillation' in self.criterions:
            criterion_fd = self.criterions['feature_distillation']
        if self.opt.save_mask:
            images, labels, idx = data_batch
        else:
            images, labels = data_batch

        logits = None
        
        # Distillation
        if self.opt.distill_mode == 'kd':
            with torch.no_grad():
                logits_teacher = self.teacher(images)
            logits = model(images)
            loss_distill = criterion_kd(logits, logits_teacher)
        elif self.opt.distill_mode == 'kd_image_mask':
            images_masked = self.masking(images, labels)
            with torch.no_grad():
                logits_teacher_masked = self.teacher(images_masked)
            logits_masked = model(images_masked)
            loss_distill = criterion_kd(logits_masked, logits_teacher_masked)
        elif self.opt.distill_mode == 'fd':
            with torch.no_grad():
                features_teacher = self.teacher(images, get_logits=False, get_features=True)
            features = model(images, get_logits=False, get_features=True)
            loss_distill = criterion_fd(features, features_teacher)
        elif self.opt.distill_mode == 'fd_image_mask':
            images_masked = self.masking(images, labels)
            if self.opt.save_mask:
                images_masked, patch_mask = images_masked
                self.save_mask(epoch_id, idx, patch_mask)
            with torch.no_grad():
                features_teacher_masked = self.teacher(images_masked, get_logits=False, get_features=True)
            features_masked = model(images_masked, get_logits=False, get_features=True)
            loss_distill = criterion_fd(features_masked, features_teacher_masked)
        elif self.opt.distill_mode == 'fd_mae_mask':
            mask = self.masking(images, labels)
            with torch.no_grad():
                features_teacher_masked = self.teacher(images, mask=mask, get_logits=False, get_features=True)
            features_masked = model(images, mask=mask, get_logits=False, get_features=True)
            loss_distill = criterion_fd(features_masked, features_teacher_masked)
        elif self.opt.distill_mode == 'fd_attn_mask':
            attn_mask = self.masking(images, labels)
            with torch.no_grad():
                features_teacher_masked = self.teacher(images, attn_mask=attn_mask, get_logits=False, get_features=True)
            features_masked = model(images, attn_mask=attn_mask, get_logits=False, get_features=True)
            loss_distill = criterion_fd(features_masked, features_teacher_masked)
        elif self.opt.distill_mode == 'none':
            loss_distill = 0
        else:
            raise NotImplementedError()

        # Task
        if self.opt.task == 'std':
            if logits is None:
                logits = model(images)
            loss_task = criterion_cls(logits, labels)
        else:
            raise NotImplementedError()

        loss = self.opt.w_task * loss_task + self.opt.w_distill * loss_distill
        loss /= n_accum_steps
        loss.backward()
        if self.opt.grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), self.opt.grad_clip)

        self.loop_meters['loss'].update(loss)
        self.loop_meters['loss_task'].update(loss_task)
        if self.opt.distill_mode != 'none':
            self.loop_meters['loss_distill'].update(loss_distill)
        self._update_acc_meter('acc', logits, labels)

    def do_step_test(self, data_batch, config):
        if self.opt.save_mask:
            data_batch = data_batch[:-1]
        super().do_step_test(data_batch, config)
