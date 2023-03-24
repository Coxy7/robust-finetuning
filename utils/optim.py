import math
import torch.optim as optim
from trainers import OptimizerWithSchedule


def get_optim(parameters, optimizer_name, lr, schedule,
              weight_decay, num_epochs, num_iters_train,
              cyclic_stepsize=None, multistep_milestones=None,
              onecycle_pct_start=0.25, adam_beta=0.5) -> OptimizerWithSchedule:

    if optimizer_name == 'sgd':
        optimizer = optim.SGD(
            parameters, lr=lr, momentum=0.9, weight_decay=weight_decay,
            nesterov=False)
    elif optimizer_name == 'sgd_nesterov':
        optimizer = optim.SGD(
            parameters, lr=lr, momentum=0.9, weight_decay=weight_decay,
            nesterov=True)
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(
            parameters, lr=lr, weight_decay=weight_decay,
            betas=(adam_beta, 0.999))
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(
            parameters, lr=lr, weight_decay=weight_decay,
            betas=(adam_beta, 0.999))

    if schedule == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs)
        schedule_step = 'epoch'
    elif schedule == 'sgdr':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=1)
        schedule_step = 'epoch'
    elif schedule == 'cyclic':
        if cyclic_stepsize is None:
            cyclic_stepsize = 0.5 * num_epochs
        scheduler = optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=0, max_lr=lr,
            cycle_momentum=(optimizer_name == 'sgd'),
            step_size_up=int(cyclic_stepsize * num_iters_train))
        schedule_step = 'iter'
    elif schedule == '1cycle':
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr,
            epochs=num_epochs, steps_per_epoch=num_iters_train,
            pct_start=onecycle_pct_start, anneal_strategy='cos')
        schedule_step = 'iter'
    elif schedule == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=multistep_milestones, gamma=0.1)
        schedule_step = 'epoch'
    elif schedule == 'none':
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, (lambda epoch: 1))
        schedule_step = 'epoch'
    
    config = OptimizerWithSchedule(optimizer, scheduler, schedule_step)
    return config
