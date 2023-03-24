import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from utils.experiman import manager
from data import *
from models import get_clip_model
from trainers import StandardTrainer, StandardLoopConfig
from utils.misc import parse
from utils.optim import get_optim


def add_parser_argument(parser):
    ## ======================== Data ==========================
    parser.add_argument('--dataset', default='imagenet', type=str)
    parser.add_argument('--train_split', default='original', type=str)
    parser.add_argument('--val_size', default=10240, type=int)
    parser.add_argument('--data_split_seed', default=0, type=int)
    parser.add_argument('--batch', default=512, type=int)
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--image_size', default=224, type=int)
    parser.add_argument('--transform', default='clip', type=str)
    ## ======================= Model ==========================
    parser.add_argument('--arch', type=str)
    parser.add_argument('--arch_variant', default='zeroshot', type=str)
    parser.add_argument('--load_pretrained', type=str)
    parser.add_argument('--load_ckpt', type=str)
    parser.add_argument('--load_run_name', type=str)
    parser.add_argument('--load_run_number', type=str)
    parser.add_argument('--freeze_backbone', action='store_true')
    parser.add_argument('--sync_bn', action='store_true')
    ## ===================== Training =========================
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--resume_ckpt', type=str)
    parser.add_argument('--label_smooth', action='store_true')
    ## ==================== Optimization ======================
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--num_iters_train', type=int,
                        help="default: len(trainloader)")
    parser.add_argument('--num_iters_test', type=int,
                        help="default: len(testloader)")
    parser.add_argument('--num_iters_trainset_test', type=int,
                        help="default: len(raw_trainloader)")
    parser.add_argument('--accum_steps', type=int, default=1)
    parser.add_argument('--lr', default=3e-5, type=float)
    parser.add_argument('--lr_bb', type=float)
    parser.add_argument('--lr_schedule', default='1cycle', type=str)
    parser.add_argument('--multistep_milestones', type=int, nargs='+')
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument('--adam_beta', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-1, type=float)
    parser.add_argument('--cyclic_step', type=float)
    parser.add_argument('--onecycle_pct_start', default=0.02, type=float)
    parser.add_argument('--grad_clip', default=1, type=float)
    ## ====================== Logging =========================
    parser.add_argument('--log_period', default=5, type=int, metavar='LP',
                        help='log every LP iterations')
    parser.add_argument('--ckpt_period', type=int, metavar='CP',
                        help='make checkpoints every CP epochs')
    parser.add_argument('--test_period', default=1, type=int, metavar='TP',
                        help='test every TP epochs')
    parser.add_argument('--trainset_test_period', type=int, metavar='TP',
                        help='test on training set every TP epochs')
    parser.add_argument('--comment', default='', type=str)
    ## ==================== Experimental ======================
    parser.add_argument('--wise_alpha', default=0.5, type=float)
    parser.add_argument('--wise_base_run_name', type=str)
    parser.add_argument('--wise_base_run_number', type=str)


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(local_rank)
    torch.cuda.set_device(device)

    # Parse arguments and setup ExperiMan
    parser = manager.get_basic_arg_parser()
    add_parser_argument(parser)
    opt = parser.parse_args()
    if opt.resume_ckpt or opt.auto_resume:
        opt.option_for_existing_dir = 'k'
    manager.setup(opt, rank=rank, world_size=world_size,
                  third_party_tools=('tensorboard',))
    if world_size > 1:
        dist.init_process_group("nccl")
        if rank == 0:
            t = torch.tensor([opt.run_number + .1], device=device)
        else:
            t = torch.empty(1, device=device)
        dist.broadcast(t, src=0)
        opt.run_number = int(t.item())
        manager.set_run_dir(manager.get_run_dir(opt.run_name, opt.run_number))
    logger = manager.get_logger()
    logger.info(f'==> Number of devices: {world_size}')
    use_clip = opt.arch.startswith('clip')

    # Data
    logger.info('==> Preparing data')
    dataset = get_dataset(opt.dataset, opt.data_dir, size=opt.image_size, transform=opt.transform)
    assert opt.batch % world_size == 0
    batch = opt.batch // world_size
    data_kwargs = dict(
        batch_size=batch, num_workers=opt.num_workers, with_index=False,
        train_split=opt.train_split, val_size=opt.val_size,
        split_seed=opt.data_split_seed,
        world_size=world_size, rank=rank)
    if opt.val_size > 0:
        trainloader, raw_trainloader, valloader, testloader = \
            dataset.get_loader(**data_kwargs)
    else:
        trainloader, raw_trainloader, testloader = \
            dataset.get_loader(**data_kwargs)
        valloader = []
    num_iters_train = parse(opt.num_iters_train, len(trainloader) // opt.accum_steps)
    num_iters_val = len(valloader)
    num_iters_trainset_test = parse(opt.num_iters_trainset_test, len(raw_trainloader))
    num_iters_test = parse(opt.num_iters_test, len(testloader))

    # Model
    logger.info('==> Building models')
    if use_clip:
        model = get_clip_model(
            arch=opt.arch,
            dataset=dataset,
            variant=opt.arch_variant,
            model_dir=opt.load_pretrained,
            device=device,
            get_zeroshot_weights=(not (opt.load_ckpt or opt.load_run_name)),
        )
    else:
        raise NotImplementedError()
    if opt.freeze_backbone:
        model.freeze_backbone()
    if world_size > 1:
        if opt.sync_bn:
            logger.info('==> Using SyncBN')
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, device_ids=[local_rank])
    models = {'model': model}

    # Criterions
    criterions = {}
    criterions['classification'] = nn.CrossEntropyLoss()
    for criterion in criterions.values():
        criterion.to(device)

    # Optimizer
    bare_model = model.module if world_size > 1 else model
    head_parameters = [
        p for n, p in model.named_parameters() if 'backbone' not in n
    ]
    if opt.freeze_backbone:
        parameters = head_parameters
    elif opt.lr_bb is not None:
        parameters = [
            {'params': bare_model.backbone.parameters(), 'lr': opt.lr_bb},
            {'params': head_parameters}
        ]
    else:
        parameters = model.parameters()
    optimizer = get_optim(
        parameters=parameters,
        optimizer_name=opt.optimizer,
        lr=opt.lr,
        schedule=opt.lr_schedule,
        weight_decay=opt.weight_decay,
        num_epochs=opt.epoch,
        num_iters_train=num_iters_train,
        cyclic_stepsize=opt.cyclic_step,
        onecycle_pct_start=opt.onecycle_pct_start,
        multistep_milestones=opt.multistep_milestones,
        adam_beta=opt.adam_beta,
    )
    optimizers = {'optimizer': optimizer}
    
    # Load
    resume_ckpt = None
    bare_model = model.module if world_size > 1 else model
    if opt.auto_resume:
        assert opt.resume_ckpt is None
        load_path = os.path.join(manager.get_checkpoint_dir(), 'ckpt-last.pt')
        if os.path.exists(load_path):
            opt.resume_ckpt = 'ckpt-last.pt'
    if opt.resume_ckpt:
        load_path = os.path.join(manager.get_checkpoint_dir(), opt.resume_ckpt)
        logger.info(f'==> Resume from checkpoint {load_path}')
        resume_ckpt = torch.load(load_path, map_location='cpu')
    elif opt.load_ckpt or opt.load_run_name:
        if opt.load_ckpt:
            load_path = opt.load_ckpt
        else:
            ckpt_dir = manager.get_checkpoint_dir(
                opt.load_run_name, opt.load_run_number)
            load_path = os.path.join(ckpt_dir, 'ckpt-last.pt')
        logger.info(f'==> Loading model from {load_path}')
        checkpoint = torch.load(load_path, map_location='cpu')
        bare_model.load_state_dict(checkpoint['model'])
        if opt.wise_base_run_name:
            ckpt_dir = manager.get_checkpoint_dir(
                opt.wise_base_run_name, opt.wise_base_run_number)
            load_path = os.path.join(ckpt_dir, 'ckpt-last.pt')
            logger.info(f'==> Loading WiSE base model from {load_path}')
            base_state_dict = torch.load(load_path, map_location='cpu')['model']
            model_state_dict = checkpoint['model']
            wise_state_dict = {
                name: (1 - opt.wise_alpha) * base_state_dict[name] + \
                    opt.wise_alpha * model_state_dict[name]
                for name in model_state_dict
            }
            bare_model.load_state_dict(wise_state_dict)
    elif opt.load_pretrained:
        if not use_clip:
            logger.info(f'==> Loading pretrained backbone from {opt.load_pretrained}')
            pretrained_dict = torch.load(opt.load_pretrained, map_location='cpu')
            bare_model.backbone.load_pretrained(pretrained_dict)
    else:
        logger.info(f'==> Will train from scratch')

    # Trainer
    loop_configs = [
        StandardLoopConfig('train', dataset, trainloader,
                           training=True, n_iterations=num_iters_train,
                           n_computation_steps=opt.accum_steps),
        StandardLoopConfig('val', dataset, valloader,
                           training=False, n_iterations=num_iters_val,
                           for_best_meter=True),
        StandardLoopConfig('test-trainset', dataset, raw_trainloader,
                           training=False, n_iterations=num_iters_trainset_test,
                           run_every_n_epochs=opt.trainset_test_period,
                           run_at_checkpoint=False),
        StandardLoopConfig('test-testset', dataset, testloader,
                           training=False, n_iterations=num_iters_test,
                           run_every_n_epochs=opt.test_period),
    ]
    trainer = StandardTrainer(
        manager=manager,
        models=models,
        criterions=criterions,
        n_epochs=opt.epoch,
        loop_configs=loop_configs,
        optimizers=optimizers,
        log_period=opt.log_period,
        ckpt_period=opt.ckpt_period,
        device=device,
        keep_eval_mode=opt.freeze_backbone,
        resume_ckpt=resume_ckpt,
        num_classes=dataset.num_classes,
    )

    trainer.train()


if __name__ == "__main__":
    # Set the environment variables if not launched by torchrun
    if 'RANK' not in os.environ:
        os.environ['RANK'] = '0'
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = os.environ['RANK']
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = '1'
    if 'LOCAL_WORLD_SIZE' not in os.environ:
        os.environ['LOCAL_WORLD_SIZE'] = os.environ['WORLD_SIZE']
    main()
