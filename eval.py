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


def add_parser_argument(parser):
    ## ======================== Data ==========================
    parser.add_argument('--dataset', type=str,
        default='imagenet,imagenet_v2,imagenet_r,imagenet_sketch,objectnet,imagenet_a')
    parser.add_argument('--train_split', default='original', type=str)
    parser.add_argument('--val_size', default=10240, type=int)
    parser.add_argument('--data_split_seed', default=0, type=int)
    parser.add_argument('--batch', default=512, type=int)
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--image_size', default=224, type=int)
    parser.add_argument('--transform', default='clip', type=str)
    parser.add_argument('--same_ignored_classes', action='store_true')
    ## ======================= Model ==========================
    parser.add_argument('--arch', type=str)
    parser.add_argument('--arch_variant', default='zeroshot', type=str)
    parser.add_argument('--load_pretrained', type=str)
    parser.add_argument('--load_ckpt', type=str)
    parser.add_argument('--load_run_name', type=str)
    parser.add_argument('--load_run_number', type=str)
    parser.add_argument('--load_run_ckpt_name', type=str, default='ckpt-best')
    parser.add_argument('--freeze_backbone', action='store_true')
    ## ===================== Evaluation =======================
    parser.add_argument('--num_iters_test', type=int,
                        help="default: len(testloader)")
    ## ====================== Logging =========================
    parser.add_argument('--log_period', default=5, type=int, metavar='LP',
                        help='log every LP iterations')
    parser.add_argument('--ckpt_period', type=int, metavar='CP',
                        help='make checkpoints every CP epochs')
    parser.add_argument('--comment', default='', type=str)
    ## ==================== Experimental ======================


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
    assert opt.batch % world_size == 0
    batch = opt.batch // world_size
    data_kwargs = dict(
        batch_size=batch, num_workers=opt.num_workers, with_index=False,
        train_split=opt.train_split, val_size=opt.val_size,
        split_seed=opt.data_split_seed,
        world_size=world_size, rank=rank)
    dataset_names = opt.dataset.split(',')
    datasets = []
    testloaders = []
    num_classes = None
    ignored_classes = set()     # get the union of ignored classes
    for dataset_name in dataset_names:
        dataset = get_dataset(
            dataset_name, opt.data_dir, size=opt.image_size, transform=opt.transform)
        loader = dataset.get_loader(**data_kwargs)
        testloader = loader[-1] if isinstance(loader, (list, tuple)) else loader
        datasets.append(dataset)
        testloaders.append(testloader)
        num_classes = num_classes or dataset.num_classes
        assert num_classes == dataset.num_classes
        if hasattr(dataset, 'ignored_classes'):
            ignored_classes |= set(dataset.ignored_classes)
    if opt.same_ignored_classes:
        ignored_classes = list(ignored_classes)
    else:
        ignored_classes = None

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
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # Load
    bare_model = model.module if world_size > 1 else model
    if opt.load_ckpt:
        load_path = opt.load_ckpt
    else:
        ckpt_dir = manager.get_checkpoint_dir(
            opt.load_run_name, opt.load_run_number)
        load_path = os.path.join(ckpt_dir, f'{opt.load_run_ckpt_name}.pt')
    logger.info(f'==> Loading model from {load_path}')
    checkpoint = torch.load(load_path, map_location='cpu')
    bare_model.load_state_dict(checkpoint['model'])

    # Trainer
    loop_configs = []
    for dataset_name, dataset, testloader in zip(dataset_names, datasets, testloaders):
        if isinstance(testloader, dict):
            for split_name, split_loader in testloader.items():
                num_iters_test = parse(opt.num_iters_test, len(split_loader))
                config = StandardLoopConfig(
                    f'{dataset_name}-{split_name}', dataset, split_loader,
                    training=False, n_iterations=num_iters_test)
                loop_configs.append(config)
        else:
            num_iters_test = parse(opt.num_iters_test, len(testloader))
            config = StandardLoopConfig(
                dataset_name, dataset, testloader,
                training=False, n_iterations=num_iters_test)
            loop_configs.append(config)
    trainer = StandardTrainer(
        manager=manager,
        models={'model': model},
        criterions={},
        n_epochs=1,
        loop_configs=loop_configs,
        optimizers={},
        log_period=opt.log_period,
        ckpt_period=opt.ckpt_period,
        device=device,
        num_classes=num_classes,
        ignored_classes=ignored_classes,
    )

    trainer.test()


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
