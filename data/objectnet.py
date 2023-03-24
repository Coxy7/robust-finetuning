import json
import os
import numpy as np
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from .imagenet import ImageNetDataset
from data.base import get_dataloader


class ImageFolderWithSpecificClasses(ImageFolder):

    def __init__(self, *args, class_idx=None, **kwargs):
        self._specified_class_idx = class_idx
        super().__init__(*args, **kwargs)

    def find_classes(self, directory):
        all_classes, class_to_idx = super().find_classes(directory)
        classes = [all_classes[i] for i in self._specified_class_idx]
        class_set = set(classes)
        for cls in all_classes:
            if cls not in class_set:
                class_to_idx.pop(cls)
        return classes, class_to_idx


class ObjectNetDataset(ImageNetDataset):

    def __init__(self, data_dir, size=224, interpolation='bicubic', transform='std'):
        super().__init__(data_dir, size, interpolation, transform)
        self.root_dir = os.path.join(self.data_dir, 'objectnet-1.0')
        ON_classes, IN_classes, ON_pid_to_IN_pids = \
            self._parse_metadata(self.root_dir)
        self.ON_classes = ON_classes
        self.ignored_classes = [i for i in range(self.num_classes)
                                if i not in IN_classes]
        self.target_transform = self._get_target_transform(ON_pid_to_IN_pids)
        self.transforms_test = transforms.Compose([
            self.crop_red_border,
            self.transforms_test,
        ])
    
    @staticmethod
    def crop_red_border(img):
        width, height = img.size
        cropArea = (2, 2, width - 2, height - 2)
        img = img.crop(cropArea)
        return img

    def _parse_metadata(self, root_dir):
        mapping_dir = os.path.join(root_dir, 'mappings')
        with open(os.path.join(mapping_dir, 'folder_to_objectnet_label.json'), 'r') as f:
            folder_to_ON_label = json.load(f)
            ON_pid_to_ON_label = [v for k, v in sorted(folder_to_ON_label.items())]
        with open(os.path.join(mapping_dir, 'objectnet_to_imagenet_1k.json'), 'r') as f:
            ON_label_to_IN_labels = {ON_label: IN_labels.split('; ')
                                     for ON_label, IN_labels in json.load(f).items()}
        with open(os.path.join(mapping_dir, 'imagenet_to_label_2012_v2'), 'r') as f:
            IN_label_to_IN_id = {v.strip(): i for i, v in enumerate(f)}
        with open(os.path.join(mapping_dir, 'pytorch_to_imagenet_2012_id.json'), 'r') as f:
            IN_pid_to_IN_id = json.load(f)
            IN_id_to_IN_pid = {id: int(pid) for pid, id in IN_pid_to_IN_id.items()}
        num_ON_classes = len(ON_pid_to_ON_label)
        ON_classes = []
        IN_classes = set()
        ON_pid_to_IN_pids = []
        for ON_pid in range(num_ON_classes):
            ON_label = ON_pid_to_ON_label[ON_pid]
            if ON_label in ON_label_to_IN_labels:
                IN_labels = ON_label_to_IN_labels[ON_label]
                IN_ids = [IN_label_to_IN_id[IN_label] for IN_label in IN_labels]
                IN_pids = [IN_id_to_IN_pid[IN_id] for IN_id in IN_ids]
                ON_classes.append(ON_pid)
            else:
                IN_pids = []
            ON_pid_to_IN_pids.append(IN_pids)
            IN_classes |= set(IN_pids)
        return ON_classes, IN_classes, ON_pid_to_IN_pids

    def _get_target_transform(self, ON_pid_to_IN_pids):
        def transform(ON_pid):
            IN_pids = ON_pid_to_IN_pids[ON_pid]
            target = np.zeros(self.num_classes, dtype=int)
            target[IN_pids] = 1
            return target
        return transform

    def get_loader(self, batch_size, num_workers, with_index=False,
                   train_split='original', val_size=0, split_seed=0,
                   shuffle_test=False, augment=True, drop_last=True,
                   world_size=1, rank=0):
        images_dir = os.path.join(self.root_dir, 'images')

        testset = ImageFolderWithSpecificClasses(
            root=images_dir, transform=self.transforms_test,
            target_transform=self.target_transform,
            class_idx=self.ON_classes)

        kwargs = dict(batch_size=batch_size, num_workers=num_workers,
                      with_index=with_index, num_replicas=world_size, rank=rank)
        testloader = get_dataloader(
            testset, shuffle=shuffle_test, drop_last=False, **kwargs)
        return testloader
