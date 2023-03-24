import torch
import torch.utils.data
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataset import Subset
import torchvision.transforms.functional as TF
from sklearn.model_selection import train_test_split


class DatasetWithIndex(torch.utils.data.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self._dataset = dataset
    
    def __getitem__(self, index):
        item = self._dataset[index]
        return item, index

    def __len__(self):
        return len(self._dataset)

    def __getattr__(self, name):
        return getattr(self._dataset, name)



def get_dataloader(dataset, shuffle=False, drop_last=False, with_index=False, num_replicas=1, rank=0, **kwargs):
    if with_index:
        dataset = DatasetWithIndex(dataset)
    if num_replicas > 1:
        sampler = DistributedSampler(
            dataset, num_replicas, rank, shuffle=shuffle)
        loader = torch.utils.data.DataLoader(
            dataset, sampler=sampler, drop_last=drop_last, **kwargs)
    else:
        loader = torch.utils.data.DataLoader(
            dataset, shuffle=shuffle, drop_last=drop_last, **kwargs)
    return loader


def stratified_random_split(dataset, labels, train_split, seed=7):
    indices = list(range(len(dataset)))
    indices_train, indices_test = train_test_split(
        indices,
        train_size=train_split,
        random_state=seed,
        stratify=labels,
    )
    trainset = Subset(dataset, indices_train)
    testset = Subset(dataset, indices_test)
    return trainset, testset


class BaseDataset():
    
    def __init__(self, data_dir, size=None, mean=None, std=None):
        self.data_dir = data_dir
        self.size = size
        self.mean = mean
        self.std = std

    def get_loader(self, batch_size, num_workers, with_index=False):
        raise NotImplementedError()
    
    def preprocess(self, images):
        if self.mean:
            return TF.normalize(images, self.mean, self.std)
        return images
