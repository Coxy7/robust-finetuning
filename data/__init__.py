from .imagenet import ImageNetDataset
from .imagenet_a import ImageNetADataset
from .imagenet_r import ImageNetRDataset
from .imagenet_sketch import ImageNetSketchDataset
from .imagenet_v2 import ImageNetV2Dataset
from .objectnet import ObjectNetDataset


def get_dataset(name, data_dir, **kwargs):
    if name == 'imagenet':
        return ImageNetDataset(data_dir, **kwargs)
    elif name == 'imagenet_a':
        return ImageNetADataset(data_dir, **kwargs)
    elif name == 'imagenet_r':
        return ImageNetRDataset(data_dir, **kwargs)
    elif name == 'imagenet_sketch':
        return ImageNetSketchDataset(data_dir, **kwargs)
    elif name == 'imagenet_v2':
        return ImageNetV2Dataset(data_dir, **kwargs)
    elif name == 'objectnet':
        return ObjectNetDataset(data_dir, **kwargs)
    else:
        raise NotImplementedError()
