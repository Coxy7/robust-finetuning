from .imagenet import EvaluationDataset


# https://github.com/modestyachts/ImageNetV2/issues/6
IMAGENET_V2_CLASSES = [int(s) for s in sorted([str(i) for i in range(1000)])]


class ImageNetV2Dataset(EvaluationDataset):

    def __init__(self, data_dir, size=224, interpolation='bicubic', transform='std'):
        super().__init__(data_dir, size, interpolation, transform)
        self.sub_dir = 'imagenetv2-matched-frequency'
        self.target_transform = (lambda y: IMAGENET_V2_CLASSES[y])
