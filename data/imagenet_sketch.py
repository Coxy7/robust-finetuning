from .imagenet import EvaluationDataset


class ImageNetSketchDataset(EvaluationDataset):

    def __init__(self, data_dir, size=224, interpolation='bicubic', transform='std'):
        super().__init__(data_dir, size, interpolation, transform)
        self.sub_dir = 'imagenet-sketch'
