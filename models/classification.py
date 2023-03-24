import torch.nn as nn

class ClassificationModel(nn.Module):

    def __init__(self, num_classes, backbone, classifier,
                 preprocess_fn=None, use_dataset_preprocess=True):
        super().__init__()
        self.num_classes = num_classes
        if preprocess_fn is not None:
            self.preprocess_fn = preprocess_fn
        else:
            self.preprocess_fn = lambda x: x
        self.use_dataset_preprocess = use_dataset_preprocess
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, images, get_logits=True, get_features=False, **kwargs):
        x = self.preprocess_fn(images)
        f = self.backbone(x, **kwargs)
        if not get_logits:
            return f
        y = self.classifier(f)
        if get_features:
            return y, f
        else:
            return y

    def freeze_backbone(self, freeze=True):
        self.backbone.requires_grad_(not freeze)

    def set_preprocess(self, preprocess_fn):
        self.preprocess_fn = preprocess_fn
