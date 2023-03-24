from .clip import load as load_clip
from .zeroshot import get_zeroshot_classifier, ClassificationHead

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from models.classification import ClassificationModel

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def get_clip_model(arch, dataset, variant, model_dir, device,
                   get_zeroshot_weights=False):
    num_classes = dataset.num_classes
    clip_model, _ = load_clip(
        name=arch[5:],
        device=device,
        download_root=model_dir,
    )
    clip_model.float()
    dtype = clip_model.dtype
    def clip_preprocess(images):
        images = images.type(dtype)
        return TF.normalize(images, CLIP_MEAN, CLIP_STD)
    if variant == 'std':        # random init linear classifier
        backbone = clip_model.visual
        classifier = nn.Linear(backbone.output_dim, num_classes, device=device, dtype=dtype)
        model = ClassificationModel(
            num_classes=num_classes,
            backbone=backbone,
            classifier=classifier,
            preprocess_fn=clip_preprocess,
            use_dataset_preprocess=False,
        ).to(device)
    elif variant == 'zeroshot':
        backbone = clip_model.visual
        if get_zeroshot_weights:
            template = 'openai_imagenet_template'
            classifier = get_zeroshot_classifier(
                dataset, clip_model, device,
                template=template,
            )
        else:
            classifier = ClassificationHead(
                normalize=True,
                weights=torch.zeros((num_classes, backbone.output_dim)),
            )
        model = ClassificationModel(
            num_classes=num_classes,
            backbone=backbone,
            classifier=classifier,
            preprocess_fn=clip_preprocess,
            use_dataset_preprocess=False,
        ).to(device)
    else:
        raise NotImplementedError()
    return model
