import torch
from tqdm import tqdm
import numpy as np

from . import templates, clip


class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)


def get_zeroshot_classifier(dataset=None, clip_model=None, device=None,
                            template='openai_imagenet_template'):
    template = getattr(templates, template)
    logit_scale = clip_model.logit_scale
    clip_model.eval()
    clip_model.to(device)

    print('Getting zeroshot weights.')
    with torch.no_grad():
        zeroshot_weights = []
        for class_name in tqdm(dataset.class_names):
            texts = []
            for t in template:
                texts.append(t(class_name))
            texts = clip.tokenize(texts).to(device) # tokenize
            embeddings = clip_model.encode_text(texts) # embed with text encoder
            embeddings /= embeddings.norm(dim=-1, keepdim=True)

            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()

            zeroshot_weights.append(embeddings)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)

        zeroshot_weights *= logit_scale.exp()
        
        zeroshot_weights = zeroshot_weights.squeeze().float()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)

    return ClassificationHead(normalize=True, weights=zeroshot_weights)
