import re
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchattacks.attack import Attack
from torchvision.transforms import InterpolationMode


def get_GradCAM(model, images, labels, upsample=None):
    # Ref: https://github.com/jacobgil/pytorch-grad-cam

    # Get activations and gradients for target layer
    bare_model = model.module if hasattr(model, 'module') else model
    target_layer = bare_model.backbone.transformer.resblocks[-1].ln_1
    activations = []
    def save_activation(module, input, output):
        activations.append(output)
    handle = target_layer.register_forward_hook(save_activation)
    logits = model(images)
    handle.remove()
    loss = logits.gather(1, labels.unsqueeze(1)).sum()
    grad = torch.autograd.grad(loss, activations[0])[0]

    act = activations[0].detach()                       # L * N * C
    weights = grad.sum(dim=0, keepdim=True)             # 1 * N * C
    cam = F.relu(torch.sum(weights * act, dim=2))[1:]   # (L-1) * N
    if upsample:
        Np, N = cam.size()
        s = round(Np ** 0.5)
        cam = cam.T.reshape(N, s, s)
        cam = TF.resize(cam, images.size()[2:], upsample)  # N * H * W
    # cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
    return cam


def get_MMCAM(model, images, labels, upsample=None, get_patch_cam=False):
    # Ref: https://github.com/hila-chefer/Transformer-MM-Explainability

    # Get activations and gradients for each attention map
    attentions = []
    logits = model(images, attn_out=attentions)
    loss = logits.gather(1, labels.unsqueeze(1)).sum()
    grads = torch.autograd.grad(loss, attentions)
    attentions = [attn.detach() for attn in attentions]

    # Compute CAM
    bare_model = model.module if hasattr(model, 'module') else model
    N = images.size(0)
    L = bare_model.backbone.positional_embedding.size(0)
    R = torch.eye(L, device=images.device).repeat(N, 1, 1)
    for attn, grad in zip(attentions, grads):
        A = (grad * attn).clamp(min=0)                  # (N * Nh) * L * L
        A = A.reshape(N, -1, L, L).mean(dim=1)          # N * L * L
        R += torch.matmul(A, R)                         # N * L * L
    patch_cam = R[:, 0, 1:]
    cam = patch_cam.T                                   # (L-1) * N
    if upsample:
        s = round((L - 1) ** 0.5)
        cam = patch_cam.reshape(N, s, s)
        cam = TF.resize(cam, images.size()[2:], upsample)  # N * H * W
    if get_patch_cam:
        return cam, patch_cam
    return cam


class RandMaskNoFill(Attack):

    def __init__(self, model, mask_rate):
        super().__init__("RandMaskNoFill", model)
        self.mask_rate = mask_rate
        bare_model = model.module if hasattr(model, 'module') else model 
        self.n_patch = bare_model.backbone.positional_embedding.size(0) - 1
        self.n_keep = int(self.n_patch * (1 - mask_rate))

    def forward(self, images, labels=None):
        mask = torch.stack([torch.randperm(self.n_patch)[:self.n_keep].to(self.device)
                            for _ in images], dim=1)
        mask = torch.cat([torch.zeros_like(mask[:1]), mask + 1]) # CLS token
        return mask


class CAMMaskNoFill(Attack):

    def __init__(self, model, cam_method, threshold, ctx_mask=False):
        super().__init__("CAMMaskNoFill", model)
        self.method = cam_method
        if cam_method == 'GradCAM':
            self.get_CAM = get_GradCAM
        elif cam_method == 'MMCAM':
            self.get_CAM = get_MMCAM
        else:
            raise NotImplementedError()
        self.threshold = threshold
        self.ctx_mask = ctx_mask
        bare_model = model.module if hasattr(model, 'module') else model 
        self.n_head = bare_model.backbone.heads
        self.n_patch = bare_model.backbone.positional_embedding.size(0) - 1

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        cam = self.get_CAM(self.model, images, labels)
        cam = cam / (cam.amax(dim=0, keepdim=True) + 1e-8)
        mask = (cam > self.threshold)       # (L-1) * N
        # print(torch.count_nonzero(cam > self.threshold, dim=0) / mask.size(0))
        mask = mask.T                       # N * (L-1)
        if self.ctx_mask:
            mask = ~mask
        attn_mask = torch.zeros_like(mask, dtype=float)
        attn_mask[mask] = float('-inf')                 # N * (L-1)
        attn_mask = torch.cat(
            [torch.zeros_like(attn_mask[:, :1]), attn_mask], dim=1)  # N * L
        attn_mask = attn_mask.unsqueeze(1).repeat_interleave(self.n_head, dim=0).repeat(1, self.n_patch + 1, 1)
        return attn_mask


class RandMaskSingleFill(nn.Module):

    def __init__(self, model, prob):
        super().__init__()
        self.prob = prob
        bare_model = model.module if hasattr(model, 'module') else model 
        n_patch = bare_model.backbone.positional_embedding.size(0) - 1
        self.w = round(n_patch ** 0.5)

    def forward(self, images, labels=None):
        N, _, H, W = images.size()
        p = torch.empty(N, 1, self.w, self.w, device=images.device).uniform_()
        mask = TF.resize(p, (H, W), InterpolationMode.NEAREST) < self.prob
        idx = torch.randperm(N, device=images.device)
        return images * (~mask) + images[idx] * (mask)


class CAMMaskSingleFill(Attack):

    def __init__(self, model, cam_method, threshold, ctx_mask=False, save_mask=False):
        super().__init__("CAMMaskSingleFill", model)
        self.method = cam_method
        if cam_method == 'GradCAM':
            self.get_CAM = partial(
                get_GradCAM, upsample=InterpolationMode.NEAREST)
        elif cam_method == 'MMCAM':
            self.get_CAM = partial(
                get_MMCAM, upsample=InterpolationMode.NEAREST, get_patch_cam=save_mask)
        else:
            raise NotImplementedError()
        self.threshold = threshold
        self.ctx_mask = ctx_mask
        self.save_mask = save_mask

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        cam = self.get_CAM(self.model, images, labels)
        if self.save_mask:
            cam, patch_cam = cam
            patch_cam = patch_cam / (patch_cam.amax(dim=1, keepdim=True) + 1e-8)
            patch_mask = (patch_cam > self.threshold)
        cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
        mask = (cam > self.threshold).unsqueeze(1)
        if self.ctx_mask:
            mask = ~mask
            if self.save_mask:
                patch_mask = ~patch_mask
        idx = torch.randperm(images.size()[0], device=self.device)
        images_masked = images * (~mask) + images[idx] * (mask)
        if self.save_mask:
            return images_masked, patch_mask
        return images_masked


class RandMaskMultiFill(nn.Module):

    def __init__(self, model, prob):
        super().__init__()
        self.prob = prob
        bare_model = model.module if hasattr(model, 'module') else model 
        n_patch = bare_model.backbone.positional_embedding.size(0) - 1
        self.w = round(n_patch ** 0.5)

    def forward(self, images, labels=None):
        N, C, H, W = images.size()
        w = self.w
        device = images.device
        p = torch.empty(N, 1, w, w, device=device).uniform_()
        mask = TF.resize(p, (H, W), InterpolationMode.NEAREST) < self.prob
        idx_shift = torch.randint(1, N, (N, 1, w, w), device=device)
        idx = (torch.arange(N, device=device).resize(N, 1, 1, 1) + idx_shift) % N
        # idx = idx.repeat_interleave(H / w, dim=2).repeat_interleave(W / w, dim=3)
        idx = TF.resize(idx, (H, W), InterpolationMode.NEAREST)
        idx = idx.repeat(1, C, 1, 1)
        images_m = images.gather(0, idx)
        return images * (~mask) + images_m * (mask)
        

class CAMMaskMultiFill(Attack):

    def __init__(self, model, cam_method, threshold, ctx_mask=False):
        super().__init__("CAMMaskMultiFill", model)
        self.method = cam_method
        self.threshold = threshold
        bare_model = model.module if hasattr(model, 'module') else model 
        n_patch = bare_model.backbone.positional_embedding.size(0) - 1
        self.w = round(n_patch ** 0.5)
        if cam_method == 'GradCAM':
            self.get_CAM = partial(
                get_GradCAM, upsample=InterpolationMode.NEAREST)
        elif cam_method == 'MMCAM':
            self.get_CAM = partial(
                get_MMCAM, upsample=InterpolationMode.NEAREST)
        else:
            raise NotImplementedError()
        self.ctx_mask = ctx_mask

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        cam = self.get_CAM(self.model, images, labels)
        cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
        N, C, H, W = images.size()
        w = self.w
        device = self.device
        mask = (cam > self.threshold).unsqueeze(1)
        if self.ctx_mask:
            mask = ~mask

        idx_shift = torch.randint(1, N, (N, 1, w, w), device=device)
        idx = (torch.arange(N, device=device).resize(N, 1, 1, 1) + idx_shift) % N
        idx = TF.resize(idx, (H, W), InterpolationMode.NEAREST)
        idx = idx.repeat(1, C, 1, 1)
        images_m = images.gather(0, idx)

        images_masked = images * (~mask) + images_m * (mask)
        return images_masked


def get_masking(name: str = None, **kwargs):

    cam_method = 'MMCAM'
    def get_params(s):
        # return the string inside the brackets
        return re.search(r'\((.*?)\)', s).group(1)

    if name is None or name == 'none':
        return None

    elif name.startswith('RandMaskNoFill'):    # RandMaskNoFill(mask_rate)
        mask_rate = float(get_params(name))
        return RandMaskNoFill(kwargs['model'], mask_rate)
    elif name.startswith('ObjMaskNoFill'):     # ObjMaskNoFill(threshold)
        threshold = get_params(name)
        return CAMMaskNoFill(kwargs['model'], cam_method, float(threshold))
    elif name.startswith('CtxMaskNoFill'):     # CtxMaskNoFill(threshold)
        threshold = get_params(name)
        return CAMMaskNoFill(kwargs['model'], cam_method, float(threshold), ctx_mask=True)

    elif name.startswith('RandMaskSingleFill'):   # RandMaskSingleFill(prob)
        prob = float(get_params(name))
        return RandMaskSingleFill(kwargs['model'], prob)
    elif name.startswith('ObjMaskSingleFill'):    # ObjMaskSingleFill(threshold)
        threshold = get_params(name)
        return CAMMaskSingleFill(kwargs['model'], cam_method, float(threshold), save_mask=kwargs['save_mask'])
    elif name.startswith('CtxMaskSingleFill'):    # CtxMaskSingleFill(threshold)
        threshold = get_params(name)
        return CAMMaskSingleFill(kwargs['model'], cam_method, float(threshold), ctx_mask=True)

    elif name.startswith('RandMaskMultiFill'):    # RandMaskMultiFill(prob)
        prob = float(get_params(name))
        return RandMaskMultiFill(kwargs['model'], prob)
    elif name.startswith('ObjMaskMultiFill'):     # ObjMaskMultiFill(threshold)
        threshold = get_params(name)
        return CAMMaskMultiFill(kwargs['model'], cam_method, float(threshold))
    elif name.startswith('CtxMaskMultiFill'):     # CtxMaskMultiFill(threshold)
        threshold = get_params(name)
        return CAMMaskMultiFill(kwargs['model'], cam_method, float(threshold), ctx_mask=True)

    else:
        raise NotImplementedError()