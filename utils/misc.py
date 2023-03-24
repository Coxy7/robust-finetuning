import torch
import torch.distributed as dist


class Accuracy():
    
    def __init__(self, num_classes=None, involved_classes=None,
                 ignored_classes=None, reduction='mean', denominator=100):
        if ignored_classes:
            assert involved_classes is None
            self.ignored_classes = ignored_classes
            ignored = set(ignored_classes)
            self.involved_classes = [i for i in range(num_classes)
                                     if i not in ignored]
        elif involved_classes:
            assert ignored_classes is None
            self.involved_classes = involved_classes
            involved = set(involved_classes)
            self.ignored_classes = [i for i in range(num_classes)
                                    if i not in involved]
        else:
            self.involved_classes = list(range(num_classes))
            self.ignored_classes = []
        self.involved_classes_set = set(self.involved_classes)
        self.reduction = reduction
        self.denominator = denominator

    def __call__(self, outputs, labels, reduction=None):
        outputs = outputs.detach().clone()
        labels = labels.detach().clone()

        # mask out samples with ignored classes
        if labels.dim() == 1:                   # regular labels
            mask = labels.clone().cpu().apply_(
                lambda c: c in self.involved_classes_set)
            mask = mask.to(dtype=bool, device=labels.device)
        else:                                   # one-hot / multi labels
            mask = (labels[:, self.involved_classes].sum(-1) > 0)
        outputs = outputs[mask]
        labels = labels[mask]

        # predict from outputs
        if outputs.dim() == 1:                  # hard predictions
            predictions = outputs
        else:                                   # logits / soft predictions
            if len(outputs):
                outputs[:, self.ignored_classes] = outputs.min() - 1
            predictions = outputs.argmax(-1)
        
        # decide correctness
        if labels.dim() == 1:
            correct = (predictions == labels)
        else:
            correct = labels.gather(1, predictions.unsqueeze(1))

        # produce results
        correct = correct.float() * self.denominator
        if reduction is None:
            reduction = self.reduction
        if reduction == 'sum':
            return correct.sum(), len(correct)
        elif reduction == 'mean':
            return correct.mean()
        elif reduction == 'none':
            return correct


class ScalerMeter(object):

    def __init__(self):
        self.x = None

    def update(self, x):
        if not isinstance(x, (int, float)):
            x = x.item()
        self.x = x

    def reset(self):
        self.x = None

    def get_value(self):
        if self.x:
            return self.x
        return 0

    def sync(self, device):
        pass


class AverageMeter(object):

    def __init__(self):
        self.sum = 0
        self.n = 0

    def update(self, x, n=1):
        self.sum += float(x)
        self.n += int(n)

    def reset(self):
        self.sum = 0
        self.n = 0

    def get_value(self):
        if self.n:
            return self.sum / self.n
        return 0

    def sync(self, device):
        t = torch.tensor([self.sum, self.n],
                         dtype=torch.float32, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        self.sum = t[0].item()
        self.n = round(t[1].item())


class MovingAverageMeter(object):

    def __init__(self, decay=0.95):
        self.x = None
        self.decay = decay

    def update(self, x, n=1):
        if n > 0:
            x = float(x) / int(n)
            if self.x is None:
                self.x = x
            else:
                self.x = self.x * self.decay + x * (1 - self.decay)

    def reset(self):
        self.x = None

    def get_value(self):
        if self.x:
            return self.x
        return 0

    def sync(self, device):
        if self.x is not None:
            t = torch.tensor([self.x], dtype=torch.float32, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            self.x = t[0].item() / dist.get_world_size()


class PerClassMeter(object):

    def __init__(self, meter, num_classes=None, **kwargs):
        self.meter = meter
        self.num_classes = num_classes or 0
        self.kwargs = kwargs
        self.meters = [meter(**kwargs) for _ in range(self.num_classes)]

    def update(self, x, y):
        n = int(max(y))
        if n > self.num_classes:
            self.meters += [self.meter(**self.kwargs)
                            for _ in range(n - self.num_classes)]
            self.num_classes = n
        for i in range(self.num_classes):
            mask = (y == i)
            self.meters[i].update(sum(x[mask]), sum(mask))

    def reset(self):
        for meter in self.meters:
            meter.reset()

    def get_value(self, per_class_avg=True):
        values = [meter.get_value() for meter in self.meters]
        if per_class_avg:
            return sum(values) / len(values)
        else:
            return values

    def sync(self, device):
        for meter in self.meters:
            meter.sync(device)


def consume_prefix_in_state_dict_if_present(state_dict, prefix):
    r"""Strip the prefix in state_dict, if any.
    ..note::
        Given a `state_dict` from a DP/DDP model, a local model can load it by applying
        `consume_prefix_in_state_dict_if_present(state_dict, "module.")` before calling
        :meth:`torch.nn.Module.load_state_dict`.
    Args:
        state_dict (OrderedDict): a state-dict to be loaded to the model.
        prefix (str): prefix.
    """
    keys = sorted(state_dict.keys())
    for key in keys:
        if key.startswith(prefix):
            newkey = key[len(prefix) :]
            state_dict[newkey] = state_dict.pop(key)

    # also strip the prefix in metadata if any.
    if "_metadata" in state_dict:
        metadata = state_dict["_metadata"]
        for key in list(metadata.keys()):
            # for the metadata dict, the key can be:
            # '': for the DDP module, which we want to remove.
            # 'module': for the actual model.
            # 'module.xx.xx': for the rest.

            if len(key) == 0:
                continue
            newkey = key[len(prefix) :]
            metadata[newkey] = metadata.pop(key)

def parse(arg, default):
    if arg is None:
        return default
    return arg
