import torch.nn as nn
import torch.nn.functional as F


class KnowledgeDistillationLoss(nn.Module):

    def __init__(self, temp=10):
        super().__init__()
        self.criterion_kl = nn.KLDivLoss(reduction='batchmean')
        self.t = temp

    def forward(self, logits, soft_labels):
        kl = (self.t * self.t) * self.criterion_kl(
            F.log_softmax(logits / self.t, dim=1),
            F.softmax(soft_labels / self.t, dim=1)
        )
        return kl
