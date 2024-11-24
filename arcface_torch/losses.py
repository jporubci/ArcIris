'''
https://github.com/deepinsight/insightface/blob/337990aa3adb67de4f8f325d995ebdea46773677/recognition/arcface_torch/losses.py

@inproceedings{deng2019arcface,
  title={Arcface: Additive angular margin loss for deep face recognition},
  author={Deng, Jiankang and Guo, Jia and Xue, Niannan and Zafeiriou, Stefanos},
  booktitle={CVPR},
  year={2019}
}
@inproceedings{an2022partialfc,
    author={An, Xiang and Deng, Jiankang and Guo, Jia and Feng, Ziyong and Zhu, XuHan and Yang, Jing and Liu, Tongliang},
    title={Killing Two Birds With One Stone: Efficient and Robust Training of Face Recognition CNNs by Partial FC},
    booktitle={CVPR},
    year={2022},
}
@inproceedings{zhu2021webface260m,
  title={Webface260m: A benchmark unveiling the power of million-scale deep face recognition},
  author={Zhu, Zheng and Huang, Guan and Deng, Jiankang and Ye, Yun and Huang, Junjie and Chen, Xinze and Zhu, Jiagang and Yang, Tian and Lu, Jiwen and Du, Dalong and Zhou, Jie},
  booktitle={CVPR},
  year={2021}
}
'''

import torch
import math


class CombinedMarginLoss(torch.nn.Module):
    def __init__(self, s, m1, m2, m3, interclass_filtering_threshold=0):
        super().__init__()
        self.s = s
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.interclass_filtering_threshold = interclass_filtering_threshold
        
        # For ArcFace
        self.cos_m = math.cos(self.m2)
        self.sin_m = math.sin(self.m2)
        self.theta = math.cos(math.pi - self.m2)
        self.sinmm = math.sin(math.pi - self.m2) * self.m2
        self.easy_margin = False


    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        indexes = torch.arange(logits.size(0))

        if self.interclass_filtering_threshold > 0:
            with torch.no_grad():
                dirty = logits > self.interclass_filtering_threshold
                dirty = dirty.float()
                mask = torch.ones([indexes.size(0), logits.size(1)], device=logits.device)
                mask.scatter_(1, labels[indexes], 0)
                dirty[indexes] *= mask
                tensor_mul = 1 - dirty    
            logits = tensor_mul * logits

        target_logit = logits[indexes, labels[indexes].view(-1)]

        if self.m1 == 1.0 and self.m3 == 0.0:
            with torch.no_grad():
                target_logit.arccos_()
                logits.arccos_()
                final_target_logit = target_logit + self.m2
                logits[indexes, labels[indexes].view(-1)] = final_target_logit
                logits.cos_()
            logits = logits * self.s        

        elif self.m3 > 0:
            final_target_logit = target_logit - self.m3
            logits[indexes, labels[indexes].view(-1)] = final_target_logit
            logits = logits * self.s
        else:
            raise

        return logits

""" ArcFace (https://arxiv.org/pdf/1801.07698v4): """
class ArcFace(torch.nn.Module):
    def __init__(self, s=64.0, margin=0.5):
        super(ArcFace, self).__init__()
        self.s = s
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin
        self.easy_margin = False


    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        indexes = torch.arange(logits.size(0))
        target_logit = logits[indexes, labels[indexes].view(-1)]

        with torch.no_grad():
            target_logit.arccos_()
            logits.arccos_()
            final_target_logit = target_logit + self.margin
            logits[indexes, labels[indexes].view(-1)] = final_target_logit
            logits.cos_()
        logits = logits * self.s
        return logits


class CosFace(torch.nn.Module):
    def __init__(self, s=64.0, m=0.40):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        indexes = torch.arange(logits.size(0))
        target_logit = logits[indexes, labels[indexes].view(-1)]
        final_target_logit = target_logit - self.m
        logits[indexes, labels[indexes].view(-1)] = final_target_logit
        logits = logits * self.s
        return logits