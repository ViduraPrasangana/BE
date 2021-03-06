import torch
import torch.nn as nn
import torchvision.models as models
from .chexnet import DenseNet121
import os

visual_extractor = "chexnet"
chexnet_checkpoint = ""
visual_extractor_pretrained = False

class VisualExtractor(nn.Module):
    def __init__(self):
        super(VisualExtractor, self).__init__()
        self.visual_extractor = visual_extractor
        self.pretrained = visual_extractor_pretrained
        if(self.visual_extractor == "chexnet"):
            model = DenseNet121()
            if os.path.isfile(chexnet_checkpoint):
                print("=> loading checkpoint")
                checkpoint = torch.load(chexnet_checkpoint)
                state_dict = checkpoint['state_dict']
                for key in list(state_dict.keys()):
                    state_dict[key[7:].replace('.1.', '1.'). replace('.2.', '2.')] = state_dict.pop(key)
                model.load_state_dict(state_dict,strict=False)
                print("=> loaded checkpoint")
            else:
                print("=> no checkpoint found")
            model = model.get_model()
            modules = list(model.children())[:-1]
        else:
            model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
            modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)
        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

    def forward(self, images):
        patch_feats = self.model(images)
        # print("model out",patch_feats.size())
        avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        return patch_feats, avg_feats
