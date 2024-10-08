import torch
import torch.nn.functional as F
import math
from typing import Any

from torch import Tensor
from torch import nn
from torch.nn import functional as F_torch
from torchvision import models
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor

def cross_entropy(input, target, weight=None, reduction='mean',ignore_index=255):
#def cross_entropy(input, target, weight=torch.tensor([0.01, 0.49, 0.50]), reduction='mean',ignore_index=255):
    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """
    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)

    return F.cross_entropy(input=input, target=target, weight=weight,
                           ignore_index=ignore_index, reduction=reduction)


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


def StyleLoss(input,target):

    G_input = gram_matrix(input)
    G_target = gram_matrix(target)
    loss = F.mse_loss(G_input, G_target,reduction='sum')
    return loss


def content_loss(input,target,mean,std):

    feature_model_extractor_node = 'features.35'
    #feature_model_normalize_mean = [0.485, 0.456, 0.406]
    #feature_model_normalize_std = [0.229, 0.224, 0.225]
    feature_model_normalize_mean=mean 
    feature_model_normalize_std =std
  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    #model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    model = models.vgg19(pretrained=True)
    model = model.to(device)

    feature_extractor = create_feature_extractor(model, [feature_model_extractor_node])
    normalize = transforms.Normalize(feature_model_normalize_mean, feature_model_normalize_std)

    sr_tensor = normalize(input)
    gt_tensor = normalize(target)

    sr_feature = feature_extractor(sr_tensor)[feature_model_extractor_node]
    gt_feature = feature_extractor(gt_tensor)[feature_model_extractor_node]

    loss = F_torch.mse_loss(sr_feature, gt_feature)
    return loss
