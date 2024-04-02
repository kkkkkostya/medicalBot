from captum.attr import GuidedGradCam
from captum.attr import LRP
import numpy as np


def intepretationGradCam(net, net_type, image, target):
    guided_gc = GuidedGradCam(net, net.encoder[24] if net_type else net.encoder[16])
    attributions = guided_gc.attribute(image, target=target)
    image = np.transpose(attributions.detach().numpy().squeeze(), (1, 2, 0))
    return image


def intepretationLRP(net, image, target):
    lrp = LRP(net)
    attribution = lrp.attribute(image, target=target)
    image = np.transpose(attribution.detach().numpy().squeeze(), (1, 2, 0))
    return image
