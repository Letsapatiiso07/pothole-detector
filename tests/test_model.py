import pytest
import torch

from model import build_model, last_conv_layer
from utils import grad_cam


@pytest.mark.parametrize('backbone', ['mobilenet_v3_small', 'resnet18'])
def test_model_gives_one_logit_and_a_heat_map(backbone):
    torch.manual_seed(0)
    model = build_model(backbone, pretrained=False).eval()
    x = torch.randn(2, 3, 160, 736)
    assert model(x).shape == (2, 1)
    cam = grad_cam(model, last_conv_layer(model), x[:1])
    assert cam.shape == (160, 736)
    assert cam.min() >= 0 and cam.max() <= 1
