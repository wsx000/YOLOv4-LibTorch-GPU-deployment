import torchvision.models as models
from torchvision import transforms as transform
from nets.yolo4 import YoloBody
import torch
import cv2
import numpy as np
import os

model = YoloBody(num_anchors=3, num_classes=2)
model.load_state_dict(torch.load(r"logs/Epoch60.pth", map_location=torch.device('cuda')))
model.eval()
model.to(device='cuda')
sample = torch.ones(1, 3, 608, 608).to(device='cuda')
traced_script_module = torch.jit.trace(model, sample)
traced_script_module.save("/home/curious/code/yolo4-cpp/111.pt")