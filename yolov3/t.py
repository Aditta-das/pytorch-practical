import numpy as np 
import os
import pandas as pd
import torch

from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from utils import (
    cells_to_bboxes,
    iou_width_height as iou,
    non_max_suppression as nms,
    plot_image
)


label_path = "../yolov3/Pascal/labels/000007.txt"

bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()

print(bboxes)

for box in bboxes:
	iou_anchors = torch.tensor([0.639, 0.5675675675675675, 0.718, 0.8408408408408409])
	print(iou_anchors)