# import 
import torch 
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import os
from torch.utils.data import (DataLoader, Dataset)
import pandas as pd
import numpy as np
import cv2

class CatsAndDogsDataset(Dataset):
	def __init__(self, csv_file, root_dir, transform=None):
		self.annontations = pd.read_csv(csv_file)
		self.root_dir = root_dir
		self.transform = transform

	def __len__(self):
		return len(self.annontations)

	def __getitem__(self, index):
		img_path = os.path.join(self.root_dir, self.annontations.iloc[index, 0])
		image = cv2.imread(img_path)
		y_label = int(torch.tensor(self.annontations.iloc[index, 1]))

		if self.transform:
			image = self.transform(image)

		return image, y_label
