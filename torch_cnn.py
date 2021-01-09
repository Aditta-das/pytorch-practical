# import all dependencies
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
# Simple CNN
# class CNN(nn.Module):
# 	def __init__(self, in_channels=1, num_classes=10):
# 		super(CNN, self).__init__()
# 		self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# 		self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
# 		self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# 		self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

# 	def forward(self, x):
# 		x = F.relu(self.conv1(x))
# 		x = self.pool(x)
# 		x = F.relu(self.conv2(x))
# 		x = self.pool(x)
# 		x = x.reshape(x.shape[0], -1)
# 		x = self.fc1(x)

# 		return x

# def save_checkpoint(state, filename="myCheck.pth.tar"):
# 	print("Save checkpoint")
# 	torch.save(state, filename)

# # set device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Hyparameter
# in_channels = 1
# num_classes = 10
# learning_rate = 0.001
# batch_size = 64
# num_epochs = 5
# load_model = True

# # Load Data
# train_dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
# test_dataset = datasets.MNIST(root="dataset/", train=False, transform=transforms.ToTensor(), download=True)
# train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# # Initialization network
# model = CNN().to(device)

# # Loss and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# # Train Network
# for epoch in range(num_epochs):
# 	losses = []
# 	for batch_idx, (data, targets) in enumerate(train_loader):
# 		data = data.to(device=device)
# 		targets = targets.to(device=device)

# 		# forward
# 		scores = model(data)
# 		loss = criterion(scores, targets)

# 		# backward
# 		optimizer.zero_grad()
# 		loss.backward()
# 		optimizer.step()

# # check accuracy
# def check_accuracy(loader, model):
# 	if loader.dataset.train:
# 		print("Train Accuracy")
# 	else:
# 		print("Test Accuracy")

# 	num_correct = 0
# 	num_samples = 0
# 	model.eval()

# 	with torch.no_grad():
# 		for x, y in loader:
# 			x = x.to(device)
# 			y = y.to(device)

# 			scores = model(x)
# 			_, predictions = scores.max(1)
# 			num_correct += (predictions == y).sum()
# 			num_samples += predictions.size(0)

# 		print(
#             f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
#         )

# 	model.train()

# check_accuracy(train_loader, model)
# check_accuracy(test_loader, model)

dataset = ImageFolder("custom/")
print(dataset)

from PIL import Image

class ImageLoader(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __getitem__(self, item):
        image = Image.open(self.dataset[item][0])
        return image, self.dataset[item][1]
    
    def __len__(self):
        return len(self.dataset)

    
images = ImageLoader(trainData)
print(images[0])