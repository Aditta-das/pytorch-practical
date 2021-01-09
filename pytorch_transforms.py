# import all dependencies
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# simple CNN
class CNN(nn.Module):
	def __init__(self, in_channels, num_classes):
		super(CNN, self).__init__()
		self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, stride=(1, 1), kernel_size=(3, 3), padding=(1, 1))
		self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		self.fc1 = nn.Linear(16 * 8 * 8, num_classes)
	#forward
	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = self.pool(x)
		x = F.relu(self.conv2(x))
		x = self.pool(x)
		x = x.reshape(x.shape[0], -1)
		x = self.fc1(x)
		return x

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameter
learning_rate = 1e-4
batch_size = 64
num_epochs = 5

# Load model
model = CNN(in_channels=3, num_classes=10)
model.classifier = nn.Sequential(nn.Linear(512, 100), nn.ReLU(), nn.Linear(100, 10))
model.to(device)

# Load data
my_transforms = transforms.Compose(
	[
		transforms.Resize((36, 36)), # Resize to (32, 32) to (36, 36)
		transforms.RandomCrop((32, 32)), # Take a random (32, 32) crop
		transforms.ColorJitter(brightness=0.5), # change brightness of image
		transforms.RandomRotation(degree=45),
		transforms.RandomHorizontalFlip(p=0.5),
		transforms.RandomVerticalFlip(p=0.05),
		transforms.RandomGrayscale(p=0.2),
		transforms.ToTensor(),
		transforms.Normalize(
				mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
		),
	]
)

train_dataset = datasets.CIFAR10(root="dataset/", train=True, download=True, transform=my_transforms)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Train Network
for epoch in range(num_epochs):
	losses = []
	for batch_idx, (data, targets) in enumerate(train_loader):
		data = data.to(device=device)
		targets = targets.to(device=device)

		# forward
		scores = model(data)
		loss = criterion(scores, targets)
		losses.append(loss)
		# backward
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	print(f"Cost at epoch {epoch} is {sum(losses)/len(losses):.5f}")
# Check Accuracy

