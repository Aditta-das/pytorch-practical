# import
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from customdataset import CatsAndDogsDataset
from torchvision.utils import save_image
from tqdm import tqdm
# device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameter
learning_rate = 0.001
num_epochs = 20
num_classes = 10
batch_size = 28
in_channels = 3

# load data
dataset = CatsAndDogsDataset(csv_file="custom/cats_dogs.csv", root_dir="custom/cats_dogs_resized", transform=transforms.ToTensor())



train_set, test_set = torch.utils.data.random_split(dataset, [5, 5])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

# model
model = torchvision.models.googlenet(pretrained=True)
model.to(device=device)

# criterion & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# train 
for epoch in range(num_epochs):
	losses = []
	loop =tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
	for batch_idx, (data, targets) in loop:
		data = data.to(device)
		targets = targets.to(device)

		# forward
		scores = model(data)
		loss = criterion(scores, targets)

		losses.append(loss.item())
		# backward
		optimizer.zero_grad()
		loss.backward()

		optimizer.step()
		# update progress bar
		loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
		loop.set_postfix(loss=loss.item(), acc=torch.rand(1).item())

	# print(f"Cost at epoch {epoch} is {sum(losses)/len(losses)}")

# check accuracy
def check_accuracy(loader, model):
	num_correct = 0 
	num_samples = 0
	model.eval()

	with torch.no_grad():
		for x, y in loader:
			x = x.to(device)
			y = y.to(device)
			scores = model(x)
			_, predictions = scores.max(1)
			num_correct += (predictions == y).sum()
			num_samples += predictions.size(0)
		print(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}")
	
	model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)