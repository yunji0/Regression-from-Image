import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from CustomDataSet import PetDataSet
import wandb


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "8"

dataset = PetDataSet(train=True) #image , yvalue, features
dataloader = DataLoader(dataset, batch_size=32 , shuffle= True, drop_last= True)

test_dataset = PetDataSet(train=False)
test_dataloader = DataLoader(test_dataset, batch_size=32 , shuffle= False, drop_last= True)
###############################################################
#if i use resizing image, Cute dog and cat are no longer cute
# How to use GPU
#
###############################################################


import torchvision.models as models
model = models.resnet18(pretrained=True)
pre_output = model.fc.in_features
model.fc =nn.Linear(pre_output,1)
model = model.cuda()

# model= models.resnet34(pretrained =False)
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, 10
# model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay=0.00001)
criterion = nn.MSELoss()

wandb.init(project="Kaggle3", entity="ahagi")
wandb.watch(model,criterion, log = "all", log_freq=1)

import math
def train(model, train_loader, optimizer):
	model.train()
	loss_temp = 0.0
	for batch_idx,(image, y, feature) in enumerate(train_loader):
		#print("{}Epoch {} Iteration START!!!".format(i, batch_idx))
		image = image.cuda()
		y = y.cuda()
		feature = feature.cuda()

		optimizer.zero_grad()
		output = model(image)
		loss = criterion(output, y/100)
		loss.backward()
		optimizer.step()

		fakeloss = criterion(output*100, y)
		loss_temp += fakeloss
		# print("{} Epoch {} Iteration END loss:{}!!!".format(iu,batch_idx,loss.item()))
		# if(batch_idx %100 ==0):
		print("{}/280 iteration".format(batch_idx), end="\r")
	return math.sqrt(loss_temp / 281)




def evaluate(model, dataloader):
	model.eval()
	RMSE = 0.0
	with torch.no_grad():
		for batch_idx,(image, y, feature ) in enumerate(dataloader):
			image = image.cuda()

			y = y.cuda()
			feature = feature.cuda()

			output = model(image)
			loss = criterion(output*100, y)
			RMSE += loss
			print("{}/27 iteration".format(batch_idx), end="\r")
		return math.sqrt(RMSE / 28)


for i in range(0,300):
	print("{} EPOCH START!!!".format(i))
	loss =train(model, dataloader, optimizer)
	valRMSE = evaluate(model, test_dataloader)
	print("{} EPOCH END!!!".format(i))
	wandb.log({"train_RMSE": loss , "val_RMSE": valRMSE})

	#if(math.sqrt(loss) <=5):
	#	break
#wandb.log({"EPOCH":i , "loss": loss })
torch.save(model,'/home/yunjihwan/바탕화면/Ahagi/Kaggle/MetaImageInput.pt')
print("reg")
"""
img = Image.open("/home/yunjihwan/바탕화면/Ahagi/Kaggle/petfinder-pawpularity-score/train/0a0da090aa9f0342444a7df4dc250c66.jpg")

img = transforms.Resize((640, 640))(img)


image = transforms.ToTensor()(img)
image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(image)

image = image.cuda()
model.eval()
y_estim = model(image)
print("############################")
print(y_estim)
print("############################")
"""
