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
class NET(nn.Module):
	def __init__(self):
		super(NET, self).__init__()
		self.conv0 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
		self.conv1_0 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
		self.conv1_1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
		self.conv16_1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
		self.conv16_2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
		self.conv2_0 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
		self.conv2_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
		self.conv32_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
		self.conv32_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
		self.conv3_0 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
		self.conv3_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
		self.conv64_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
		self.conv64_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
		self.conv4_0 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
		self.conv4_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
		self.conv128_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
		self.conv128_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
		self.conv5_0 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
		self.conv5_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
		self.conv256_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
		self.conv256_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
		self.conv256 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)


		self.shortcutcon0 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
		self.shortcutcon1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
		self.shortcutcon2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
		self.shortcutcon3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
		self.shortcutcon4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
		self.shortcutcon16 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
		self.shortcutcon32 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
		self.shortcutcon64 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
		self.shortcutcon128 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
		self.shortcutcon256 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.pool2 = nn.MaxPool2d(kernel_size=4, stride=4)
		self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

		#self.fc1 = nn.Linear(256 * 10 * 10, 1024)
		self.fc2 = nn.Linear(320, 1)

		self.ffc1 = nn.Linear(12, 512)
		self.ffc2 = nn.Linear(512, 64)


		self.dropout_prob = 0.5
		self.batch_norm1_1 = nn.BatchNorm1d(1024)
		self.batch_norm1_2 = nn.BatchNorm1d(512)
		self.batch_norm1_3 = nn.BatchNorm1d(64)
		self.batch_norm1_4 = nn.BatchNorm1d(100)
		self.batch_norm0_1 = nn.BatchNorm2d(8)
		self.batch_norm0_2 = nn.BatchNorm2d(16)
		self.batch_norm0_3 = nn.BatchNorm2d(32)
		self.batch_norm0_4 = nn.BatchNorm2d(64)
		self.batch_norm0_5 = nn.BatchNorm2d(128)
		self.batch_norm0_6 = nn.BatchNorm2d(256)

	def forward(self, x, feature):
		x = self.conv0(x)
		x = self.batch_norm0_1(x)
		x = F.relu(x)  # 3->8
		x = F.dropout2d(x, training=self.training, p=self.dropout_prob)

		shortcut0 = self.shortcutcon0(x)  # 8>>16
		x = self.conv1_0(x)  # 8 -> 16
		x = self.batch_norm0_2(x)
		x = F.relu(x)
		x = F.dropout2d(x, training=self.training, p=self.dropout_prob)
		x = self.conv1_1(x)  # 16->16
		x = self.batch_norm0_2(x)
		x += self.batch_norm0_2(shortcut0)
		x = F.relu(x)
		x = F.dropout2d(x, training=self.training, p=self.dropout_prob)
		x = self.pool1(x)

		shortcut16 = self.shortcutcon16(x)
		x = self.conv16_1(x)
		x = self.batch_norm0_2(x)
		x = F.relu(x)
		x = F.dropout2d(x, training=self.training, p=self.dropout_prob)
		x = self.conv16_2(x)
		x = self.batch_norm0_2(x)
		x += self.batch_norm0_2(shortcut16)
		x = F.relu(x)
		x = F.dropout2d(x, training=self.training, p=self.dropout_prob)

		shortcut1 = self.shortcutcon1(x)  # 16 - >32
		x = self.conv2_0(x)  # 16->32
		x = self.batch_norm0_3(x)
		x = F.relu(x)
		x = F.dropout2d(x, training=self.training, p=self.dropout_prob)
		x = self.conv2_1(x)  # 32->32
		x = self.batch_norm0_3(x)
		x += self.batch_norm0_3(shortcut1)
		x = F.relu(x)
		x = F.dropout2d(x, training=self.training, p=self.dropout_prob)
		x = self.pool1(x)

		shortcut32 = self.shortcutcon32(x)
		x = self.conv32_1(x)
		x = self.batch_norm0_3(x)
		x = F.relu(x)
		x = F.dropout2d(x, training=self.training, p=self.dropout_prob)
		x = self.conv32_2(x)
		x = self.batch_norm0_3(x)
		x += self.batch_norm0_3(shortcut32)
		x = F.relu(x)
		x = F.dropout2d(x, training=self.training, p=self.dropout_prob)

		shortcut2 = self.shortcutcon2(x)  # 32->64
		x = self.conv3_0(x)  # 32 ->64
		x = self.batch_norm0_4(x)
		x = F.relu(x)
		x = F.dropout2d(x, training=self.training, p=self.dropout_prob)
		x = self.conv3_1(x)  # 64->64
		x = self.batch_norm0_4(x)
		x += self.batch_norm0_4(shortcut2)
		x = F.relu(x)
		x = F.dropout2d(x, training=self.training, p=self.dropout_prob)
		x = self.pool1(x)

		shortcut64 = self.shortcutcon64(x)
		x = self.conv64_1(x)
		x = self.batch_norm0_4(x)
		x = F.relu(x)
		x = F.dropout2d(x, training=self.training, p=self.dropout_prob)
		x = self.conv64_2(x)
		x = self.batch_norm0_4(x)
		x += self.batch_norm0_4(shortcut64)
		x = F.relu(x)
		x = F.dropout2d(x, training=self.training, p=self.dropout_prob)

		shortcut3 = self.shortcutcon3(x)  # 64->128
		x = self.conv4_0(x)  # 64 ->128
		x = self.batch_norm0_5(x)
		x = F.relu(x)
		x = F.dropout2d(x, training=self.training, p=self.dropout_prob)
		x = self.conv4_1(x)  # 128->128
		x = self.batch_norm0_5(x)
		x += self.batch_norm0_5(shortcut3)
		x = F.relu(x)
		x = F.dropout2d(x, training=self.training, p=self.dropout_prob)
		x = self.pool1(x)

		shortcut128 = self.shortcutcon128(x)
		x = self.conv128_1(x)
		x = self.batch_norm0_5(x)
		x = F.relu(x)
		x = F.dropout2d(x, training=self.training, p=self.dropout_prob)
		x = self.conv128_2(x)
		x = self.batch_norm0_5(x)
		x += self.batch_norm0_5(shortcut128)
		x = F.relu(x)
		x = F.dropout2d(x, training=self.training, p=self.dropout_prob)

		shortcut4 = self.shortcutcon4(x)  # 128 - >256
		x = self.conv5_0(x)  # 128->256
		x = self.batch_norm0_6(x)
		x = F.relu(x)
		x = F.dropout2d(x, training=self.training, p=self.dropout_prob)
		x = self.conv5_1(x)  # 256 -> 256
		x = self.batch_norm0_6(x)
		x += self.batch_norm0_6(shortcut4)
		x = F.relu(x)
		x = F.dropout2d(x, training=self.training, p=self.dropout_prob)
		x = self.pool1(x)

		shortcut256 = self.shortcutcon256(x)
		x = self.conv256_1(x)
		x = self.batch_norm0_6(x)
		x = F.relu(x)
		x = F.dropout2d(x, training=self.training, p=self.dropout_prob)
		x = self.conv256_2(x)
		x = self.batch_norm0_6(x)
		x += self.batch_norm0_6(shortcut256)
		x = F.relu(x)
		x = F.dropout2d(x, training=self.training, p=self.dropout_prob)

		######################################################
		x = self.conv256(x)
		x = F.avg_pool2d(x, x.size()[2:])
		x= torch.squeeze(x)
		fea = self.ffc1(feature)
		fea = self.batch_norm1_2(fea)
		fea = F.relu(fea)
		fea = F.dropout(fea, training=self.training, p=self.dropout_prob)
		fea = self.ffc2(fea)
		fea = self.batch_norm1_3(fea)
		fea = F.relu(fea)
		fea = F.dropout(fea, training=self.training, p=self.dropout_prob)


		x= torch.cat((x,fea), dim=1)

		x =self.fc2(x)

		#x = torch.sigmoid(x)
		return x
#model = NET().cuda()

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