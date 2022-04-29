import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from glob import glob
from PIL import Image
import pandas as pd

class PetDataSet(Dataset):

    def __init__(self, train):
        self.train = train
        _totalTrainCsv = np.loadtxt("/home/yunjihwan/바탕화면/Ahagi/Kaggle/petfinder-pawpularity-score/train.csv", delimiter=",", dtype = np.float32, skiprows=1, usecols=range(1,14))
        #self.datalen = _totalTrainCsv.shape[0]
        self.id_es = pd.read_csv("/home/yunjihwan/바탕화면/Ahagi/Kaggle/petfinder-pawpularity-score/train.csv")


        self.features = torch.from_numpy(_totalTrainCsv[:, 0:-1])
        self.y = torch.from_numpy(_totalTrainCsv[:, [-1]])

        self.imglist = glob("/home/yunjihwan/바탕화면/Ahagi/Kaggle/petfinder-pawpularity-score/train/*.jpg")

        if(train == True):
            self.id = self.id_es[0:8992]
            self.datalen = len(self.id)
        else:
            self.id = self.id_es[8992:]
            self.datalen = len(self.id)
        #loc = self.id.Id[:5000]
        #print(loc)

    def __len__(self):
        return self.datalen

    def __getitem__(self, item):
        x = item
        if (self.train == True):
            localId = self.id.Id[item]

        else:
            localId = self.id.Id[item + 8992]
            x += 8992
        img = Image.open("/home/yunjihwan/바탕화면/Ahagi/Kaggle/petfinder-pawpularity-score/train/" + localId + ".jpg")
        img = transforms.Resize((320, 320))(img)
        # PIL.ImageShow.show(img)
        image = transforms.ToTensor()(img)
        image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(image)


        #imag = np.array(img)
       # image = torch.from_numpy(imag)

        return image, self.y[x], self.features[x]



if __name__ == "__main__":
    da =PetDataSet(train=False)
    print(len(da.id))
    ta = da[100]
    ga, na, ra = ta
    print(ga.shape)
    print(na)
    print(ra[-2])
    print(len(da))
    #hha = ga.numpy()
    #im = Image.fromarray(hha)
    #im.show()