from pyspark import SparkContext
sc =SparkContext.getOrCreate()
sc.install_pypi_package("sparkdl")
sc.install_pypi_package("torch")
sc.install_pypi_package("torchvision")
sc.install_pypi_package("PyArrow")

from pyspark.ml.image import ImageSchema
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
from torchvision.datasets.folder import default_loader  # private API
import os
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import ArrayType, FloatType
from functools import reduce
from collections import namedtuple
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

use_cuda = torch.cuda.is_available()
Params = namedtuple('Params', ['batch_size', 'test_batch_size', 'epochs', 'lr', 'momentum', 'seed', 'cuda', 'log_interval'])
args = Params(batch_size=64, test_batch_size=64, epochs=1, lr=0.001, momentum=0.5, seed=1, cuda=use_cuda, log_interval=20)
torch.manual_seed(args.seed)

brain_mri_path = 's3://braintumorproject/Healthcare_AI_Datasets/Brain_MRI/'

class ImageDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = [brain_mri_path+a[0].replace('.tif','.png') for a in paths]
        image1 = spark.read.format("image").option("dropInvalid", False).load(self.paths)
        images = image1.collect()
        Xt = []
        for i in range(len(images)):
            w = images[i][0]
            image2 = ImageSchema.toNDArray(w)
            Xt.append((w[0].split('/')[-1],image2))
        self.images = sorted(Xt, key=lambda tup: tup[0])
        yt = [(a[0].split('/')[-1],a[1]) for a in paths]
        self.labels = sorted(yt, key=lambda tup: tup[0])


        self.transform = transform
    def __len__(self):
        return  len(self.paths)
    def __getitem__(self, index):
        #if npg exist, skip the following steps
        #im = Image.open('Healthcare_AI_Datasets/Brain_MRI/'+self.paths[index][0])
        #rgb_im = im.convert('RGB')

        #im.save('Healthcare_AI_Datasets/Brain_MRI/'+self.paths[index][0].replace(".tif", ".png"), quality=95)
        
        
        #data is already loaded
        image1 = self.images[index][1]

        #read one by one
        #image1 = spark.read.format("image").option("dropInvalid", False).load('Healthcare_AI_Datasets/Brain_MRI/'+ self.paths[index][0].replace(".tif", ".png")).collect()
        #image1 = image1[0][0]
        
        #image1 = ImageSchema.readImages('Healthcare_AI_Datasets/Brain_MRI/'+ self.paths[index][0].replace(".tif", ".png")).collect()
        #image2 = ImageSchema.toNDArray(image1)
        #image = np.transpose(image2, (2, 1, 0))
        if self.transform is not None:
            image3 = Image.fromarray(image1.astype('uint8'), 'RGB')
            image = self.transform(image3)
        return image, self.labels[index][1]

data_mask_path = 's3://braintumorproject/Healthcare_AI_Datasets/Brain_MRI/data_mask.csv'

spark = SparkSession.builder.appName("how to read csv file").config("spark.driver.memory", "15g").getOrCreate()
df = spark.read.option("header","True").csv(data_mask_path)
df = df.select("image_path","mask")
list_files = df.collect()
list_files = [(list_files[i].image_path, int(list_files[i].mask)) for i in range(len(list_files))]

from random import shuffle
shuffle(list_files)
Train_size = int(len(list_files)* 0.85)
list_files_train = list_files[:Train_size]
list_files_test = list_files[Train_size:]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self.bnf = nn.BatchNorm1d(2048)
        self.fc0 = nn.Linear(2048, 512)
        self.fc1 = nn.Linear(512,256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        #x = self.bnf(x)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        #x = self.fc2(x)
        x= self.fc2(x)
        return x
    

model = Net()
model.share_memory()

model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features

model.fc = nn.Sequential(nn.Linear(num_ftrs, 512),
                                nn.ReLU(),
                                nn.Dropout(p=0.3),
                                nn.Linear(512, 256),
                                nn.ReLU(),
                                nn.Dropout(p=0.3),
                                nn.Linear(256, 2))

for p in model.parameters():
        p.requires_grad = True


cross_loss = nn.CrossEntropyLoss()

train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(224),
        #transforms.RandomRotation(degrees=(-90, 90)),
        transforms.RandomHorizontalFlip(),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        transforms.Normalize(mean=[0.08, 0.08, 0.08],std=[0.12, 0.12, 0.12]) 
])

test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(224),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        transforms.Normalize(mean=[0.08, 0.08, 0.08],std=[0.12, 0.12, 0.12])               
  ])

train_loader = torch.utils.data.DataLoader(ImageDataset(list_files_train,train_transform),
        batch_size=args.batch_size, shuffle=True, num_workers=0)

test_loader = torch.utils.data.DataLoader( ImageDataset(list_files_test,test_transform),
        batch_size=args.test_batch_size, shuffle=False, num_workers=0)

def train_epoch(epoch, args, model, data_loader, optimizer):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()      
        #for _ in range(1):
        optimizer.zero_grad()
        output = model(data)
        loss = cross_loss(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        #print(loss.item())
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), train_loss/(batch_idx+1)))

def test_epoch(model, data_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()      
            output = model(data)
            test_loss += cross_loss(output, target).data.item() # sum up batch loss
            pred = output.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum().item()

        #test_loss /= len(data_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset),
            100. * correct / len(data_loader.dataset)))

if args.cuda:
    model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
for epoch in range(1, args.epochs + 1):
    
    train_epoch(epoch, args, model, train_loader, optimizer)
    test_epoch(model, test_loader)

