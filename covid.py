from scipy.sparse.construct import random
from torchvision import datasets
from torchvision.utils import make_grid
from torch.utils.data import Subset, DataLoader, random_split
from sklearn.model_selection import StratifiedShuffleSplit
import torchvision.transforms as transforms
import collections
import numpy as np
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import os

def show(img, name, norm):
	# convert tensor to numpy array
    npimg = img.numpy()
    if norm:
        npimg = 0.5146469 + 0.3062062 * npimg
	# Convert to H*W*C shape
    npimg_tr=np.transpose(npimg, (1,2,0))
    plt.imsave(name, npimg_tr)
    
def label_statistics(data_set):
    labels = data_set.targets
    counter_stat = collections.Counter(labels)
    return counter_stat

def mean(data_set):
    meanRGB = [np.mean(x.numpy(), axis=(1,2)) for x,_ in data_set]

    meanR = np.mean([m[0] for m in meanRGB])
    meanG = np.mean([m[1] for m in meanRGB])
    meanB = np.mean([m[2] for m in meanRGB])

    return [meanR, meanG, meanB]

def std(data_set):
    stdRGB = [np.std(x.numpy(), axis=(1,2)) for x,_ in data_set]

    stdR = np.mean([m[0] for m in stdRGB])
    stdG = np.mean([m[1] for m in stdRGB])
    stdB = np.mean([m[2] for m in stdRGB])

    return [stdR, stdG, stdB]

def get_data_loader(path_to_data, batch_size, result):
    # Transformers
    temp_transformer = transforms.Compose([transforms.ToTensor()])

    # Data set
    covid_ds1 = ImageFolder(path_to_data, temp_transformer)#load_dataset(path_to_data, temp_transformer)
    covid_ds2 = ImageFolder(path_to_data, temp_transformer)#load_dataset(path_to_data, temp_transformer)
    print(label_statistics(covid_ds1))

    # index of list
    indices = list(range(len(covid_ds1)))
    max_len = 4000
    train_size = int(max_len*0.7)
    test_size = max_len - train_size
    sss = StratifiedShuffleSplit(n_splits=1, train_size=train_size, test_size=test_size, random_state=0)

    for train_index, test_index in sss.split(indices, covid_ds1.targets):
        train_ds = Subset(covid_ds1, train_index)
        test_ds = Subset(covid_ds2, test_index)
    
    # Sample images
    sample_size = 4

    train_sample = [train_ds[i][0] for i in range(sample_size)]
    test_sample = [test_ds[i][0] for i in range(sample_size)]

    train_sample = make_grid(train_sample, nrow=8, padding=1)
    test_sample = make_grid(test_sample, nrow=8, padding=1)
    
    show(train_sample, os.path.join(result, "train_sample.png"),False)
    show(train_sample, os.path.join(result, "norm_train_sample.png"),True)
    
    show(test_sample, os.path.join(result, "test_sample.png"), False)
    show(test_sample, os.path.join(result, "norm_test_sample.png"),True)
    
    # Transformers
    norm = transforms.Normalize([0.5146469, 0.51464266, 0.51463896], std=[0.3062062, 0.30619, 0.3061753])
    train_transformer = transforms.Compose([
        transforms.Resize([512,512]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        norm
    ])

    test_transformer = transforms.Compose([
        transforms.Resize([512,512]),
        transforms.ToTensor(),
        norm
    ])

    # Change transformer
    covid_ds1.transform = train_transformer
    covid_ds2.transform = test_transformer

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_dl, test_dl

