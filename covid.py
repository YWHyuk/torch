from torchvision import datasets
from torch.utils.data import Subset, DataLoader, random_split
from sklearn.model_selection import StratifiedShuffleSplit
import torchvision.transforms as transforms
import collections
import numpy as np
from torchvision.datasets import ImageFolder

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

def get_data_loader(path_to_data):
    # Transformers
    temp_transformer = transforms.Compose([transforms.ToTensor()])

    # Data set
    covid_ds1 = ImageFolder(path_to_data, temp_transformer)#load_dataset(path_to_data, temp_transformer)
    covid_ds2 = ImageFolder(path_to_data, temp_transformer)#load_dataset(path_to_data, temp_transformer)
    print(label_statistics(covid_ds1))

    #norm = transforms.Normalize(mean(covid_ds), std(covid_ds))

    # Transformers
    train_transformer = transforms.Compose([
        transforms.Resize([512,512]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        #norm
    ])

    val_transformer = transforms.Compose([
        transforms.Resize([512,512]),
        transforms.ToTensor(),
        #norm
    ])
    # Change transformer
    covid_ds1.transform = train_transformer
    covid_ds2.transform = val_transformer

    # index of list
    indices = list(range(len(covid_ds1)))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)

    for train_index, val_index in sss.split(indices, covid_ds1.targets):
        train_ds = Subset(covid_ds1, train_index)
        val_ds = Subset(covid_ds2, val_index)

    #print(label_statistics(train_ds))
    #print(label_statistics(val_ds))

    train_dl = DataLoader(train_ds, batch_size=64*2, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=64*2, shuffle=False)

    return train_dl, val_dl

