from torchvision import datasets
from torch.utils.data import Subset, DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms
import os, glob
import collections
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

class covoid_dataset(Dataset):
    def __init__(self, path, transform):
        self.transform = transform
        self.files = []
        
        path_to_data = os.path.join(path, "0")
        self.files += [tuple([os.path.join(path_to_data, y), 0]) for x in os.walk(path_to_data) for y in glob.glob(os.path.join(x[0], '*.jpg'))]

        path_to_data = os.path.join(path, "1")
        self.files += [tuple([os.path.join(path_to_data, y), 1]) for x in os.walk(path_to_data) for y in glob.glob(os.path.join(x[0], '*.jpg'))]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = Image.open(self.files[idx][0])
        image = self.transform(image)
        return image, self.files[idx][1]

def load_dataset(path, mode, transformer):
    if not os.path.exists(path):
        os.mkdir(path)
    ds = datasets.STL10(path, split=mode, download=True, transform=transformer)
    print(ds.data.shape)
    label_statistics(ds)
    return ds

def get_labels(data_set):
    return [y for _,y in data_set]

def label_statistics(data_set):
    labels = get_labels(data_set)
    counter_stat = collections.Counter(labels)
    print(counter_stat)
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
    train_ds = load_dataset(path_to_data, "train", temp_transformer)
    test0_ds = load_dataset(path_to_data, "test", temp_transformer)

    # Transformers
    train_transformer = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        #transforms.Normalize(mean(train_ds), std(train_ds))
    ])

    test_transformer = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(mean(train_ds), std(train_ds))
    ])

    # Change transformer
    train_ds.transform = train_transformer
    test0_ds.trainform = test_transformer

    y_test0 = get_labels(test0_ds)

    # split validation and test
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)

    # index of list
    indices = list(range(len(test0_ds)))

    for test_index, val_index in sss.split(indices, y_test0):
        val_ds = Subset(test0_ds, val_index)
        test_ds = Subset(test0_ds, test_index)
        
        val_ds.transform = test_transformer
        label_statistics(val_ds)
        label_statistics(test_ds)

        train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=64, shuffle=False)
        break

    return train_dl, val_dl

