import torch
from covid import mean, std
from covid_model import tune_resnet, loss_epoch
from torch import nn, optim
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader, Dataset
from torchvision.datasets import ImageFolder

def test(model, data_loader):
    metric = 0.0
    len_data = len(data_loader.dataset)
    
    for xb, yb in data_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        output = model(xb)
        pred = output.argmax(dim=1, keepdim=True)
        metric += pred.eq(yb.view_as(pred)).sum().item()
    return metric / len_data

if __name__ == "__main__":
    path_to_weights = "./covid_models/resnet18.pt"
    
    temp_transformer = transforms.Compose([transforms.ToTensor()])
    ds = ImageFolder("./COVID19", temp_transformer)
    
    norm = transforms.Normalize(mean(ds), std(ds))
    val_transformer = transforms.Compose([
        transforms.ToTensor(),
        norm
    ])

    ds.transform = val_transformer
    test_dl = DataLoader(ds, batch_size=64, shuffle=False)
    
    device = torch.device("cuda:0")
    model = tune_resnet(False, device)
    model.load_state_dict(torch.load(path_to_weights))

    model.eval()
    with torch.no_grad():
        acc = test(model, test_dl)
    print("test acc: %.2f" %(acc*100))

