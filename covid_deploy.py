import torch
from covid import mean, std, label_statistics
from covid_model import load_model, loss_epoch
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
        tmp = pred.eq(yb.view_as(pred)).sum().item()
        metric += tmp
        print("batch result: %d/%d %f" % (tmp, len(xb), tmp / len(xb) * 100)) 
    return metric / len_data

if __name__ == "__main__":
    path_to_weights = "./covid_models2/resnet18.pt"
    
    #norm = transforms.Normalize(mean(ds), std(ds))
    val_transformer = transforms.Compose([
        transforms.Resize([512,512]),
        transforms.ToTensor(),
#        norm
    ])
    ds = ImageFolder("./full_data", val_transformer)
    print(label_statistics(ds))
    ds.transform = val_transformer

    test_dl = DataLoader(ds, batch_size=64, shuffle=True)
    
    device = torch.device("cuda:0")
    model = load_model(False, device)
    model.load_state_dict(torch.load(path_to_weights))

    model.eval()
    with torch.no_grad():
        acc = test(model, test_dl)
    print("test acc: %.2f" %(acc*100))

