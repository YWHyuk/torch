from torchvision import models
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from matplotlib import pyplot as plt
import copy, os

from covid import get_data_loader

def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

    current_lr = get_lr(opt)
    print('current lr=%d' % (current_lr))

def metrics_batch(output, target):
    pred = output.argmax(dim=1, keepdim=True)
    corrects = pred.eq(target.view_as(pred)).sum().item()
    return corrects

def loss_batch(loss_func, output, target, opt=None):
    loss = loss_func(output, target)
    metric_b = metrics_batch(output, target)
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item(), metric_b

def loss_epoch(model, device, loss_func, data_loader, sanity=False, opt=None):
    running_loss = 0.0
    running_metric = 0.0
    len_data = len(data_loader.dataset)

    for xb, yb in data_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        output = model(xb)
        loss_b, metric_b = loss_batch(loss_func, output, yb, opt)
        running_loss += loss_b

        if metric_b is not None:
            running_metric += metric_b
        if sanity is True:
            break

    loss = running_loss / float(len_data)
    metric = running_metric / float(len_data)
    return loss, metric

def train_val(model, params):
    num_epochs = params["num_epochs"]
    loss_func = params["loss_func"]
    opt = params["opt"]
    train_dl = params["train_dl"]
    val_dl = params["val_dl"]
    sanity = params["sanity"]
    lr_scheduler = params["lr_scheduler"]
    path_to_weights = params["path_to_weights"]
    device = params["device"]

    loss_history = {
            "train": [],
            "val": [],
    }
    metric_history = {
            "train": [],
            "val": [],
    }

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")

    for epoch in range(num_epochs):
        current_lr = get_lr(opt)
        print("Epoch %d/%d, current lr = %f" % (epoch, num_epochs - 1, current_lr))

        model.train()
        train_loss, train_metric = loss_epoch(model, device, loss_func, train_dl, sanity, opt)

        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)

        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model, device, loss_func, val_dl, sanity)

        loss_history["val"].append(val_loss)
        metric_history["val"].append(val_metric)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), path_to_weights)
            print("Copied best model weights!")
 
        lr_scheduler.step()
        print("train loss: %.6f, dev loss: %.6f, train accuracy: %.2f, dev accuracy: %.2f" %
                (train_loss, val_loss, 100*train_metric, 100*val_metric))

    model.load_state_dict(best_model_wts)
    return model, loss_history, metric_history

def tune_resnet(pretrained, device):
    model_resnet18 = models.resnet18(pretrained=pretrained)

    num_classes = 2
    num_ftrs= model_resnet18.fc.in_features
    model_resnet18.fc = nn.Linear(num_ftrs, num_classes)

    model_resnet18.to(device)
 
    return model_resnet18

if __name__ == "__main__":
    os.makedirs("./covid_models", exist_ok=True)

    device = torch.device("cuda:0")
    model_resnet18 = tune_resnet(False, device)
    loss_func = nn.CrossEntropyLoss(reduction="mean")
    opt = optim.Adam(model_resnet18.parameters(), lr=1e-3)
    lr_scheduler = StepLR(opt, step_size=20, gamma=0.1)
    train_dl, val_dl = get_data_loader("./full_data")
    num_epochs = 99

    params_train = {
            "num_epochs" : num_epochs,
            "opt" : opt,
            "loss_func" : loss_func,
            "train_dl" : train_dl,
            "val_dl" : val_dl,
            "sanity" : False,
            "lr_scheduler" : lr_scheduler,
            "path_to_weights" : "./covid_models/resnet18.pt",
            "device" : device
            }

    model_resnt18, loss_hist, metric_hist = train_val(model_resnet18, params_train)
    os.makedirs("result", exist_ok=True)
    plt.title("Train-Val Loss")
    plt.plot(range(1,num_epochs+1), loss_hist["train"], label="train")
    plt.plot(range(1,num_epochs+1), loss_hist["val"], label="val")
    plt.ylabel("Loss")
    plt.xlabel("Training Epochs")
    plt.legend()
    plt.savefig("result/norm_Train-val-loss.png")
    plt.clf()

    plt.title("Train-Val Accuracy")
    plt.plot(range(1,num_epochs+1), metric_hist["train"], label="train")
    plt.plot(range(1,num_epochs+1), metric_hist["val"], label="val")
    plt.ylabel("Accuracy")
    plt.xlabel("Training Epochs")
    plt.legend()
    plt.savefig("result/norm_Train-val-Accuracy.png")

