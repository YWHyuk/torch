from torchvision import models
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from matplotlib import pyplot as plt
import copy, os, argparse, pathlib

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
        print("train loss: %.6f, val loss: %.6f, train accuracy: %.2f, val accuracy: %.2f" %
                (train_loss, val_loss, 100*train_metric, 100*val_metric))

    model.load_state_dict(best_model_wts)
    return model, loss_history, metric_history

def load_model(model_name, pretrained, device):
    if model_name == "resnet":
        model = models.resnet18(pretrained=pretrained)
    elif model_name == "vgg":
        model = models.vgg16(pretrained=pretrained)
    else:
        print("Undefined model name")
        exit(1)

    num_classes = 2
    num_ftrs= model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    model.to(device)
 
    return model

def draw_result(title, name, log ,num_epochs):
    plt.title("Train-Val " + title)
    plt.plot(range(1,num_epochs+1), log["train"], label="train")
    plt.plot(range(1,num_epochs+1), log["val"], label="val")
    plt.ylabel(title)
    plt.xlabel("Training Epochs")
    plt.legend()
    plt.savefig(os.path.join(output_folder, name))
    plt.clf()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prameter for ML')
    parser.add_argument('-O', type=pathlib.Path, help='Output folder path', required=True)
    parser.add_argument('-I', type=pathlib.Path, help='input image folder path', required=True)
    parser.add_argument('-M', choices=['resnet', 'vgg'], required=True)
    parser.add_argument('-P', choices=['T', 'F'], required=True)
    parser.add_argument('-E', type=int, required=True)
    parser.add_argument('-B', type=int, required=True)
    parsed = parser.parse_args()

    output_folder = str(parsed.O)
    input_data = str(parsed.I)
    model_name = parsed.M
    pretrained = parsed.P == 'T'
    num_epochs = parsed.E
    batch_size = parsed.B

    try:
        os.makedirs(output_folder, exist_ok=False)
    except FileExistsError:
        print("Overwriting output folder!")
        
    device = torch.device("cuda:0")
    # Load model and data
    model = load_model(model_name, pretrained, device)
    train_dl, val_dl = get_data_loader(input_data, batch_size, output_folder)
    
    # Set loss, optimizer, learning late scheduler
    loss_func = nn.CrossEntropyLoss(reduction="mean")
    opt = optim.Adam(model.parameters(), lr=1e-3)
    lr_scheduler = StepLR(opt, step_size=20, gamma=0.1)
    
    params_train = {
            "num_epochs" : num_epochs,
            "opt" : opt,
            "loss_func" : loss_func,
            "train_dl" : train_dl,
            "val_dl" : val_dl,
            "sanity" : False,
            "lr_scheduler" : lr_scheduler,
            "path_to_weights" : os.path.join(output_folder, "weight.pt"),
            "device" : device
            }

    model_resnt18, loss_hist, metric_hist = train_val(model, params_train)

    draw_result("Loss", "Train-val-loss.png", loss_hist, num_epochs)
    draw_result("Accuracy", "Train-val-Accuracy.png", metric_hist, num_epochs)
