from dataset import DogsCats
# from models.unet import UNet
from engine import trainer
from utils import plot_results
from model import get_model

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision


from torchvision.transforms import v2
torchvision.disable_beta_transforms_warning()

import matplotlib.pyplot as plt
import numpy as np
import random

from utils import EarlyStopper

import yaml
import json
import time
import os
import wandb

config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)


run = wandb.init(entity='biomed', project='cv_assignment4', config=config)


LEARNING_RATE = float(config["LEARNING_RATE"])
LEARNING_SCHEDULER = config["LEARNING_SCHEDULER"]
BATCH_SIZE = int(config["BATCH_SIZE"])
NUM_EPOCHS = int(config["NUM_EPOCHS"])
LINEAR_PROBING = config["LINEAR_PROBING"]
PROBING_EPOCHS = int(config["PROBING_EPOCHS"])
PATIENCE = int(config["PATIENCE"])

LOSS = config["LOSS"]

IMAGE_SIZE = int(config["IMAGE_SIZE"])
MODEL = config["MODEL"]
PRETRAINED = config["PRETRAINED"]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {DEVICE} device")


def START_seed():
    seed = 9
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed) 


def main():
    START_seed()

    #run id is date and time of the run
    run_id = time.strftime("%Y-%m-%d_%H-%M-%S")

    #create folder for this run in runs folder
    os.mkdir("./runs/" + run_id)

    save_dir = "./runs/" + run_id
    

    #load data
    train_transform = v2.Compose([
            v2.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
            v2.ToImage(),
            v2.ToDtype(torch.float),
            # v2.ElasticTransform(alpha=60),
            v2.RandomRotation(degrees=(-70, 70)),
            v2.RandomAffine(degrees=(-15, 15), translate=(0.25, 0.25), scale=(0.7, 1.3), shear=(-15, 15, -15, 15)),
            v2.RandomPerspective(distortion_scale=0.2, p=0.2),
            v2.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), scale=(0.7, 1.4), antialias=True),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            v2.RandomAutocontrast(p=0.2),
            v2.RandomEqualize(p=0.2),
        ])

    val_transform = transforms.Compose([
        v2.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
        v2.ToImage(),
        v2.ToDtype(torch.float),
        v2.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
    ])

    train_dataset = DogsCats(root_dir='dataset/DogsCats/train/',
                                        transform=train_transform)

    test_dataset = DogsCats(root_dir='dataset/DogsCats/val/', 
                                          transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    #load model
    model = get_model(MODEL, PRETRAINED)


    model.to(DEVICE)
    torch.compile(model)
    
    #load optimizer
    if LOSS == "MSE":
        loss = torch.nn.MSELoss()
    elif LOSS == "L1Loss":
        loss = torch.nn.L1Loss()
    elif LOSS == "SmoothL1Loss":
        loss = torch.nn.SmoothL1Loss()
    elif LOSS == "CrossEntropyLoss":
        loss = torch.nn.CrossEntropyLoss()
    elif LOSS == "BCEWithLogitsLoss":
        loss = torch.nn.BCEWithLogitsLoss()
        
    else:
        raise Exception("Loss not implemented")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    if LEARNING_SCHEDULER == "CosineAnnealingLR":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    elif LEARNING_SCHEDULER == "ReduceLROnPlateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    else:
        lr_scheduler = None

    early_stopper = EarlyStopper(patience=PATIENCE, min_delta=0.001)

    if LINEAR_PROBING:
        linear_probing_epochs = PROBING_EPOCHS
    else:
        linear_probing_epochs = None
     
    #train model
    results = trainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        loss_fn=loss,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=DEVICE,
        epochs=NUM_EPOCHS,
        save_dir=save_dir,
        early_stopper=early_stopper,
    )

    train_summary = {
        "config": config,
        "results": results,
    }

    with open(save_dir + "/train_summary.json", "w") as f:
        json.dump(train_summary, f, indent=4)

    plot_results(results, save_dir)

if __name__ == "__main__":
    main()


