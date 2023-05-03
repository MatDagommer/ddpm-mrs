import argparse
import torch
import datetime
import json
import yaml
import os

# from Data_Preparation.data_preparation import Data_Preparation
from Data_Preparation.data_preparation_mrs import Data_Preparation

from main_model import DDPM
from denoising_model_small import ConditionalModel
from utils import train, evaluate

from torch.utils.data import DataLoader, Subset, ConcatDataset, TensorDataset

from sklearn.model_selection import train_test_split
# data_path = "C:/Users/matth/Documents/Columbia/SAIL/data/"


if __name__ == "__main__":
    d_folder = os.getcwd()
    print("d_folder: ", d_folder)
    parser = argparse.ArgumentParser(description="DDPM for ECG")
    parser.add_argument("--config", type=str, default="base.yaml")
    parser.add_argument('--device', default='cuda:0', help='Device')
    parser.add_argument('--name', default='test_0', help='name of model')
    parser.add_argument('--n_type', type=int, default=1, help='noise version')
    parser.add_argument('--d_folder', default=d_folder, help='data folder')
    parser.add_argument('--n_epochs', type=int, default=50, help='data folder')
    parser.add_argument('--n_channels', type=int, default=2, help='number of channels. 1: real part only. 2: imaginary part only.')
    args = parser.parse_args()
    print(args)

    data_path = os.path.join(args.d_folder, "data")

    path = "config/" + args.config
    with open(path, "r") as f:
        config = yaml.safe_load(f)
        
    # foldername = "./check_points/noise_type_" + str(args.n_type) + "/"
    foldername = os.path.join("check_points", args.name)
    print('folder:', foldername)
    os.makedirs(foldername, exist_ok=True)


    train_set, val_set, test_set = Data_Preparation(data_path, n_channels=args.n_channels)
    #
    # if args.n_channels == 2:
    #     train_set = torch.load(os.path.join(data_path, "train_set.pt"))
    #     val_set = torch.load(os.path.join(data_path, "val_set.pt"))
    #     test_set = torch.load(os.path.join(data_path, "test_set.pt"))
    # elif args.n_channels == 1:
    #     train_set = torch.load(os.path.join(data_path, "train_set_real.pt"))
    #     val_set = torch.load(os.path.join(data_path, "val_set_real.pt"))
    #     test_set = torch.load(os.path.join(data_path, "test_set_real.pt"))

    train_loader = DataLoader(train_set, batch_size=config['train']['batch_size'],
                              shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=config['train']['batch_size'], drop_last=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=50, num_workers=0)
    
    # base_model = ConditionalModel(64,8,4).to(args.device)
    print("Initializing base model: ")
    base_model = ConditionalModel(config['train']['feats'], args.n_channels).to(args.device)
    print("Initializing DDPM: ")
    model = DDPM(base_model, config, args.device)

    print("starting Training: ")
    train(model, config['train'], train_loader, args.device,
          valid_loader=val_loader, valid_epoch_interval=1, foldername=foldername,
          n_epochs=args.n_epochs)
    
    # eval final
    print('eval final')
    evaluate(model, val_loader, 1, args.device, foldername=foldername)
    
    # eval best
    print('eval best')
    # foldername = "./check_points/noise_type_" + str(1) + "/"
    #foldername = "./check_points/noise_type_1/"
    output_path = foldername + "/model.pth"
    model.load_state_dict(torch.load(output_path))
    evaluate(model, val_loader, 1, args.device, foldername=foldername)
    
    # don't use before final model is determined
    print('eval test')
    evaluate(model, test_loader, 1, args.device, foldername=foldername)