import argparse
import torch
import datetime
import json
import yaml
import os

# TEST

# from Data_Preparation.data_preparation import Data_Preparation
# from Data_Preparation.data_preparation_mrs import Data_Preparation

from main_model import DDPM
from denoising_model_small import ConditionalModel
from utils import train, evaluate

from torch.utils.data import DataLoader, Subset, ConcatDataset, TensorDataset

from sklearn.model_selection import train_test_split
data_path = "C:/Users/matth/Documents/Columbia/SAIL/data/"

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="DDPM for ECG")
    parser.add_argument("--config", type=str, default="base.yaml")
    parser.add_argument('--device', default='cuda:0', help='Device')
    parser.add_argument('--name', default='test_0', help='name of model')
    parser.add_argument('--n_type', type=int, default=1, help='noise version')
    args = parser.parse_args()
    print(args)
    
    path = "config/" + args.config
    with open(path, "r") as f:
        config = yaml.safe_load(f)
        
    # foldername = "./check_points/noise_type_" + str(args.n_type) + "/"
    foldername = args.name
    print('folder:', foldername)
    os.makedirs(foldername, exist_ok=True)
    
    """Replaced by Matthieu"""
    # [X_train, y_train, X_test, y_test] = Data_Preparation(args.n_type)
    #
    # X_train = torch.FloatTensor(X_train)
    # X_train = X_train.permute(0,2,1)
    #
    # y_train = torch.FloatTensor(y_train)
    # y_train = y_train.permute(0,2,1)
    #
    # X_test = torch.FloatTensor(X_test)
    # X_test = X_test.permute(0,2,1)
    #
    # y_test = torch.FloatTensor(y_test)
    # y_test = y_test.permute(0,2,1)
    #
    # train_val_set = TensorDataset(y_train, X_train)
    # test_set = TensorDataset(y_test, X_test)
    #
    # train_idx, val_idx = train_test_split(list(range(len(train_val_set))), test_size=0.3)
    # train_set = Subset(train_val_set, train_idx)
    # val_set = Subset(train_val_set, val_idx)
    #
    # print("# of train samples: ", len(train_idx))
    # print("# of validation samples: ", len(val_idx))

    """New version"""
    # train_set, val_set, test_set = Data_Preparation()
    train_set = torch.load(os.path.join(data_path, "train_set.pt"))
    val_set = torch.load(os.path.join(data_path, "val_set.pt"))
    test_set = torch.load(os.path.join(data_path, "test_set.pt"))

    # train_set = torch.load(os.path.join(data_path, "train_set_real.pt"))
    # val_set = torch.load(os.path.join(data_path, "val_set_real.pt"))
    # test_set = torch.load(os.path.join(data_path, "test_set_real.pt"))

    train_loader = DataLoader(train_set, batch_size=config['train']['batch_size'],
                              shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=config['train']['batch_size'], drop_last=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=50, num_workers=0)
    
    # base_model = ConditionalModel(64,8,4).to(args.device)
    print("Initializing base model: ")
    base_model = ConditionalModel(config['train']['feats'], config['train']['nchannels']).to(args.device)
    print("Initializing DDPM: ")
    model = DDPM(base_model, config, args.device)

    print("starting Training: ")
    train(model, config['train'], train_loader, args.device, 
          valid_loader=val_loader, valid_epoch_interval=1, foldername=foldername)
    
    # eval final
    print('eval final')
    evaluate(model, val_loader, 1, args.device, foldername=foldername)
    
    # eval best
    print('eval best')
    foldername = "./check_points/noise_type_" + str(1) + "/"
    output_path = foldername + "/model.pth"
    model.load_state_dict(torch.load(output_path))
    evaluate(model, val_loader, 1, args.device, foldername=foldername)
    
    # don't use before final model is determined
    print('eval test')
    evaluate(model, test_loader, 1, args.device, foldername=foldername)