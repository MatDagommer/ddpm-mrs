import os
import numpy as np
import torch
from tqdm import tqdm
from scipy.io import savemat, loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Subset, ConcatDataset, TensorDataset

data_path = "C:/Users/matth/Documents/Columbia/SAIL/data/"

def Data_Preparation():

    np.random.seed(1234)

    print("Loading raw data...")
    #data_path = "C:/Users/matth/Documents/Columbia/SAIL/data/"
    PH_invivoData = loadmat(os.path.join(data_path, "PH_InVivoData.mat"))

    FidsOFF = PH_invivoData['OFFdata']
    # FidsOFF = np.abs(FidsOFF)

    FidsOFF_real = np.expand_dims(np.real(FidsOFF), axis=-1)
    FidsOFF_img = np.expand_dims(np.imag(FidsOFF), axis=-1)
    FidsOFF_amp = np.expand_dims(np.abs(FidsOFF), axis=-1)

    repeat_ = np.expand_dims(np.repeat(FidsOFF_amp.max(axis=2), \
                                       FidsOFF_amp.shape[2], axis=2), axis=-1)
    # FidsOFF_real = np.divide(FidsOFF_real, FidsOFF_amp.max(axis=2))
    # FidsOFF_img = np.divide(FidsOFF_img, FidsOFF_amp.max(axis=

    # Normalizing by dividing by max amplitude
    FidsOFF_real = np.divide(FidsOFF_real, repeat_)
    FidsOFF_img = np.divide(FidsOFF_img, repeat_)

    FidsOFF = np.concatenate((FidsOFF_real, FidsOFF_img), axis=-1)
    print("Check shape of FidsOFF: ")
    print("FidsOFF shape: ", FidsOFF.shape)
    # 2048 * 160 * 101 (duration * N_reps * N_subjects)


    patch_size = FidsOFF.shape[0]
    N_reps = FidsOFF.shape[1]
    N_subjects = FidsOFF.shape[2]
    N_samples_per_subject = 100
    N_channel = FidsOFF.shape[-1]


    FidsOFF_mean = np.mean(FidsOFF, axis=1)#.transpose()
    print("Mean shape: ", FidsOFF_mean.shape)
    FidsOFF_gt = np.zeros((N_subjects, N_samples_per_subject, patch_size, N_channel))
    FidsOFF_noisy = np.zeros((N_subjects, N_samples_per_subject, patch_size, N_channel))

    print("GT shape: ", FidsOFF_gt.shape)
    print("Noisy shape: ", FidsOFF_noisy.shape)

    print("Starting dataset generation...")
    for i in tqdm(range(N_subjects)):
        for j in range(N_samples_per_subject):
            n_averages = np.random.randint(1, N_reps-1)
            sample_idx = np.arange(0, N_reps)
            np.random.shuffle(sample_idx)
            sample_idx = sample_idx[:n_averages]
            FidsOFF_gt[i, j] = FidsOFF_mean[:, i]
            FidsOFF_noisy[i, j] = np.mean(FidsOFF[:, sample_idx, i], axis=1)

    train_idx, val_idx = train_test_split(range(N_subjects), test_size=0.4)
    val_idx, test_idx = train_test_split(val_idx, test_size=0.5)

    X_train = FidsOFF_noisy[train_idx].reshape(-1, patch_size, N_channel)
    X_val = FidsOFF_noisy[val_idx].reshape(-1, patch_size, N_channel)
    X_test = FidsOFF_noisy[test_idx].reshape(-1, patch_size, N_channel)

    y_train = FidsOFF_gt[train_idx].reshape(-1, patch_size, N_channel)
    y_val = FidsOFF_gt[val_idx].reshape(-1, patch_size, N_channel)
    y_test = FidsOFF_gt[test_idx].reshape(-1, patch_size, N_channel)

    # Normalizing the dataset
    # print("Normalizing dataset...")
    # scaler = MinMaxScaler(feature_range=(-1, 1))
    # y_train = scaler.fit_transform(y_train)
    # X_train = scaler.transform(X_train)
    # X_val = scaler.transform(X_val)
    # X_test = scaler.transform(X_test)
    # y_val = scaler.transform(y_val)
    # y_test = scaler.transform(y_test)

    print("Converting data to tensors...")
    #X_train = np.expand_dims(X_train, axis = 1)
    X_train = torch.FloatTensor(X_train)
    X_train = X_train.permute(0, 2, 1)
    #y_train = np.expand_dims(y_train, axis=1)
    y_train = torch.FloatTensor(y_train)
    y_train = y_train.permute(0, 2, 1)

    #X_val = np.expand_dims(X_val, axis=1)
    X_val = torch.FloatTensor(X_val)
    X_val = X_val.permute(0, 2, 1)
    #y_val = np.expand_dims(y_val, axis=1)
    y_val = torch.FloatTensor(y_val)
    y_val = y_val.permute(0, 2, 1)

    #X_test = np.expand_dims(X_test, axis=1)
    X_test = torch.FloatTensor(X_test)
    X_test = X_test.permute(0, 2, 1)
    #y_test = np.expand_dims(y_test, axis=1)
    y_test = torch.FloatTensor(y_test)
    y_test = y_test.permute(0, 2, 1)

    train_set = TensorDataset(y_train, X_train)
    val_set = TensorDataset(y_val, X_val)
    test_set = TensorDataset(y_test, X_test)

    print("Dataset ready.")

    #return train_set, val_set, test_set

    torch.save(train_set, os.path.join(data_path, "train_set.pt"))
    torch.save(val_set, os.path.join(data_path, "val_set.pt"))
    torch.save(test_set, os.path.join(data_path, "test_set.pt"))

    return train_set, val_set, test_set
