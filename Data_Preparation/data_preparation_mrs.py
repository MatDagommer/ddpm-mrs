import os
import numpy as np
import torch
from tqdm import tqdm
from scipy.io import savemat, loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Subset, ConcatDataset, TensorDataset


def Data_Preparation(data_path, n_channels=2):

    np.random.seed(1234)

    print("Loading raw data...")
    PH_invivoData = loadmat(os.path.join(data_path, "PH_InVivoData.mat"))

    FidsOFF = PH_invivoData['OFFdata']

    FidsOFF_real = np.expand_dims(np.real(FidsOFF), axis=-1)
    FidsOFF_img = np.expand_dims(np.imag(FidsOFF), axis=-1)
    FidsOFF_amp = np.expand_dims(np.abs(FidsOFF), axis=-1)

    repeat_ = np.expand_dims(np.repeat(FidsOFF_amp.max(axis=2), \
                                       FidsOFF_amp.shape[2], axis=2), axis=-1)

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


    FidsOFF_mean = np.mean(FidsOFF, axis=1) #.transpose()
    print("Mean shape: ", FidsOFF_mean.shape)
    FidsOFF_gt = np.zeros((N_subjects, N_samples_per_subject, patch_size, N_channel))
    FidsOFF_noisy = np.zeros((N_subjects, N_samples_per_subject, patch_size, N_channel))

    print("GT shape: ", FidsOFF_gt.shape)
    print("Noisy shape: ", FidsOFF_noisy.shape)

    print("Starting dataset generation...")
    for i in tqdm(range(N_subjects)):
        for j in range(N_samples_per_subject):
            n_averages = np.random.randint(1, N_reps//2)
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

    print("Converting data to tensors...")
    X_train = torch.FloatTensor(X_train)
    X_train = X_train.permute(0, 2, 1)
    y_train = torch.FloatTensor(y_train)
    y_train = y_train.permute(0, 2, 1)

    X_val = torch.FloatTensor(X_val)
    X_val = X_val.permute(0, 2, 1)
    y_val = torch.FloatTensor(y_val)
    y_val = y_val.permute(0, 2, 1)

    X_test = torch.FloatTensor(X_test)
    X_test = X_test.permute(0, 2, 1)
    y_test = torch.FloatTensor(y_test)
    y_test = y_test.permute(0, 2, 1)

    if n_channels == 2:
        train_set = TensorDataset(X_train, y_train)
        val_set = TensorDataset(X_val, y_val)
        test_set = TensorDataset(X_test, y_test)
        torch.save(train_set, os.path.join(data_path, "train_set.pt"))
        torch.save(val_set, os.path.join(data_path, "val_set.pt"))
        torch.save(test_set, os.path.join(data_path, "test_set.pt"))

    elif n_channels == 1:
        train_set = TensorDataset(X_train[:, 0:1], y_train[:, 0:1])
        val_set = TensorDataset(X_val[:, 0:1], y_val[:, 0:1])
        test_set = TensorDataset(X_test[:, 0:1], y_test[:, 0:1])
        torch.save(train_set, os.path.join(data_path, "train_set_real.pt"))
        torch.save(val_set, os.path.join(data_path, "val_set_real.pt"))
        torch.save(test_set, os.path.join(data_path, "test_set_real.pt"))
    print("Dataset ready.")

    return train_set, val_set, test_set
