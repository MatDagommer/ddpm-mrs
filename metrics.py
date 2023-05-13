import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import scipy

def SSD(y, y_pred):
    return np.sum(np.square(y - y_pred), axis=1)  # axis 1 is the signal dimension


def MAD(y, y_pred):
    return np.max(np.abs(y - y_pred), axis=1) # axis 1 is the signal dimension


def PRD(y, y_pred):
    N = np.sum(np.square(y_pred - y), axis=1)
    D = np.sum(np.square(y_pred - np.mean(y)), axis=1)

    PRD = np.sqrt(N/D) * 100

    return PRD


# def COS_SIM(y, y_pred):
#     cos_sim = []
#     y = np.squeeze(y, axis=-1)
#     y_pred = np.squeeze(y_pred, axis=-1)
#     for idx in range(len(y)):
#         kl_temp = cosine_similarity(y[idx].reshape(1, -1), y_pred[idx].reshape(1, -1))
#         cos_sim.append(kl_temp)
#
#     cos_sim = np.array(cos_sim)
#     return cos_sim

    
def SNR(y1,y2):
    print("y1 shape: ", y1.shape)
    print("y2 shape: ", y2.shape)
    #N = np.sum(np.abs(y1), axis=1)
    #D = np.sum(np.square(y2 - y1), axis=1)
    D = np.sum(np.square(y2 - y1), axis=1)
    N = np.zeros_like(D) + np.max(np.abs(y1), axis = 1)
    SNR = 20*np.log10(np.divide(N,D))
    print("SNR shape: ", SNR.shape)
    
    return SNR
    
def SNR_improvement(y_in, y_out, y_clean):
    return SNR(y_clean, y_out)-SNR(y_clean, y_in)
    
""" Source Code:
@inproceedings{agarla2021spectralmeasures,
 author = {Agarla, Mirko and Bianco, Simone and Celona, Luigi and Schettini, Raimondo and Tchobanou, Mikhail},
 year = {2021},
 title = {An analysis of spectral similarity measures},
 organization = {Society for Imaging Science and Technology},
 booktitle = {Color and Imaging Conference},
 volume = {2021},
 number = {6},
 doi = {https://doi.org/10.2352/issn.2169-2629.2021.29.300},
 pages = {300--305},
}
"""
def computeGFC(groundTruth, recovered):

    groundTruth[np.where(groundTruth > 1)] = 1
    groundTruth[np.where(groundTruth < 0)] = 0

    recovered[np.where(recovered > 1)] = 1
    recovered[np.where(recovered < 0)] = 0

    GFCn = np.sum(np.multiply(groundTruth, recovered))
    GFCd = np.multiply(np.sqrt(np.sum(groundTruth**2)), np.sqrt(np.sum(recovered**2)))
    GFC = np.divide(GFCn, GFCd)

    return GFC


def PSNR(original, compressed):

    mse = np.mean((original.reshape(-1) - compressed.reshape(-1))**2)
    if mse == 0:
        return 100

    psnr = 20 * np.log10(np.max(original.reshape(-1)) / np.sqrt(mse))
    return psnr

def computePearson(groundTruth, recovered):

    groundTruth = np.nan_to_num(groundTruth).reshape(-1)
    recovered = np.nan_to_num(recovered).reshape(-1)

    batch_pearson = scipy.stats.pearsonr(recovered, groundTruth)
    return batch_pearson