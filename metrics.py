import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def SSD(y, y_pred):
    return np.sum(np.square(y - y_pred), axis=1)  # axis 1 is the signal dimension


def MAD(y, y_pred):
    return np.max(np.abs(y - y_pred), axis=1) # axis 1 is the signal dimension


def PRD(y, y_pred):
    N = np.sum(np.square(y_pred - y), axis=1)
    D = np.sum(np.square(y_pred - np.mean(y)), axis=1)

    PRD = np.sqrt(N/D) * 100

    return PRD


def COS_SIM(y, y_pred):
    cos_sim = []
    y = np.squeeze(y, axis=-1)
    y_pred = np.squeeze(y_pred, axis=-1)
    for idx in range(len(y)):
        kl_temp = cosine_similarity(y[idx].reshape(1, -1), y_pred[idx].reshape(1, -1))
        cos_sim.append(kl_temp)

    cos_sim = np.array(cos_sim)
    return cos_sim

    
def SNR(y1,y2):
    N = np.sum(np.square(y1), axis=1)
    D = np.sum(np.square(y2 - y1), axis=1)
    
    SNR = 10*np.log10(N/D)
    
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
    """
    Compute Goodness-of-Fit Coefficient (GFC) between the recovered and the
    corresponding ground-truth image
    :param groundTruth: ground truth reference image.
        numpy.ndarray (Height x Width x Spectral_Dimension)
    :param recovered: image under evaluation.
        numpy.ndarray (Height x Width x Spectral_Dimension)
    Returns:
        GFC between `recovered` and `groundTruth`
    """
    assert groundTruth.shape == recovered.shape, \
        "Size not match for groundtruth and recovered spectral images"
    #Slight modification made
    groundTruth = torch.clip(groundTruth.view(-1).float(), 0, 1)
    recovered = torch.clip(recovered.view(-1).float(), 0, 1)

    GFCn = torch.sum(torch.multiply(groundTruth, recovered))

    GFCd = torch.multiply(torch.sqrt(torch.sum(torch.pow(groundTruth, 2))),
                       torch.sqrt(torch.sum(torch.pow(recovered, 2))))

    GFC = torch.divide(GFCn, GFCd)

    return torch.mean(GFC).cpu().detach().item()


def PSNR(original, compressed):
    mse = torch.mean((original.view(-1).float() - compressed.view(-1).float()) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        return 100  # Therefore PSNR have no importance.

    psnr = 20 * torch.log10(torch.max(original.view(-1).float()) / torch.sqrt(mse))

    return psnr.cpu().detach().item()  # db
    
    
    
    
    