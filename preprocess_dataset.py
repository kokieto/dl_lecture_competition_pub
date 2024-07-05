import os, sys
import numpy as np
import torch
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm

from src.datasets import ThingsMEGDataset
from src.models import BasicConvClassifier
from src.utils import set_seed

import numpy as np
from scipy.signal import windows
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA
import multiprocessing as mp

ICA_COMPS = 125

def get_patch(idx, X, window, dtype="train"):
    current_patch = X[dtype][idx, :, :]
    mean = np.mean(current_patch, axis=1, keepdims=True)
    std = np.std(current_patch, axis=1, keepdims=True)
    standardized_patch = (current_patch - mean) / std
    windowed_patch = standardized_patch * window
    return windowed_patch


def process_feature(args):
    idx, X, window, dtype, ica = args
    feature = get_patch(idx, X, window, dtype)
    feature = ica.transform(feature.T)
    return np.array(feature)


# def standardize(X):
#     mean = np.mean(X, axis=1, keepdims=True)
#     std = np.std(X, axis=1, keepdims=True)
#     return (X - mean) / std


if __name__ == "__main__":
    data_dir = "D:\\MEG_data"

    dtypes = ["train", "val", "test"]
    X = {dtype: torch.load(os.path.join(data_dir, f"{dtype}_X.pt")) for dtype in dtypes}
    X = {dtype: train_X.to('cpu').detach().numpy().copy() for dtype, train_X in X.items()}

    num_samples, num_channels, time_length = X['train'].shape
    window = windows.hann(time_length)

    conc_data = []
    hop_num = 5
    for idx in tqdm(range(0, num_samples, hop_num)):
        conc_data.append(get_patch(idx, X, window))
    np_conc_data = np.concatenate(conc_data, axis=1)
    np_conc_data = np_conc_data.astype(np.float32)

    ica = FastICA(n_components=ICA_COMPS) # n_samples, n_features
    ica.fit(np_conc_data.T)

    # for dtype in dtypes:
    #     args = [(idx, X, window, dtype, ica) for idx in range(X[dtype].shape[0])]
    #     features = []
        # with mp.Pool(12) as pool:
        #     for result in pool.imap(process_feature, args):
        #         features.append(result)
        # features = np.array(features)
        # torch.save(torch.tensor(features), os.path.join(data_dir, f"preprocessed_{dtype}_X.pt"))
    
    for dtype in dtypes:
        features = []
        for idx in tqdm(range(X[dtype].shape[0])):
            feature = get_patch(idx, X, window, dtype)
            feature = ica.transform(feature.T)
            features.append(np.array(feature.T))
        features = np.array(features)
        torch.save(torch.tensor(features), os.path.join(data_dir, f"preprocessed_{dtype}_X.pt"))


    # for dtype in dtypes:
    #     features = []
    #     for idx in tqdm(range(X[dtype].shape[0])):
    #         # feature = standardize(X[dtype][idx, :, :])
    #         feature = get_patch(idx, X, window, dtype)
    #         feature = ica.transform(feature.T)
    #         # results = []
    #         # for j in range(feature.shape[1]):
    #         #     arr = feature[:,j]
    #         #     fft_res = np.fft.fft(arr)
    #         #     results.append(10*np.log10(np.abs(fft_res))[1:fft_res.shape[0]//4])
    #         #     # results.append(np.abs(fft_res)[1:fft_res.shape[0]//4])
    #         features.append(np.array(feature))
    #     features = np.array(features)
    #     torch.save(torch.tensor(features), os.path.join(data_dir, f"preprocessed_{dtype}_X.pt"))
        
# if __name__ == "__main__":
#     data_dir = "D:\\MEG_data"

#     dtypes = ["train", "val", "test"]
#     X = {dtype: torch.load(os.path.join(data_dir, f"{dtype}_X.pt")) for dtype in dtypes}
#     X = {dtype: train_X.to('cpu').detach().numpy().copy() for dtype, train_X in X.items()}

#     num_samples, num_channels, time_length = X['train'].shape
#     window = windows.hann(time_length)
#     print(X['train'].shape)

#     for dtype in dtypes:
#         features = []
#         for idx in tqdm(range(X[dtype].shape[0])):
#             feature = get_patch(idx, X, window, dtype).T
#             results = []
#             for j in range(feature.shape[1]):
#                 arr = feature[:,j]
#                 fft_res = np.fft.fft(arr)
#                 results.append(np.abs(fft_res)[0:fft_res.shape[0]//4])
#             features.append(np.array(results))
#         features = np.array(features)
#         torch.save(torch.tensor(features), os.path.join(data_dir, f"preprocessed_{dtype}_X.pt"))
        