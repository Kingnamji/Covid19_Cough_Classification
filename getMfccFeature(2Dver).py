import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

import librosa
from tqdm.auto import tqdm

import warnings
warnings.filterwarnings(action='ignore')

def get_mfcc_feature2d(df, data_type, save_path):
    temp_x, temp_y = [], []

    # Data Folder path
    root_folder = '/content/drive/MyDrive/dacon_covid'
    root_folder = os.path.join(root_folder, data_type)
    
    features = []
    idx = 0

    for uid in tqdm(df['id']):
        path = os.path.join(root_folder, str(uid).zfill(5)+'.wav')

        # librosa패키지를 사용하여 wav 파일 load
        y, sr = librosa.load(path, sr=CFG['SR'])
        y = librosa.util.fix_length(y, 128000) # 16000 * 8 = 128000

        # librosa패키지를 사용하여 mfcc 추출
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=CFG['N_MFCC'])
        
        if data_type[-5:] == 'train':
            temp_x.append(mfcc)
            temp_y.append(int(df.loc[df['id']==uid, 'covid19'].values))
        else:
            temp_x.append(mfcc)
    
    if data_type[-5:] == 'train':
        result_x = np.array(temp_x)
        result_y = np.array(temp_y)
    
        save_path_x = save_path + data_type + '_x.npy'
        save_path_y = save_path + data_type + '_y.npy'

        np.save(save_path_x, result_x)
        np.save(save_path_y, result_y)
    else:
        result_x = np.array(temp_x)
        save_path_x = save_path + data_type + '_x.npy'
        np.save(save_path_x, result_x)

get_mfcc_feature2d(train_df, 'train', '/content/drive/MyDrive/dacon_covid/train_mfcc2d/')
get_mfcc_feature2d(test_df, 'test', '/content/drive/MyDrive/dacon_covid/test_mfcc2d/')

# for Augmented Data
temp_df = train_df[train_df['covid19']==1]
temp_df.reset_index(drop=True, inplace=True)
get_mfcc_feature2d(temp_df, 'augmented_train', '/content/drive/MyDrive/dacon_covid/augmented_train_mfcc2d/')
