#!pip install audiomentations
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import librosa

from tqdm.auto import tqdm

import warnings
warnings.filterwarnings(action='ignore')

# ============================ Random Seed  ===================================
CFG = {
    'SR':16000,
    'N_MFCC':39,
    'SEED':1209
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(CFG['SEED']) # Seed 고정


# ============================ Augmentation ===================================
augment = Compose([
    TimeStretch(min_rate=0.7, max_rate=1.3, p=0.7),
    PitchShift(min_semitones=-2, max_semitones=4, p=0.9),
    Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
    Trim(p=1),Gain(p=1),
    PolarityInversion(p=0.8)
])

save_path = '/content/drive/MyDrive/dacon_covid/augmented_train'

def augmentationOne(df, data_type, root_path, save_path):
    root_folder = os.path.join(root_path, data_type)

    for uid in tqdm(df['id']):
        path = os.path.join(root_folder, str(uid).zfill(5)+'.wav')
        y, _ = librosa.load(path, sr=CFG['SR'])

        if int(df.loc[df['id']==uid, 'covid19'].values):
            augmented_y = augment(y, CFG['SR'])
            final_save_path = os.path.join(save_path, str(uid).zfill(5)+'.wav')
            write(final_save_path, CFG['SR'], augmented_y)

augmentationOne(train_df, 'train', path, save_path)
