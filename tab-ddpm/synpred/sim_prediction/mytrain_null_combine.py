SEED = 2023
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
device = "cuda:0"
#device=f"cuda:{torch.cuda.current_device()}"
#device="cuda"
print(device)
import numpy as np
import pandas as pd
from sklearn.preprocessing import quantile_transform

from tqdm import tqdm
import json
import pickle
import matplotlib.pyplot as plt
from sample import TrueSampler

TDDPM_DIR = "/home/lw764/pizhao/syn/tab-ddpm"

import sys
sys.path.insert(0, os.path.join(TDDPM_DIR, "utils/"))

os.environ["REPO_DIR"] = "/home/lw764/pizhao/syn/"

from utils_tabddpm import (
    train_tabddpm,
    generate_sample,
)

import argparse
parser = argparse.ArgumentParser(description="Process 3 arguments.")
# Add two arguments
parser.add_argument("arg1", type=int, help="n_pretrain")
parser.add_argument("arg2", type=str, help="short for n_pretrain")
parser.add_argument("arg3", type=int, help="n_feature")
parser.add_argument("arg4", type=float, help="sigma")
parser.add_argument("arg5", type=float, help="lr")
parser.add_argument("arg6", type=int, help="seed")
parser.add_argument("arg7", type=int, help="nstep")


# Parse the arguments
args = parser.parse_args()

# Use the arguments
print(f"Argument 1: {args.arg1}")
print(f"Argument 2: {args.arg2}")
print(f"Argument 3: {args.arg3}")
print(f"Argument 4: {args.arg4}")
print(f"Argument 5: {args.arg5}")
print(f"Argument 6: {args.arg6}")
print(f"Argument 7: {args.arg7}")

n_pretrain = args.arg1
n_feature = args.arg3
lr = args.arg5
SEED = args.arg6

# Configurations
#sigma = 0.2
sigma = args.arg4
keyword = f"reg_{n_pretrain}"
data_folder=f"mydata_toCombine{SEED}_{args.arg2}_{n_feature}_{sigma}"
synthetic_sample_dir = f"./ckpt_{data_folder}/{keyword}/"

## Raw data split
#n_train = 500000  # raw training size
n_train = 500
n_val = 200  # validation size
n_test = 1000  # test or evaluation size

true_sampler = TrueSampler(sigma=sigma, null_feature = True, num_features = n_feature)
np.random.seed(SEED)

X_pretrain, y_pretrain = true_sampler.sample(n_pretrain)
X_train, y_train = true_sampler.sample(n_train)
X_val, y_val = true_sampler.sample(n_val)
X_test, y_test = true_sampler.sample(n_test)
#X_pretrain, y_pretrain = X_train, y_train
X_combined = np.r_[X_train, X_pretrain]
y_combined = np.r_[y_train, y_pretrain]


# Save the data in format suggested by the TDDPM repo
raw_data_dir = os.path.join(TDDPM_DIR, f"{data_folder}/reg_raw")
if not os.path.exists(raw_data_dir):
    os.makedirs(raw_data_dir)

    print(f"Saving raw data to {raw_data_dir} ...")

    np.save(os.path.join(raw_data_dir, "X_num_train.npy"), X_train)
    np.save(os.path.join(raw_data_dir, "y_train.npy"), y_train)

    np.save(os.path.join(raw_data_dir, "X_num_val.npy"), X_val)
    np.save(os.path.join(raw_data_dir, "y_val.npy"), y_val)

    np.save(os.path.join(raw_data_dir, "X_num_test.npy"), X_test)
    np.save(os.path.join(raw_data_dir, "y_test.npy"), y_test)

    info_dict = {
        "task_type": "regression",
        "name": "reg_raw",
        "id": "reg_raw",
        "train_size": n_train,
        "val_size": n_val,
        "test_size": n_test,
        "n_num_features": X_test.shape[1],
    }
    print(f"Saving raw dataset meta information to {raw_data_dir} ...")
    json.dump(info_dict, open(os.path.join(raw_data_dir, "info.json"), "w"))
else:
    print(
        f"Raw data information already exists in {raw_data_dir}, use existing validation and test set."
    )

    # use the same validation and test set in the pre-training configuration
    X_val = np.load(os.path.join(raw_data_dir, "X_num_val.npy"))
    y_val = np.load(os.path.join(raw_data_dir, "y_val.npy"))
    X_test = np.load(os.path.join(raw_data_dir, "X_num_test.npy"))
    y_test = np.load(os.path.join(raw_data_dir, "y_test.npy"))


pretrain_data_dir = os.path.join(TDDPM_DIR, f"{data_folder}/reg_{n_pretrain}")
if not os.path.exists(pretrain_data_dir):
    os.makedirs(pretrain_data_dir)

    print(f"Saving pre-training data to {pretrain_data_dir} ...")

    np.save(os.path.join(pretrain_data_dir, "X_num_train.npy"), X_pretrain)
    np.save(os.path.join(pretrain_data_dir, "y_train.npy"), y_pretrain)

    np.save(os.path.join(pretrain_data_dir, "X_num_val.npy"), X_val)
    np.save(os.path.join(pretrain_data_dir, "y_val.npy"), y_val)

    np.save(os.path.join(pretrain_data_dir, "X_num_test.npy"), X_test)
    np.save(os.path.join(pretrain_data_dir, "y_test.npy"), y_test)

    info_dict = {
        "task_type": "regression",
        "name": f"reg_{n_pretrain}",
        "id": f"reg_{n_pretrain}",
        "train_size": n_pretrain,
        "val_size": n_val,
        "test_size": n_test,
        "n_num_features": X_test.shape[1],
    }
    print(f"Saving pre-training dataset meta information to {pretrain_data_dir} ...")
    json.dump(info_dict, open(os.path.join(pretrain_data_dir, "info.json"), "w"))
else:
    print(f"Pre-training data information already exists in {pretrain_data_dir}")


combined_data_dir = os.path.join(TDDPM_DIR, f"{data_folder}/reg_{n_pretrain}_combined")
if not os.path.exists(combined_data_dir):
    os.makedirs(combined_data_dir)

    print(f"Saving combined-training data to {combined_data_dir} ...")

    np.save(os.path.join(combined_data_dir, "X_num_train.npy"), X_combined)
    np.save(os.path.join(combined_data_dir, "y_train.npy"), y_combined)

    np.save(os.path.join(combined_data_dir, "X_num_val.npy"), X_val)
    np.save(os.path.join(combined_data_dir, "y_val.npy"), y_val)

    np.save(os.path.join(combined_data_dir, "X_num_test.npy"), X_test)
    np.save(os.path.join(combined_data_dir, "y_test.npy"), y_test)

    info_dict = {
        "task_type": "regression",
        "name": f"reg_{n_pretrain}_combined",
        "id": f"reg_{n_pretrain}_combined",
        "train_size": n_pretrain + n_train,
        "val_size": n_val,
        "test_size": n_test,
        "n_num_features": X_test.shape[1],
    }
    print(f"Saving combined-training dataset meta information to {combined_data_dir} ...")
    json.dump(info_dict, open(os.path.join(combined_data_dir, "info.json"), "w"))
else:
    print(f"Combined-training data information already exists in {combined_data_dir}")



keyword = f"reg_{n_pretrain}"
keyword


train_tabddpm(
    pipeline_config_path=f"./ckpt/base_config_p{n_feature}.toml",
    real_data_dir=os.path.join(TDDPM_DIR, f"{data_folder}/{keyword}"),
    #steps=1000000,
    steps=50000,
    #lr = lr,
    temp_parent_dir=synthetic_sample_dir,
    device=device,
)

print(f"Pre-training finished. Pre-trained model saved to {synthetic_sample_dir}")


generate_sample(
    pipeline_config_path=f"./ckpt_{data_folder}/{keyword}/config.toml",
    ckpt_path=f"./ckpt_{data_folder}/{keyword}/model.pt",
    num_samples=10000,
    batch_size=10000,
    temp_parent_dir=synthetic_sample_dir,
)

########## Fine-tune the model on raw data ##########
ckpt_dir = f"./ckpt_{data_folder}/{keyword}"


train_tabddpm(
    pipeline_config_path=os.path.join(ckpt_dir, "config.toml"),
    real_data_dir=raw_data_dir,
    ckpt_path=os.path.join(ckpt_dir, "model.pt"),
    pipeline_dict_path=os.path.join(ckpt_dir, "pipeline_dict.joblib"),
    steps=args.arg7,
    #steps=5000,
    lr=3e-6,
    #lr=1e-3,
    temp_parent_dir=f"./ckpt_{data_folder}/{keyword}_finetuned",
    device=device,
)
