TDDPM_DIR = "/home/lw764/pizhao/syn/tab-ddpm"

import sys
import os
sys.path.insert(0, os.path.join(TDDPM_DIR, "utils/"))
from utils_syn import (
    concat_data,
    catboost_pred_model,
    test_rmse,
)

## combined train and pre-trained data

n_feature = 7

#data_folder = "mydata_toCombine2023_1k5w1k_lr1e-3_lr3e-4_7_0.2"
data_folder = "mydata_toCombine2023_5k5w1k_lr1e-3_lr3e-6_7_0.2"

raw_data_dir = os.path.join(TDDPM_DIR, f"{data_folder}/reg_raw")
########## Get raw data and its splits ready ##########

num_features_list = [f"num_{i}" for i in range(n_feature)]

train_df = concat_data(raw_data_dir, split="train")
val_df = concat_data(raw_data_dir, split="val")
test_df = concat_data(raw_data_dir, split="test")

########## Train CatBoost model on raw data ##########

raw_model = catboost_pred_model(
    train_df,
    val_df,
    num_features_list=num_features_list,
    iterations=2000,
    loss_function="RMSE",
    verbose=False,
)

test_rmse_raw = test_rmse(raw_model, test_df)

########### Combine raw and pretrain ################

#combined_data_dir = os.path.join(TDDPM_DIR, f"{data_folder}/reg_1000_combined")
combined_data_dir = os.path.join(TDDPM_DIR, f"{data_folder}/reg_5000_combined")

train_df = concat_data(combined_data_dir, split="train")
val_df = concat_data(combined_data_dir, split="val")
test_df = concat_data(combined_data_dir, split="test")

raw_model = catboost_pred_model(
    train_df,
    val_df,
    num_features_list=num_features_list,
    iterations=2000,
    loss_function="RMSE",
    verbose=False,
)

test_rmse_combined = test_rmse(raw_model, test_df)