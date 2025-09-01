import os

TDDPM_DIR = "/home/lw764/pizhao/syn/tab-ddpm"
os.environ["REPO_DIR"] = "/home/lw764/pizhao/syn/"
REPO_DIR = os.environ.get("REPO_DIR")


import numpy as np

from tqdm import tqdm
import pickle

import sys
sys.path.insert(0, os.path.join(TDDPM_DIR, "utils/"))

from utils_tabddpm import (
    generate_sample,
)

from utils_syn import (
    concat_data,
    catboost_pred_model,
    test_rmse,
)

import argparse
parser = argparse.ArgumentParser(description="Process 3 arguments.")
# Add two arguments
parser.add_argument("arg1", type=int, help="First argument")
parser.add_argument("arg2", type=str, help="Second argument")
parser.add_argument("arg3", type=int, help="n_feature")
parser.add_argument("arg4", type=float, help="sigma")
parser.add_argument("arg5", type=float, help="lr")
parser.add_argument("arg6", type=int, help="seed")

# Parse the arguments
args = parser.parse_args()

# Use the arguments
print(f"Argument 1: {args.arg1}")
print(f"Argument 2: {args.arg2}")
print(f"Argument 3: {args.arg3}")
print(f"Argument 4: {args.arg4}")
print(f"Argument 5: {args.arg5}")
print(f"Argument 6 seed: {args.arg6}")


n_pretrain = args.arg1
n_feature = args.arg3
lr = args.arg5

#data_folder = "mydata_5w"
data_folder = f"mydata{args.arg6}_{args.arg2}_{args.arg3}_{args.arg4}"

keyword = f"reg_{n_pretrain}"
ckpt_dir = f"./ckpt_{data_folder}/{keyword}"
raw_data_dir = os.path.join(TDDPM_DIR, f"{data_folder}/reg_raw")
pretrain_data_dir = os.path.join(TDDPM_DIR, f"{data_folder}/reg_{n_pretrain}")

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
print("Regression using raw training data:")
print("Validation:", raw_model.get_best_score())
print("Test RMSE:", test_rmse_raw)


pretrain_df = concat_data(pretrain_data_dir, split="train")

pretrain_model = catboost_pred_model(
    pretrain_df,
    val_df,
    num_features_list=num_features_list,
    iterations=2000,
    loss_function="RMSE",
    verbose=False,
)

print("Regression using pre-training data:")
print("Validation:", pretrain_model.get_best_score())
print("Test RMSE:", test_rmse(pretrain_model, test_df))

if not os.path.exists(f"./results_{data_folder}"):
    os.makedirs(f"./results_{data_folder}")
    
########## Syn-Boost tuning ############
#rho_min, rho_max, step_size = 1, 30, 1
#rho_list = np.linspace(rho_min, rho_max, int((rho_max - rho_min) / step_size) + 1)
#
#result_dict = {"rhos": rho_list[19:], "scores": []}
#
#for rho in tqdm(rho_list[19:]):
#    m = int(len(train_df) * rho)
#
#    temp_dir = generate_sample(
#        pipeline_config_path=f"./ckpt_{data_folder}/{keyword}/config.toml",
#        ckpt_path=f"./ckpt_{data_folder}/{keyword}/model.pt",
#        pipeline_dict_path=f"./ckpt_{data_folder}/{keyword}/pipeline_dict.joblib",
#        num_samples=m,
#        batch_size=m,
#        temp_parent_dir=f"./temp/tmp_{data_folder}",
#    )
#
#    fake_train_df = concat_data(temp_dir, split="train")
#
#    fake_train_model = catboost_pred_model(
#        fake_train_df,
#        val_df,
#        num_features_list=num_features_list,
#        iterations=2000,
#        loss_function="RMSE",
#        verbose=False,
#    )
#
#    score = test_rmse(fake_train_model, test_df)
#    result_dict["scores"].append(score)
#
#    pickle.dump(result_dict, open(f"./results_{data_folder}/{keyword}.pkl", "wb"))
#    print(f"rho = {rho}, m = {m}: Test RMSE is {score}.")
#

######### Syn-Boost tuning ############ (finetuned)
rho_min, rho_max, step_size = 1, 30, 1
rho_list = np.linspace(rho_min, rho_max, int((rho_max - rho_min) / step_size) + 1)

result_dict = {"rhos": rho_list[16:], "scores": []}

for rho in tqdm(rho_list[16:]):
    m = int(len(train_df) * rho)

    temp_dir = generate_sample(
        pipeline_config_path=f"./ckpt_{data_folder}/{keyword}_finetuned/config.toml",
        ckpt_path=f"./ckpt_{data_folder}/{keyword}_finetuned/model.pt",
        pipeline_dict_path=f"./ckpt_{data_folder}/{keyword}_finetuned/pipeline_dict.joblib",
        num_samples=m,
        batch_size=m,
        temp_parent_dir=f"./temp/tmp_{data_folder}",
    )

    fake_train_df = concat_data(temp_dir, split="train")

    fake_train_model = catboost_pred_model(
        fake_train_df,
        val_df,
        num_features_list=num_features_list,
        iterations=2000,
        loss_function="RMSE",
        verbose=False,
    )

    score = test_rmse(fake_train_model, test_df)
    result_dict["scores"].append(score)

    pickle.dump(result_dict, open(f"./results_{data_folder}/{keyword}_finetuned.pkl", "wb"))
    print(f"rho = {rho}, m = {m}: Test RMSE is {score}.")
