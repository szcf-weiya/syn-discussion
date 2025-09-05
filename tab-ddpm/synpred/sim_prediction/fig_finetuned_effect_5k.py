import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

n_pretrain = 5000
n_feature = 7
seed = 2023
sigma = 0.2
lr = "1e-3"
keyword = f"reg_{n_pretrain}"
data_folder = f"mydata_toCombine{seed}_5k5w1k_lr{lr}_lr3e-6_{n_feature}_{sigma}"
synthetic_sample_dir = f"./ckpt_{data_folder}/{keyword}_finetuned"
loss_1k = np.genfromtxt(os.path.join(synthetic_sample_dir, "loss.csv"), delimiter = ",", skip_header=1)
result_1k = pickle.load(open(f"./results_{data_folder}/{keyword}_finetuned.pkl", "rb"))
result = pickle.load(open(f"./results_{data_folder}/{keyword}.pkl", "rb"))

data_folder = f"mydata_toCombine{seed}_5k5w1w_lr{lr}_lr3e-6_{n_feature}_{sigma}"
synthetic_sample_dir = f"./ckpt_{data_folder}/{keyword}_finetuned"
loss_1w = np.genfromtxt(os.path.join(synthetic_sample_dir, "loss.csv"), delimiter = ",", skip_header=1)
result_1w = pickle.load(open(f"./results_{data_folder}/{keyword}_finetuned.pkl", "rb"))

data_folder = f"mydata_toCombine{seed}_5k5w10w_lr{lr}_lr3e-6_{n_feature}_{sigma}"
synthetic_sample_dir = f"./ckpt_{data_folder}/{keyword}_finetuned"
loss_10w = np.genfromtxt(os.path.join(synthetic_sample_dir, "loss.csv"), delimiter = ",", skip_header=1)
result_10w = pickle.load(open(f"./results_{data_folder}/{keyword}_finetuned.pkl", "rb"))

plt.style.use("bmh")
fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(loss_1k[:,0], loss_1k[:,3], label = "nstep = 1,000", color = "red")
ax.plot(loss_1w[:,0], loss_1w[:,3], label = "nstep = 10,000", color = "blue")
ax.plot(loss_10w[:,0], loss_10w[:,3], label = "nstep = 100,000", color = "orange")
ax.set_xlabel("training step", fontsize = 26)
ax.set_ylabel("loss", fontsize = 26)
ax.set_title(
    f"Syn-Boost fine-tuning (n_pretrain = 5000)",
    weight="bold",
    fontsize=30,
    loc="left",
    y=1.04,
)
ax.legend(fontsize = 24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.savefig(f"finetuned-effect-loss-5k.pdf", bbox_inches='tight')
#plt.show()

## calculated from calc_baseline_raw_and_syn.py
test_rmse_raw = 0.22319979348920505
test_rmse_combined = 0.20306243462385987

plt.style.use("bmh")
fig, ax = plt.subplots(figsize=(14, 8))
#ax.plot(result_100w["rhos"], result_100w["scores"], label=f"100w") 
#ax.plot(result_50w["rhos"], result_50w["scores"], label=f"50w") 
#ax.plot(result_20w["rhos"], result_20w["scores"], label=f"20w") 
ax.plot(result["rhos"], result["scores"], label=f"Syn-Boost",  
        color="red",
        marker="s",
        linestyle="-", ms = 10) 
ax.plot(result_1k["rhos"], result_1k["scores"], label=f"Syn-Boost (finetuned with nstep = 1,000)",
       linestyle=":", marker="^", color="red", ms = 10) 
ax.plot(result_1w["rhos"], result_1w["scores"], label=f"Syn-Boost (finetuned with nstep = 10,000)",
       linestyle=":", marker = "o", color = "blue", ms = 10) 
ax.plot(result_10w["rhos"], result_10w["scores"], label=f"Syn-Boost (finetuned with nstep = 100,000)",
       linestyle=":", marker = "x", color = "orange", ms = 10) 
ax.axhline(test_rmse_raw, linestyle="--", color="grey", label=f"CatBoost (raw)")
ax.axhline(test_rmse_combined, linestyle="-.", color="grey", label=f"CatBoost (raw + pretrain)")
ax.axhline(0.2, linestyle="-", color="black", label="Bayes (sqrt)")
ax.set_xlabel("synthetic to raw ratio", fontsize=26)
ax.set_ylabel("RMSE", fontsize=26)
ax.set_title(
    f"Syn-Boost tuning curves (n_pretrain = 5000)",
    weight="bold",
    fontsize=30,
    loc="left",
    y=1.04,
)
ax.legend(fontsize = 20)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.savefig(f"finetuned-effect-rmse-5k.pdf", bbox_inches='tight')
#plt.show()