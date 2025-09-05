import matplotlib.pyplot as plt
import pickle

test_rmse_raw = 0.23559376436058052
test_rmse_combined = 0.21797760607825045

n_feature = 7
n_pretrain = 1000
seed = 2023
sigma = 0.2
lr = "1e-3"
keyword = f"reg_{n_pretrain}"
data_folder = f"mydata_toCombine{seed}_1k5w1k_lr{lr}_lr3e-4_{n_feature}_{sigma}"
result_1k = pickle.load(open(f"./results_{data_folder}/{keyword}_finetuned.pkl", "rb"))
result_1k_combined = pickle.load(open(f"./results_{data_folder}/{keyword}_combined.pkl", "rb"))
result = pickle.load(open(f"./results_{data_folder}/{keyword}.pkl", "rb"))

plt.style.use("bmh")
fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(result["rhos"], result["scores"], label=f"Syn-Boost",  
        color="red",
        marker="s",
        linestyle="-", ms = 10) 
ax.plot(result_1k["rhos"], result_1k["scores"], label=f"Syn-Boost (finetuned with nstep = 1,000)",
       linestyle=":", marker="^", color="red", ms = 10) 
ax.plot(result_1k_combined["rhos"], result_1k_combined["scores"], label=f"Syn-Boost (raw + synthetic)",
       linestyle=":", marker = "o", color = "blue", ms = 10) 
ax.axhline(test_rmse_raw, linestyle="--", color="grey", 
       #label=f"CatBoost (n_train = 500)")
       label=f"CatBoost (raw)")
ax.axhline(test_rmse_combined, linestyle="-.", color="grey", 
       #label=f"CatBoost (n_train + n_pretrain = {n_pretrain + 500})")
       label=f"CatBoost (raw + pretrain)")
ax.axhline(0.2, linestyle="-", color="black", label="Bayes (sqrt)")
ax.set_xlabel("synthetic to raw ratio", fontsize=26)
ax.set_ylabel("RMSE", fontsize=26)
ax.set_title(
    #f"Syn-Boost tuning curves (p = {n_feature}, n_pretrain = 1000)",
    f"Syn-Boost tuning curves (n_pretrain = 1000)",
    weight="bold",
    fontsize=30,
    loc="left",
    y=1.04,
)
ax.legend(fontsize = 22)
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)
plt.subplots_adjust(left=0.11, right=0.99, bottom=0.1, top=0.9)
plt.savefig(f"finetuned-effect-rmse-1k-combined.pdf")
#plt.show()