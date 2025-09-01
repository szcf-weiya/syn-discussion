import matplotlib.pyplot as plt
import pickle

test_rmse_raw = 0.22319979348920505
test_rmse_combined = 0.20306243462385987

n_feature = 7
n_pretrain = 5000
seed = 2023
sigma = 0.2
lr = "1e-3"
keyword = f"reg_{n_pretrain}"
data_folder = f"mydata_toCombine{seed}_5k5w1k_lr{lr}_lr3e-6_{n_feature}_{sigma}"
result_finetuned = pickle.load(open(f"./results_{data_folder}/{keyword}_finetuned.pkl", "rb"))
result_combined = pickle.load(open(f"./results_{data_folder}/{keyword}_combined.pkl", "rb"))
result = pickle.load(open(f"./results_{data_folder}/{keyword}.pkl", "rb"))

plt.style.use("bmh")
fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(result["rhos"], result["scores"], label=f"Syn-Boost",  
        color="red",
        marker="s",
        linestyle="-") 
ax.plot(result_finetuned["rhos"], result_finetuned["scores"], label=f"Syn-Boost (finetuned with nstep = 1,000)",
       linestyle=":", marker="^", color="red") 
ax.plot(result_combined["rhos"], result_combined["scores"], label=f"Syn-Boost (raw + synthetic)",
       linestyle=":", marker = "o", color = "blue") 
ax.axhline(test_rmse_raw, linestyle="--", color="grey", 
       #label=f"CatBoost (n_train = 500)")
       label=f"CatBoost (raw)")
ax.axhline(test_rmse_combined, linestyle="-.", color="grey", 
       #label=f"CatBoost (n_train + n_pretrain = {n_pretrain + 500})")
       label=f"CatBoost (raw + pretrain)")
ax.axhline(0.2, linestyle="-", color="black", label="Bayes (sqrt)")
ax.set_xlabel("synthetic to raw ratio", fontsize=22)
ax.set_ylabel("RMSE", fontsize=22)
ax.set_title(
    #f"Syn-Boost tuning curves (p = {n_feature}, n_pretrain = 1000)",
    f"Syn-Boost tuning curves (n_pretrain = 5000)",
    weight="bold",
    fontsize=24,
    loc="left",
    y=1.04,
)
ax.legend(fontsize = 20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig(f"finetuned-effect-rmse-5k-combined.pdf")
#plt.show()