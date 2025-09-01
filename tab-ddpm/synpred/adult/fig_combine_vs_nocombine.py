import pickle
import matplotlib.pyplot as plt
import numpy as np

## obtained by running `prediction.py` after setting whether to combine the fake and raw data (L203 to L212)
scores_combined = pickle.load(open("scores.pkl", "rb"))
scores_nocombine = pickle.load(open("scores_alone.pkl", "rb"))
ratios = pickle.load(open("ratios.pkl", "rb"))

fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(ratios[1:], 1 - np.array(scores_combined[1:]), label = "Syn-Boost (combine synthetic and raw)", marker = "s")
ax.plot(ratios[1:], 1 - np.array(scores_nocombine[1:]), label = "Syn-Boost (only synthetic)", marker = "^")
#ax.axhline(acc_catboost, linestyle="--", label = "CatBoost")
ax.axhline(1-scores_nocombine[0], linestyle="--", label = "CatBoost")
ax.legend(fontsize = 20)
ax.set_xlabel("synthetic to raw ratio", fontsize=22)
ax.set_ylabel("misclassification error", fontsize=22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig("adult-combine-vs-nocombine.pdf")