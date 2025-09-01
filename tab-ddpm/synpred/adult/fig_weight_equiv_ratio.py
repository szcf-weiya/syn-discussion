import pickle
import matplotlib.pyplot as plt
import numpy as np

ratios = pickle.load(open("ratios.pkl", "rb"))
scores_combined = pickle.load(open("scores.pkl", "rb"))

ratio = 10
scores1 = pickle.load(open("scores_ratio10_fakeweight0.1.pkl", "rb"))
weights1 = pickle.load(open("weights_ratio10_fakeweight0.1.pkl", "rb"))

# ratio = 2
# scores1 = pickle.load(open("scores_ratio2_fakeweight0.1.pkl", "rb"))
# weights1 = pickle.load(open("weights_ratio2_fakeweight0.1.pkl", "rb"))

# ratio = 5
# scores1 = pickle.load(open("scores_ratio5_fakeweight0.1.pkl", "rb"))
# weights1 = pickle.load(open("weights_ratio5_fakeweight0.1.pkl", "rb"))

fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(ratios[1:], 1 - np.array(scores_combined[1:]), marker="^", label="Syn-Boost")
ax.plot(ratio * weights1, 1 - np.array(scores1), marker="*", label="Syn-Boost (weighted CatBoost)")
ax.axhline(1 - scores_combined[0], linestyle="--", label = "CatBoost")
ax.set_xlabel("equivalent synthetic to raw ratio", fontsize=22)
ax.set_ylabel("misclassification error", fontsize=22)
ax.set_title(f"Adult: combine raw and synthetic", 
    weight="bold",
    fontsize=24,
    loc="left",
    y=1.04)
ax.legend(fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig(f"weights_ratio{ratio}_equiv_ratio.pdf")