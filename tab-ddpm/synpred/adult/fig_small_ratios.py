import pickle
import matplotlib.pyplot as plt
import numpy as np

scores = pickle.load(open('scores_w0.1-1.9.pkl', 'rb'))
ratios = pickle.load(open('ratios_w0.1-1.9.pkl', 'rb'))

scores_combined = pickle.load(open("scores.pkl", "rb"))
ratios_combined = pickle.load(open("ratios.pkl", "rb"))


fig, ax = plt.subplots(figsize=(14, 8))
#ax.plot(ratios_01_09, 1 - np.array(scores_01_09))
#ax.plot(ratios_01_09, 1 - np.array(scores_01_09))
ax.plot(ratios_combined[1:], 1 - np.array(scores_combined[1:]), marker = "^", color = "red", label = "Syn-Boost")
ax.plot(ratios[:10], 1 - np.array(scores[:10]), marker = "x", label = "Syn-Boost (small ratios)")
ax.axhline(1 - scores_combined[0], linestyle="--", label = "CatBoost")
ax.set_xlabel("synthetic to raw ratio", fontsize=22)
ax.set_ylabel("misclassification error", fontsize=22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
ax.legend(fontsize = 20)
plt.savefig("adult-small-ratios.pdf", bbox_inches='tight')