import pickle
import matplotlib.pyplot as plt
import numpy as np

score_catboost = 0.9303703703703704
score_fnn = 0.9259259259259259
score_catboost_combined = 0.9377777777777778

result_female = pickle.load(open(f"./synboost_transfer_adult_female.pkl", "rb"))
result_female_combined = pickle.load(open(f"./synboost_transfer_adult_female_combined_from_male.pkl", "rb"))


## both twin1 and twin2 run this
tuned_value = 1 - np.array(result_female["scores"]).max()
combined_value = 1 - np.array(result_female_combined["scores"]).max()

fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(result_female["rhos"], 1 - np.array(result_female["scores"]), label = "finetuned", marker = "s")
ax.plot(result_female_combined["rhos"], 1 - np.array(result_female_combined["scores"]), label = "combined", marker = "^")
ax.axhline(tuned_value, linestyle="--",lw = 4, color="C0", label = "best finetuned")
ax.axhline(combined_value, linestyle="--", color="C1", label = "best combined")
ax.axhline(1 - score_catboost, linestyle=":", color="C2", label = "CatBoost")
ax.axhline(1 - score_catboost_combined, linestyle=":", lw=4, color="C2", label = "CatBoost (Male + Female)")
# ax.axhline(1 - score_fnn, linestyle="-.", color="C4", label = "FNN")
ax.legend(fontsize = 20)
ax.set_xlabel("synthetic to raw ratio", fontsize=22)
ax.set_ylabel("misclassification error", fontsize=22)
ax.set_title(
    "Syn-Boost (Adult-Female, adult_female_3000_twin_1)",
    weight="bold",
    fontsize=20,
    loc="left",
    y=1.03,
)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig("combined_vs_finetuned-adult-female-twin1.pdf")
