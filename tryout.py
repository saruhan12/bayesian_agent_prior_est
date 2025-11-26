import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
def gumbal(x,x0,b):
    #reparam of gumbal as 10^(b(x-a) = e(ln(10)(b(x-a)))
    #
    return  1 - np.exp(-np.exp(b*(x-x0)))

file = "experiment_results_lvl1_combined_sorted.csv"
stims = "Stimulus 1 Value"
choice = "Comparison Result"


df = pd.read_csv(file)
x = np.asarray(df[stims],dtype=float)
y = np.asarray(df[choice],dtype=float)

x0 = np.median(x)
b = 1e-6

popt, pcov = curve_fit(gumbal, x, y, p0=[x0, b])
prior_mean_est, slope_est = popt

print("\n===== PSYCHOMETRIC FIT RESULTS =====")
print(f"Estimated Prior Mean (decision boundary): {prior_mean_est:.3f}")
print(f"Slope (sensitivity to difference):        {slope_est:.3f}\n")

# --- PLOT ---

xx = np.linspace(min(x), max(x), 400)
yy = gumbal(xx, prior_mean_est, slope_est)

plt.figure(figsize=(6,4))
plt.scatter(x, y, alpha=0.4, label="Observed choices")
plt.plot(xx, yy, 'r', linewidth=2, label=f"Fit curve\nPrior ≈ {prior_mean_est:.2f}")
plt.xlabel("Stimulus 1 Value")
plt.ylabel("P(choose S1)")
plt.title("Psychometric Curve Fit")
plt.legend()
plt.tight_layout()
plt.show