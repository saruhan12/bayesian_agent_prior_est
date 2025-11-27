import pandas as pd
import numpy as np
import os

# LEVEL 1
# This script computes the input data for the mean prior experiment on the agents website

# ==========================================
# CONFIG
# ==========================================
data_dir = "data"
experiment_name = "experiment_1"
type_of_data = "website_input"
type_of_var = "mean"
output_file_name = "mean_inputs_experiment_1.csv"

folder = os.path.join(data_dir, experiment_name, type_of_data, type_of_var)


# === REFERENCE ===
# We will fix the reference cue (S2) to be very noisy.
# This will ensure that the agent relies mostly on its prior.
S2_val = 0.0            # reference
S2_std = 8.0            # very noisy cue (≈ prior-only)

# === GENERATE S1 VALUES ===
trials_per_s1 = 50
s1_min, s1_max, s1_step = -4.0, 4.0, 0.1
s1_vals = np.arange(s1_min, s1_max + 1e-9, s1_step)

s1_std_vals = np.arange(0.1,0.9,0.1)

rows = []
trial_id = 1
for val_std in s1_std_vals:
    for val_obs in s1_vals:
        for _ in range(trials_per_s1):
            rows.append({
                "Trial": trial_id,
                "S1_val": float(val_obs),
                "S1_std": float(val_std),
                "S2_val": float(S2_val),
                "S2_std": float(S2_std),
            })
            trial_id += 1

df = pd.DataFrame(rows)

print(df.head())
print(df.dtypes)

df.to_csv(folder + "/" + output_file_name, index=False)
