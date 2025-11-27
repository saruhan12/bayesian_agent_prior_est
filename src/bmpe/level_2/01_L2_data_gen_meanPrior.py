import pandas as pd
import numpy as np
import os

# LEVEL 2

# This script computes the input data for the experiment website. It generates a csv with the following columns:
# S1_A_val: stimulus 1 value for cue A
# S1_A_std: stimulus 1 standard deviation for cue A
# S1_B_val: stimulus 1 value for cue B
# S1_B_std: stimulus 1 standard deviation for cue B
# S2_A_val: stimulus 2 value for cue A
# S2_A_std: stimulus 2 standard deviation for cue A
# S2_B_val: stimulus 2 value for cue B
# S2_B_std: stimulus 2 standard deviation for cue B
# Trial: trial number
# Trial_condition: condition number (1 or 2), where condition 1 is same size cues and condition 2 is S1_A != S1_B;  S1_A + S1_B2 = 2 * S

# CONSTRAINTS
# S_val ∈ [-10.00, 10.00] S_std ∈ [0.00, 8.00].
# Cue B has a fixed, hidden standard deviation; cue A’s reliability can vary from trial to trial.

# ==========================================
# DATA CONFIG
# ==========================================
data_dir = "data"
experiment_name = "level_2"
type_of_data = "website_input"
folder = os.path.join(data_dir, experiment_name, type_of_data)

number_of_exp_attempt = 1
output_file_name = f"inputs_level_1_attempt_{number_of_exp_attempt}.csv"

# === GENERATE VALUES ===
S1_B_std = 4.0          # fixed hidden std for cue B

S2_B_std = 4.0          # fixed hidden std for cue B


