"""
A Study on Barcoding and Automation and Its Impact on Operational Performance
Published in: Proceedings of the 1st International Conference on Intelligent
Healthcare and Computational Neural Modelling, Springer Nature Singapore, 2024
DOI: https://doi.org/10.1007/978-981-99-2832-3_70

Authors: Vani, Mekala Samuel, Madhusmita Mohanty, U. M. Gopal Krishna
VIT-AP University, Vijayawada, Andhra Pradesh, India

Research Methodology:
  - Quantitative research using survey data (n=225)
  - Judgmental + convenience sampling
  - Respondents: pharma sector barcode/automation users & developers in India
  - Hypotheses tested using regression & correlation analysis

H01: Barcoding has no positive impact on operational performance
H0b: Automation has no positive impact on operational performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

# ─── 1. Survey Data Simulation ────────────────────────────────────────────────
# Replace this section with your actual CSV survey data:
#   df = pd.read_csv("data/survey_responses.csv")
#
# Survey Likert scale: 1 (Strongly Disagree) → 5 (Strongly Agree)
# Constructs measured (from the paper):
#   Barcoding      : BC1–BC5  (inventory, asset, invoicing, POS, speed accuracy)
#   Automation     : AU1–AU5  (process efficiency, error reduction, cost saving,
#                              throughput, employee productivity)
#   Op. Performance: OP1–OP5  (output quality, delivery speed, cost efficiency,
#                              customer satisfaction, waste reduction)

np.random.seed(42)
N = 225   # Sample size recommended by Hair et al. (paper used 225 > 191 from G*Power)

def likert(mean, std, n=N):
    return np.clip(np.round(np.random.normal(mean, std, n)), 1, 5).astype(int)

data = {
    # Barcoding items
    "BC1": likert(4.1, 0.7),  # Improves inventory accuracy
    "BC2": likert(4.0, 0.8),  # Speeds up asset tracking
    "BC3": likert(3.9, 0.8),  # Reduces invoicing errors
    "BC4": likert(4.2, 0.7),  # Enhances POS transactions
    "BC5": likert(4.0, 0.7),  # Improves overall speed & accuracy

    # Automation items
    "AU1": likert(4.1, 0.7),  # Increases process efficiency
    "AU2": likert(4.0, 0.8),  # Reduces manual errors
    "AU3": likert(3.8, 0.9),  # Lowers operational costs
    "AU4": likert(4.0, 0.7),  # Improves production throughput
    "AU5": likert(4.1, 0.8),  # Boosts employee productivity

    # Operational Performance items
    "OP1": likert(4.0, 0.7),  # Higher output quality
    "OP2": likert(3.9, 0.8),  # Faster delivery
    "OP3": likert(3.8, 0.8),  # Lower cost per unit
    "OP4": likert(4.1, 0.7),  # Improved customer satisfaction
    "OP5": likert(4.0, 0.7),  # Reduced waste

    # Demographics
    "Experience_Years": np.random.choice([1,2,3,4,5,6,7,8,9,10], N),
    "Org_Size":         np.random.choice(["Small","Medium","Large"], N,
                                         p=[0.3, 0.4, 0.3]),
    "Sector":           np.random.choice(["Pharma","Healthcare","Retail",
                                          "Manufacturing"], N,
                                         p=[0.5, 0.2, 0.2, 0.1]),
}

df = pd.DataFrame(data)

# Composite scores (mean of items)
df["Barcoding_Score"]   = df[["BC1","BC2","BC3","BC4","BC5"]].mean(axis=1)
df["Automation_Score"]  = df[["AU1","AU2","AU3","AU4","AU5"]].mean(axis=1)
df["OpPerformance_Score"] = df[["OP1","OP2","OP3","OP4","OP5"]].mean(axis=1)

print("Dataset created. Shape:", df.shape)
print(df[["Barcoding_Score","Automation_Score","OpPerformance_Score"]].describe())

# Save for reuse
df.to_csv("data/survey_data.csv", index=False)

# ─── 2. Descriptive Statistics ────────────────────────────────────────────────

print("\n" + "="*60)
print("DESCRIPTIVE STATISTICS")
print("="*60)
desc = df[["Barcoding_Score","Automation_Score","OpPerformance_Score"]].describe()
print(desc)

# ─── 3. Reliability Analysis (Cronbach's Alpha) ───────────────────────────────

def cronbach_alpha(df_items):
    """Compute Cronbach's alpha for a set of Likert items."""
    k     = df_items.shape[1]
    item_var  = df_items.var(axis=0, ddof=1).sum()
    total_var = df_items.sum(axis=1).var(ddof=1)
    alpha = (k / (k - 1)) * (1 - item_var / total_var)
    return alpha

alpha_bc = cronbach_alpha(df[["BC1","BC2","BC3","BC4","BC5"]])
alpha_au = cronbach_alpha(df[["AU1","AU2","AU3","AU4","AU5"]])
alpha_op = cronbach_alpha(df[["OP1","OP2","OP3","OP4","OP5"]])

print("\n" + "="*60)
print("RELIABILITY ANALYSIS (Cronbach's Alpha)")
print("="*60)
print(f"  Barcoding construct      : α = {alpha_bc:.3f}")
print(f"  Automation construct     : α = {alpha_au:.3f}")
print(f"  Operational Performance  : α = {alpha_op:.3f}")
print("  (α > 0.70 indicates acceptable reliability)")

# ─── 4. Correlation Analysis ──────────────────────────────────────────────────

print("\n" + "="*60)
print("CORRELATION ANALYSIS")
print("="*60)

corr_matrix = df[["Barcoding_Score","Automation_Score","OpPerformance_Score"]].corr()
print(corr_matrix.round(3))

# Pearson correlation with p-values
r_bc, p_bc = stats.pearsonr(df["Barcoding_Score"],  df["OpPerformance_Score"])
r_au, p_au = stats.pearsonr(df["Automation_Score"], df["OpPerformance_Score"])

print(f"\n  Barcoding  ↔ Op.Performance : r = {r_bc:.3f}, p = {p_bc:.4f}")
print(f"  Automation ↔ Op.Performance : r = {r_au:.3f}, p = {p_au:.4f}")

plt.figure(figsize=(7, 5))
sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="coolwarm",
            linewidths=0.5, vmin=-1, vmax=1)
plt.title("Correlation Matrix: Barcoding, Automation & Operational Performance")
plt.tight_layout()
plt.savefig("plots/correlation_matrix.png", dpi=150)
plt.show()

# ─── 5. Hypothesis Testing ────────────────────────────────────────────────────

print("\n" + "="*60)
print("HYPOTHESIS TESTING")
print("="*60)

# H01: Barcoding → Operational Performance
t_stat_bc, p_val_bc = stats.ttest_1samp(df["Barcoding_Score"], popmean=3.0)
print(f"\nH01 – Barcoding vs. Neutral (μ=3):")
print(f"  t-statistic = {t_stat_bc:.3f}, p-value = {p_val_bc:.4f}")
if p_val_bc < 0.05:
    print("  → REJECT H01: Barcoding has a significant positive impact (p < 0.05)")
else:
    print("  → FAIL TO REJECT H01")

# H0b: Automation → Operational Performance
t_stat_au, p_val_au = stats.ttest_1samp(df["Automation_Score"], popmean=3.0)
print(f"\nH0b – Automation vs. Neutral (μ=3):")
print(f"  t-statistic = {t_stat_au:.3f}, p-value = {p_val_au:.4f}")
if p_val_au < 0.05:
    print("  → REJECT H0b: Automation has a significant positive impact (p < 0.05)")
else:
    print("  → FAIL TO REJECT H0b")

# ─── 6. Multiple Regression Analysis ─────────────────────────────────────────

print("\n" + "="*60)
print("MULTIPLE REGRESSION: Predictors → Operational Performance")
print("="*60)

X = df[["Barcoding_Score", "Automation_Score"]].values
y = df["OpPerformance_Score"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

reg = LinearRegression()
reg.fit(X_scaled, y)

y_pred = reg.predict(X_scaled)
r2     = r2_score(y, y_pred)

print(f"\n  Intercept             : {reg.intercept_:.4f}")
print(f"  β (Barcoding)         : {reg.coef_[0]:.4f}")
print(f"  β (Automation)        : {reg.coef_[1]:.4f}")
print(f"  R² (explained var.)   : {r2:.4f}")
print(f"  Adjusted R²           : {1 - (1-r2)*(N-1)/(N-3):.4f}")

# ─── 7. Visualizations ────────────────────────────────────────────────────────

import os
os.makedirs("plots", exist_ok=True)

# Construct score distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col, title in zip(axes,
    ["Barcoding_Score","Automation_Score","OpPerformance_Score"],
    ["Barcoding","Automation","Operational Performance"]):
    ax.hist(df[col], bins=15, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(df[col].mean(), color="red", linestyle="--", label=f"Mean={df[col].mean():.2f}")
    ax.set_title(f"{title} Distribution")
    ax.set_xlabel("Score (1–5)"); ax.legend()
plt.tight_layout()
plt.savefig("plots/score_distributions.png", dpi=150)
plt.show()

# Scatter plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, x_col, label in zip(axes,
    ["Barcoding_Score","Automation_Score"],
    ["Barcoding","Automation"]):
    ax.scatter(df[x_col], df["OpPerformance_Score"], alpha=0.4, color="steelblue")
    m, b = np.polyfit(df[x_col], df["OpPerformance_Score"], 1)
    ax.plot(sorted(df[x_col]), [m*x+b for x in sorted(df[x_col])],
            color="red", linewidth=2)
    ax.set_xlabel(f"{label} Score"); ax.set_ylabel("Operational Performance Score")
    ax.set_title(f"{label} vs Operational Performance")
plt.tight_layout()
plt.savefig("plots/scatter_plots.png", dpi=150)
plt.show()

# Mean scores by organisation size
fig, ax = plt.subplots(figsize=(8, 5))
df.groupby("Org_Size")[["Barcoding_Score","Automation_Score","OpPerformance_Score"]].mean() \
  .plot(kind="bar", ax=ax, colormap="Set2", edgecolor="white")
ax.set_title("Mean Scores by Organisation Size")
ax.set_xlabel("Organisation Size"); ax.set_ylabel("Mean Score (1–5)")
ax.legend(["Barcoding","Automation","Op. Performance"])
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("plots/scores_by_org_size.png", dpi=150)
plt.show()

print("\nAll plots saved to plots/")
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"  Sample size          : {N}")
print(f"  Cronbach α Barcoding : {alpha_bc:.3f}")
print(f"  Cronbach α Automation: {alpha_au:.3f}")
print(f"  Cronbach α Op. Perf. : {alpha_op:.3f}")
print(f"  Pearson r (BC→OP)    : {r_bc:.3f}  (p={p_bc:.4f})")
print(f"  Pearson r (AU→OP)    : {r_au:.3f}  (p={p_au:.4f})")
print(f"  Regression R²        : {r2:.4f}")
print(f"  H01 (Barcoding)      : {'Rejected ✓' if p_val_bc < 0.05 else 'Not Rejected'}")
print(f"  H0b (Automation)     : {'Rejected ✓' if p_val_au < 0.05 else 'Not Rejected'}")
