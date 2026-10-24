import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main():
    # Ensure directories exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)

    # Load sleepstudy dataset
    data_path = os.path.join("data", "sleepstudy.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'. Please ensure 'sleepstudy.csv' is in the 'data/' directory."
        )

    df = pd.read_csv(data_path)

    # =========================================================================
    # Problem 2(a): Visualization of Individual Trajectories (10 pts)
    # =========================================================================
    subjects = sorted(df["Subject"].unique())
    num_subjects = len(subjects)
    
    # Grid layout for individual subplots
    ncols = 6
    nrows = int(np.ceil(num_subjects / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3 * nrows), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, subj in enumerate(subjects):
        ax = axes[idx]
        subj_df = df[df["Subject"] == subj]

        # Scatter plot of reaction times
        ax.scatter(subj_df["Days"], subj_df["Reaction"], color="indigo", alpha=0.7, s=30)

        # Fit individual OLS regression line
        m, b = np.polyfit(subj_df["Days"], subj_df["Reaction"], 1)
        x_vals = np.array([subj_df["Days"].min(), subj_df["Days"].max()])
        ax.plot(x_vals, m * x_vals + b, color="crimson", linewidth=1.5, linestyle="--")

        ax.set_title(f"Subject {subj}", fontsize=10, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)

    # Remove unused subplots
    for idx in range(num_subjects, len(axes)):
        fig.delaxes(axes[idx])

    fig.supxlabel("Days of Sleep Deprivation", fontweight="bold", y=0.01)
    fig.supylabel("Average Reaction Time (ms)", fontweight="bold", x=0.01)
    fig.suptitle("Problem 2(a): Individual Trajectories & OLS Fits across Days", fontsize=14, fontweight="bold")

    plt.tight_layout()
    fig_path = os.path.join("results", "figures", "problem2a_individual_trajectories.png")
    plt.savefig(fig_path, dpi=300)
    plt.show()

    # =========================================================================
    # Problem 2(b): Model Comparisons (OLS vs Mixed Models) (10 pts)
    # =========================================================================
    
    # 1. Standard OLS Model (Ignores subject-level grouping)
    ols_mod = smf.ols("Reaction ~ Days", data=df).fit()

    # 2. Random Intercept Model
    # Reaction ~ Days, random intercept for each Subject
    ri_mod = smf.mixedlm("Reaction ~ Days", df, groups=df["Subject"]).fit()

    # 3. Random Slope Model
    # Reaction ~ Days, random slope for Days (no random intercept, re_formula="0 + Days")
    rs_mod = smf.mixedlm("Reaction ~ Days", df, groups=df["Subject"], re_formula="0 + Days").fit()

    # 4. Random Intercept and Random Slope Model
    # Reaction ~ Days, random intercept and slope for Days (re_formula="1 + Days")
    ris_mod = smf.mixedlm("Reaction ~ Days", df, groups=df["Subject"], re_formula="~Days").fit()

    # Model Evaluation Metrics Table
    comparison_df = pd.DataFrame({
        "Model": [
            "Ordinary Least Squares (OLS)",
            "Random Intercept",
            "Random Slope",
            "Random Intercept & Slope"
        ],
        "Log-Likelihood": [
            ols_mod.llf,
            ri_mod.llf,
            rs_mod.llf,
            ris_mod.llf
        ],
        "AIC": [
            ols_mod.aic,
            ri_mod.aic,
            rs_mod.aic,
            ris_mod.aic
        ],
        "BIC": [
            ols_mod.bic,
            ri_mod.bic,
            rs_mod.bic,
            ris_mod.bic
        ]
    })

    print("\n--- Problem 2 Model Comparison Metrics ---")
    print(comparison_df.to_string(index=False))

    table_path = os.path.join("results", "tables", "problem2_model_comparison.csv")
    comparison_df.to_csv(table_path, index=False)

    # Print summary for best fitting model (Random Intercept & Slope)
    print("\n--- Best Model Summary (Random Intercept & Slope) ---")
    print(ris_mod.summary())


if __name__ == "__main__":
    main()