import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter


def main():
    # Ensure directories exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)

    # Load dataset
    data_path = os.path.join("data", "heartfailure.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'. Please ensure 'heartfailure.csv' is in the 'data/' directory."
        )

    df = pd.read_csv(data_path)

    # Clean column names if necessary (handle casing variations)
    df.columns = [c.lower() for c in df.columns]

    # Standard column mapping check
    time_col = "time"
    event_col = "death_event"
    ef_col = "ejection_fraction"
    age_col = "age"
    creat_col = "serum_creatinine"

    # Set visualization theme
    sns.set_theme(style="whitegrid")

    # =========================================================================
    # Problem 3(a): Kaplan-Meier Survival Curves by Ejection Fraction (10 pts)
    # =========================================================================
    plt.figure(figsize=(9, 6))

    # Stratify by EF threshold (< 35% vs >= 35%)
    group_low = df[df[ef_col] < 35]
    group_high = df[df[ef_col] >= 35]

    kmf_low = KaplanMeierFitter()
    kmf_high = KaplanMeierFitter()

    kmf_low.fit(
        group_low[time_col],
        event_observed=group_low[event_col],
        label="EF < 35% (Low EF)",
    )
    kmf_high.fit(
        group_high[time_col],
        event_observed=group_high[event_col],
        label="EF ≥ 35% (Normal/Mild EF)",
    )

    ax = plt.subplot(111)
    kmf_low.plot_survival_function(ax=ax, color="crimson", linewidth=2.5)
    kmf_high.plot_survival_function(ax=ax, color="navy", linewidth=2.5)

    plt.title(
        "3(a) Kaplan-Meier Survival Estimates by Ejection Fraction Group",
        fontweight="bold",
        fontsize=12,
    )
    plt.xlabel("Follow-up Period (Days)", fontweight="bold")
    plt.ylabel("Survival Probability", fontweight="bold")
    plt.ylim(0, 1.02)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    fig_a_path = os.path.join(
        "results", "figures", "problem3a_km_survival_curves.png"
    )
    plt.savefig(fig_a_path, dpi=300)
    plt.show()

    # =========================================================================
    # Problem 3(b): Cox Proportional Hazards Model & Prediction (10 pts)
    # =========================================================================

    # Fit Cox PH Model with age, serum_creatinine, and ejection_fraction
    cph_data = df[[time_col, event_col, age_col, creat_col, ef_col]].copy()
    cph = CoxPHFitter()
    cph.fit(cph_data, duration_col=time_col, event_col=event_col)

    print("\n--- Cox Proportional Hazards Model Summary ---")
    cph.print_summary()

    # (i) Forest Plot for Hazard Ratios
    plt.figure(figsize=(8, 4))
    cph.plot()
    plt.title("3(b.i) Cox PH Model Hazard Ratio Forest Plot", fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    fig_bi_path = os.path.join(
        "results", "figures", "problem3bi_hazard_ratio_forest_plot.png"
    )
    plt.savefig(fig_bi_path, dpi=300)
    plt.show()

    # (ii) Predicted Survival Curves varying EF (25%, 40%, 55%) holding age & creatinine at median
    median_age = df[age_col].median()
    median_creat = df[creat_col].median()

    # Create hypothetical patient profiles
    covariates_df = pd.DataFrame(
        {
            age_col: [median_age, median_age, median_age],
            creat_col: [median_creat, median_creat, median_creat],
            ef_col: [25, 40, 55],
        },
        index=["EF = 25%", "EF = 40%", "EF = 55%"],
    )

    pred_survival = cph.predict_survival_function(covariates_df)

    plt.figure(figsize=(9, 6))
    colors = ["darkred", "orange", "forestgreen"]
    for idx, col in enumerate(pred_survival.columns):
        plt.plot(
            pred_survival.index,
            pred_survival[col],
            label=f"{col} (Age={median_age:.0f}, Creatinine={median_creat:.1f})",
            color=colors[idx],
            linewidth=2.5,
        )

    plt.title(
        "3(b.ii) Predicted Cox PH Survival Curves by Ejection Fraction",
        fontweight="bold",
        fontsize=12,
    )
    plt.xlabel("Follow-up Period (Days)", fontweight="bold")
    plt.ylabel("Predicted Survival Probability", fontweight="bold")
    plt.ylim(0, 1.02)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    fig_bii_path = os.path.join(
        "results", "figures", "problem3bii_predicted_survival_ef.png"
    )
    plt.savefig(fig_bii_path, dpi=300)
    plt.show()

    # =========================================================================
    # Problem 3(c): Custom Scale Hazard Ratios (5 pts)
    # =========================================================================

    # Extract log-hazard coefficients (beta)
    b_age = cph.params_[age_col]
    b_creat = cph.params_[creat_col]
    b_ef = cph.params_[ef_col]

    # Calculate scaled hazard ratios: HR = exp(beta * delta)
    hr_age_10yr = np.exp(b_age * 10)
    hr_creat_05 = np.exp(b_creat * 0.5)
    hr_ef_10pct = np.exp(b_ef * 10)

    # 95% Confidence Intervals
    se_age = cph.standard_errors_[age_col]
    se_creat = cph.standard_errors_[creat_col]
    se_ef = cph.standard_errors_[ef_col]

    ci_age = (np.exp((b_age - 1.96 * se_age) * 10), np.exp((b_age + 1.96 * se_age) * 10))
    ci_creat = (np.exp((b_creat - 1.96 * se_creat) * 0.5), np.exp((b_creat + 1.96 * se_creat) * 0.5))
    ci_ef = (np.exp((b_ef - 1.96 * se_ef) * 10), np.exp((b_ef + 1.96 * se_ef) * 10))

    hr_table = pd.DataFrame(
        {
            "Predictor Unit Increase": [
                "+10 Years in Age",
                "+0.5 mg/dL in Serum Creatinine",
                "+10% in Ejection Fraction",
            ],
            "Hazard Ratio (HR)": [hr_age_10yr, hr_creat_05, hr_ef_10pct],
            "95% CI Lower": [ci_age[0], ci_creat[0], ci_ef[0]],
            "95% CI Upper": [ci_age[1], ci_creat[1], ci_ef[1]],
        }
    )

    print("\n--- Problem 3(c) Scaled Hazard Ratios ---")
    print(hr_table.to_string(index=False))

    hr_table_path = os.path.join(
        "results", "tables", "problem3c_scaled_hazard_ratios.csv"
    )
    hr_table.to_csv(hr_table_path, index=False)


if __name__ == "__main__":
    main()