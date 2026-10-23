import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def main():
    # Ensure required directories exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)

    # Load dataset
    data_path = os.path.join("data", "insurance.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'. Please ensure 'insurance.csv' is in the 'data/' directory."
        )

    df = pd.read_csv(data_path)

    # 1. Fit Ordinary Linear Regression Model (OLS)
    ols_model = smf.ols("charges ~ age + bmi + C(smoker)", data=df).fit()
    df["pred_ols"] = ols_model.predict(df)
    df["resid_ols"] = df["charges"] - df["pred_ols"]

    # 2. Fit Generalized Linear Model (GLM) - Gamma with Log Link
    glm_model = smf.glm(
        "charges ~ age + bmi + C(smoker)",
        data=df,
        family=sm.families.Gamma(link=sm.families.links.Log()),
    ).fit()
    df["pred_glm"] = glm_model.predict(df)
    df["resid_glm"] = df["charges"] - df["pred_glm"]

    # --- Problem 1(b): Visualization ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.set_theme(style="whitegrid")

    # (i) Histogram plot comparing observed vs predicted charges
    # OLS Histogram
    sns.histplot(
        df["charges"],
        label="Observed",
        color="blue",
        alpha=0.4,
        kde=True,
        ax=axes[0, 0],
    )
    sns.histplot(
        df["pred_ols"],
        label="OLS Predicted",
        color="red",
        alpha=0.4,
        kde=True,
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("OLS: Observed vs Predicted Charges")
    axes[0, 0].set_xlabel("Hospital Charges ($)")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].legend()

    # GLM Histogram
    sns.histplot(
        df["charges"],
        label="Observed",
        color="blue",
        alpha=0.4,
        kde=True,
        ax=axes[0, 1],
    )
    sns.histplot(
        df["pred_glm"],
        label="GLM (Gamma-Log) Predicted",
        color="green",
        alpha=0.4,
        kde=True,
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("GLM (Gamma): Observed vs Predicted Charges")
    axes[0, 1].set_xlabel("Hospital Charges ($)")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].legend()

    # (ii) Scatterplot comparing predicted charges to residuals
    # OLS Residuals
    sns.scatterplot(
        x="pred_ols", y="resid_ols", data=df, alpha=0.6, color="red", ax=axes[1, 0]
    )
    axes[1, 0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1, 0].set_title("OLS: Predicted Charges vs Residuals")
    axes[1, 0].set_xlabel("Predicted Charges ($)")
    axes[1, 0].set_ylabel("Residuals ($)")

    # GLM Residuals
    sns.scatterplot(
        x="pred_glm",
        y="resid_glm",
        data=df,
        alpha=0.6,
        color="green",
        ax=axes[1, 1],
    )
    axes[1, 1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1, 1].set_title("GLM (Gamma): Predicted Charges vs Residuals")
    axes[1, 1].set_xlabel("Predicted Charges ($)")
    axes[1, 1].set_ylabel("Residuals ($)")

    plt.tight_layout()
    fig_path = os.path.join("results", "figures", "problem1_model_comparison.png")
    plt.savefig(fig_path, dpi=300)
    plt.show()

    # --- Problem 1(c): Metrics comparison ---
    mae_ols = mean_absolute_error(df["charges"], df["pred_ols"])
    rmse_ols = root_mean_squared_error(df["charges"], df["pred_ols"])

    mae_glm = mean_absolute_error(df["charges"], df["pred_glm"])
    rmse_glm = root_mean_squared_error(df["charges"], df["pred_glm"])

    metrics_df = pd.DataFrame(
        {
            "Model": ["Ordinary Least Squares (OLS)", "GLM (Gamma with Log Link)"],
            "MAE ($)": [mae_ols, mae_glm],
            "RMSE ($)": [rmse_ols, rmse_glm],
        }
    )

    print("\n--- Problem 1 Model Comparison Metrics ---")
    print(metrics_df.to_string(index=False))

    table_path = os.path.join("results", "tables", "problem1_metrics.csv")
    metrics_df.to_csv(table_path, index=False)


if __name__ == "__main__":
    main()