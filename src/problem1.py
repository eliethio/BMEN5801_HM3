import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import gamma, norm, probplot
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

    # =========================================================================
    # Problem 1(a): Distribution & Diagnostic Plots
    # =========================================================================
    fig_a, axes_a = plt.subplots(1, 2, figsize=(14, 5))
    sns.set_theme(style="whitegrid")

    # Plot 1: Empirical Distribution vs Fitted Normal & Gamma PDFs
    sns.histplot(
        df["charges"],
        stat="density",
        bins=30,
        color="skyblue",
        alpha=0.6,
        ax=axes_a[0],
        label="Observed Data",
    )

    x_range = np.linspace(df["charges"].min(), df["charges"].max(), 500)

    # Fit Normal PDF
    mu_norm, std_norm = norm.fit(df["charges"])
    norm_pdf = norm.pdf(x_range, mu_norm, std_norm)
    axes_a[0].plot(
        x_range, norm_pdf, "r--", linewidth=2, label="Fitted Normal PDF"
    )

    # Fit Gamma PDF
    shape_g, loc_g, scale_g = gamma.fit(df["charges"], floc=0)
    gamma_pdf = gamma.pdf(x_range, shape_g, loc_g, scale_g)
    axes_a[0].plot(
        x_range, gamma_pdf, "g-", linewidth=2, label="Fitted Gamma PDF"
    )

    axes_a[0].set_title(
        "1(a) Empirical Density vs Normal & Gamma Fits", fontweight="bold"
    )
    axes_a[0].set_xlabel("Hospital Charges ($)")
    axes_a[0].set_ylabel("Density")
    axes_a[0].legend()

    # Plot 2: Gamma Q-Q Plot (Empirical vs Theoretical Quantiles)
    (osm_g, osr_g), (slope_g, intercept_g, r_g) = probplot(
        df["charges"], dist=gamma(shape_g, loc=loc_g, scale=scale_g)
    )
    axes_a[1].plot(osm_g, osr_g, "o", color="teal", alpha=0.5, label="Data Quantiles")
    axes_a[1].plot(
        osm_g,
        slope_g * np.array(osm_g) + intercept_g,
        "r--",
        linewidth=2,
        label=f"Reference Line ($R^2 = {r_g**2:.3f}$)",
    )

    axes_a[1].set_title(
        "1(a) Gamma Q-Q Plot (Goodness-of-Fit)", fontweight="bold"
    )
    axes_a[1].set_xlabel("Theoretical Gamma Quantiles ($)")
    axes_a[1].set_ylabel("Ordered Observed Charges ($)")
    axes_a[1].legend()

    plt.tight_layout()
    fig_a_path = os.path.join(
        "results", "figures", "problem1a_distribution_justification.png"
    )
    plt.savefig(fig_a_path, dpi=300)
    plt.show()

    # =========================================================================
    # Problem 1(b) & 1(c): Fitting OLS & Gamma GLM and Comparing Results
    # =========================================================================

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
    fig_b, axes_b = plt.subplots(2, 2, figsize=(14, 10))

    # (i) Histogram plot comparing observed vs predicted charges
    # OLS Histogram
    sns.histplot(
        df["charges"],
        label="Observed",
        color="blue",
        alpha=0.4,
        kde=True,
        ax=axes_b[0, 0],
    )
    sns.histplot(
        df["pred_ols"],
        label="OLS Predicted",
        color="red",
        alpha=0.4,
        kde=True,
        ax=axes_b[0, 0],
    )
    axes_b[0, 0].set_title("OLS: Observed vs Predicted Charges")
    axes_b[0, 0].set_xlabel("Hospital Charges ($)")
    axes_b[0, 0].set_ylabel("Count")
    axes_b[0, 0].legend()

    # GLM Histogram
    sns.histplot(
        df["charges"],
        label="Observed",
        color="blue",
        alpha=0.4,
        kde=True,
        ax=axes_b[0, 1],
    )
    sns.histplot(
        df["pred_glm"],
        label="GLM (Gamma-Log) Predicted",
        color="green",
        alpha=0.4,
        kde=True,
        ax=axes_b[0, 1],
    )
    axes_b[0, 1].set_title("GLM (Gamma): Observed vs Predicted Charges")
    axes_b[0, 1].set_xlabel("Hospital Charges ($)")
    axes_b[0, 1].set_ylabel("Count")
    axes_b[0, 1].legend()

    # (ii) Scatterplot comparing predicted charges to residuals
    # OLS Residuals
    sns.scatterplot(
        x="pred_ols", y="resid_ols", data=df, alpha=0.6, color="red", ax=axes_b[1, 0]
    )
    axes_b[1, 0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes_b[1, 0].set_title("OLS: Predicted Charges vs Residuals")
    axes_b[1, 0].set_xlabel("Predicted Charges ($)")
    axes_b[1, 0].set_ylabel("Residuals ($)")

    # GLM Residuals
    sns.scatterplot(
        x="pred_glm",
        y="resid_glm",
        data=df,
        alpha=0.6,
        color="green",
        ax=axes_b[1, 1],
    )
    axes_b[1, 1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes_b[1, 1].set_title("GLM (Gamma): Predicted Charges vs Residuals")
    axes_b[1, 1].set_xlabel("Predicted Charges ($)")
    axes_b[1, 1].set_ylabel("Residuals ($)")

    plt.tight_layout()
    fig_b_path = os.path.join(
        "results", "figures", "problem1_model_comparison.png"
    )
    plt.savefig(fig_b_path, dpi=300)
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