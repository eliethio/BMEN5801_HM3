import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


def targeted_class_split(df, class_percentages, RANDOM_STATE=42):
    """
    Splits dataset by sampling specific percentages or fixed case counts per class.
    Example: 30% of Hernias (18 cases), 12% of Normals (12 cases), 32 Spondylolisthesis cases.
    """
    test_indices = []
    train_indices = []

    for cls, pct in class_percentages.items():
        cls_df = df[df['Class'] == cls]
        if isinstance(pct, float) and pct <= 1.0:
            n_test = int(round(len(cls_df) * pct))
        else:
            n_test = int(pct)  # Fixed integer count

        cls_test_idx = cls_df.sample(n=n_test, random_state=RANDOM_STATE).index
        cls_train_idx = cls_df.index.difference(cls_test_idx)

        test_indices.extend(cls_test_idx)
        train_indices.extend(cls_train_idx)

    train_df = df.loc[train_indices].sample(frac=1, random_state=RANDOM_STATE)
    test_df = df.loc[test_indices].sample(frac=1, random_state=RANDOM_STATE)

    return train_df, test_df


def build_4panel_figure(
    df, 
    dist_cols, 
    model_cols, 
    fig_title_prefix, 
    save_path, 
    split_type='standard',
    class_percentages=None,
    RANDOM_STATE=42
):
    """
    Generates a 4-panel diagnostic figure:
      - Panel A: Feature Distributions (uses dist_cols)
      - Panel B: Feature Medians Radar Plot (uses model_cols without SPL)
      - Panel C: Confusion Matrix (fitted strictly on model_cols)
      - Panel D: Parallel Coordinates of Misclassified Cases (uses model_cols)
    """
    class_order = ['Normal', 'Hernia', 'Spondylolisthesis']
    class_labels = ['N', 'H', 'S']
    color_dict = {'Normal': 'blue', 'Hernia': 'red', 'Spondylolisthesis': 'orange'}

    num_dist_features = len(dist_cols)
    num_model_features = len(model_cols)

    # 1. Pipeline Processing (Split data and fit model strictly on model_cols)
    if split_type == 'targeted' and class_percentages is not None:
        train_df, test_df = targeted_class_split(df, class_percentages, RANDOM_STATE=RANDOM_STATE)
        X_train = train_df[model_cols]
        y_train = train_df['Class']
        X_test = test_df[model_cols]
        y_test = test_df['Class']
    else:
        X = df[model_cols]
        y = df['Class']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred, labels=class_order)

    # 2. Figure Layout Setup
    fig = plt.figure(figsize=(2 * num_dist_features, 7))
    gs = fig.add_gridspec(2, num_dist_features, height_ratios=[1.1, 1], hspace=0.4, wspace=0.35)

    # --- Part A: Feature Distributions ---
    for i, col in enumerate(dist_cols):
        ax_a = fig.add_subplot(gs[0, i])
        for cls in class_order:
            sns.kdeplot(
                df[df['Class'] == cls][col], 
                ax=ax_a, 
                color=color_dict[cls], 
                fill=True, 
                alpha=0.35, 
                linewidth=1.5
            )
        ax_a.set_title(col, rotation=90, x=-0.15, y=0.5, va='center', ha='right', fontsize=11, fontweight='bold')
        ax_a.set_xlabel('')
        ax_a.set_ylabel('')
        ax_a.set_xticks([])
        ax_a.set_yticks([])
        sns.despine(ax=ax_a, left=True, bottom=False)
        ax_a.spines['bottom'].set_color('black')
        if i == 0:
            ax_a.text(0.35, 0.90, 'Normal', color='blue', transform=ax_a.transAxes, fontsize=9, fontweight='bold')
            ax_a.text(0.35, 0.82, 'Hernia', color='red', transform=ax_a.transAxes, fontsize=9, fontweight='bold')
            ax_a.text(0.35, 0.74, 'SPL', color='orange', transform=ax_a.transAxes, fontsize=9, fontweight='bold')

    fig.text(0.40, 0.95, f'Feature Distributions ({fig_title_prefix})', fontsize=14, fontweight='bold')
    fig.text(0.02, 0.94, 'A', fontsize=16, fontweight='bold')

    # --- Part B: Feature Medians (Radar Chart) ---
    fig.text(0.02, 0.46, 'B', fontsize=16, fontweight='bold')
    fig.text(0.06, 0.46, 'Feature Medians', fontsize=14, fontweight='bold')

    radar_cols = [c for c in model_cols if c != 'SPL']
    num_vars = len(radar_cols)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist() + [0]

    left_span = max(2, num_dist_features // 3)
    mid_span = max(2, num_dist_features // 3)

    ax_b = fig.add_subplot(gs[1, 0:left_span], polar=True)
    for cls in class_order:
        medians = df[df['Class'] == cls][radar_cols].median().values.tolist()
        medians += [medians[0]]
        ax_b.plot(angles, medians, color=color_dict[cls], linewidth=2.0)
    ax_b.set_xticks(angles[:-1])
    ax_b.set_xticklabels(radar_cols, fontsize=9, fontweight='bold')

    # --- Part C: Confusion Matrix ---
    fig.text(0.36, 0.46, 'C', fontsize=16, fontweight='bold')
    fig.text(0.40, 0.46, f'Logistic Regression ({num_model_features} Feat.)', fontsize=14, fontweight='bold')

    c_end = min(num_dist_features - 2, left_span + mid_span)
    ax_c = fig.add_subplot(gs[1, left_span:c_end])
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        cbar=False,
        xticklabels=class_labels, 
        yticklabels=class_labels, 
        ax=ax_c, 
        annot_kws={"size": 11, "weight": "bold"}
    )
    ax_c.set_xlabel('Predicted', fontsize=10, fontweight='bold')
    ax_c.set_ylabel('Actual', fontsize=10, fontweight='bold')

    # --- Part D: Parallel Coordinates (Misclassified Cases) ---
    fig.text(0.68, 0.46, 'D', fontsize=16, fontweight='bold')
    fig.text(0.72, 0.46, 'Misclassified Cases', fontsize=14, fontweight='bold')
    ax_d = fig.add_subplot(gs[1, c_end:num_dist_features])

    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=model_cols)
    radar_scaled = X_test_scaled_df[radar_cols]
    is_misclassified = (y_test.values != y_pred)

    for idx in range(len(radar_scaled)):
        if not is_misclassified[idx]:
            ax_d.plot(range(num_vars), radar_scaled.iloc[idx], color='lightgray', alpha=0.3, linewidth=0.8)

    for idx in range(len(radar_scaled)):
        if is_misclassified[idx]:
            actual_cls = y_test.values[idx]
            ax_d.plot(
                range(num_vars), 
                radar_scaled.iloc[idx], 
                color=color_dict[actual_cls], 
                marker='o', 
                linewidth=1.5, 
                markersize=5
            )

    ax_d.set_xticks(range(num_vars))
    ax_d.set_xticklabels(radar_cols, fontsize=9, fontweight='bold')
    ax_d.set_xlabel('Features', fontsize=10, fontweight='bold')
    ax_d.set_ylabel('Feature Value', fontsize=10, fontweight='bold')
    sns.despine(ax=ax_d)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    print(f"✓ Saved figure to: {save_path}\n", flush=True)


def main():
    plt.ion()

    RANDOM_STATE = 42
    np.random.seed(RANDOM_STATE)

    possible_paths = [
        os.path.join("data", "vertebralcolumn.csv"),
        "vertebralcolumn.csv",
        "../data/vertebralcolumn.csv"
    ]

    data_path = None
    for p in possible_paths:
        if os.path.exists(p):
            data_path = p
            break

    if data_path is None:
        raise FileNotFoundError(
            "Could not find 'vertebralcolumn.csv'. Please upload or place the CSV in the directory."
        )

    df = pd.read_csv(data_path)

    all_cols = ['PI', 'PT', 'LLA', 'SS', 'PR', 'SPL']
    cols_5 = ['PI', 'PT', 'LLA', 'SS', 'PR']

    if len(df.columns) == 7:
        df.columns = all_cols + ['Class']
    elif 'class' in df.columns:
        df.rename(columns={'class': 'Class'}, inplace=True)

    target_percentages = {
        'Hernia': 0.30,            # 30% of 60 = 18 cases
        'Normal': 0.12,            # 12% of 100 = 12 cases
        'Spondylolisthesis': 32    # Fixed 32 cases out of 150
    }

    # Case 1: Full 6-Feature Model (Standard Split: 18 N / 12 H / 32 S)
    print("Generating Case 1 (Full 6 Features - Standard Split)...", flush=True)
    build_4panel_figure(
        df=df, 
        dist_cols=all_cols,
        model_cols=all_cols, 
        fig_title_prefix="All 6 Features (Standard 18N/12H/32S)", 
        save_path="results/figures/problem4_case1_6features_standard.png",
        split_type='standard',
        RANDOM_STATE=RANDOM_STATE
    )

    # Case 2: 5-Feature Model (Standard Split: 18 N / 12 H / 32 S)
    print("Generating Case 2 (5-Feature Model - Standard Split)...", flush=True)
    build_4panel_figure(
        df=df, 
        dist_cols=all_cols,
        model_cols=cols_5, 
        fig_title_prefix="5 Pelvic Features (Standard 18N/12H/32S)", 
        save_path="results/figures/problem4_case2_5features_standard.png",
        split_type='standard',
        RANDOM_STATE=RANDOM_STATE
    )

    # Case 3: 5-Feature Model (Targeted Baseline Split: 18 H / 12 N / 32 S)
    print("Generating Case 3 (5-Feature Model - Targeted Baseline Split: 18H/12N/32S)...", flush=True)
    build_4panel_figure(
        df=df, 
        dist_cols=all_cols,
        model_cols=cols_5, 
        fig_title_prefix="5 Pelvic Features (Targeted 18H/12N/32S Baseline)", 
        save_path="results/figures/problem4_case3_5features_targeted_baseline.png",
        split_type='targeted',
        class_percentages=target_percentages,
        RANDOM_STATE=RANDOM_STATE
    )

    # Case 4: Full 6-Feature Model (Targeted Baseline Split: 18 H / 12 N / 32 S)
    print("Generating Case 4 (Full 6 Features - Targeted Baseline Split: 18H/12N/32S)...", flush=True)
    build_4panel_figure(
        df=df, 
        dist_cols=all_cols,
        model_cols=all_cols, 
        fig_title_prefix="All 6 Features (Targeted 18H/12N/32S Baseline)", 
        save_path="results/figures/problem4_case4_6features_targeted_baseline.png",
        split_type='targeted',
        class_percentages=target_percentages,
        RANDOM_STATE=RANDOM_STATE
    )


if __name__ == "__main__":
    main()