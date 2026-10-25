import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

def main():
    RANDOM_STATE = 42
    np.random.seed(RANDOM_STATE)

    # -------------------------------------------------------------------------
    # 1. Load Data
    # -------------------------------------------------------------------------
    data_path = os.path.join("data", "vertebralcolumn.csv")
    df = pd.read_csv(data_path if os.path.exists(data_path) else "vertebralcolumn.csv")

    feature_cols = ['PI', 'PT', 'LLA', 'SS', 'PR', 'SPL']
    if len(df.columns) == 7:
        df.columns = feature_cols + ['Class']

    class_order = ['Normal', 'Hernia', 'Spondylolisthesis']
    class_labels = ['N', 'H', 'S']
    color_dict = {'Normal': 'blue', 'Hernia': 'red', 'Spondylolisthesis': 'orange'}

    # -------------------------------------------------------------------------
    # 2. Pipeline Execution
    # -------------------------------------------------------------------------
    X = df[feature_cols]
    y = df['Class']

    # Unstratified split with shuffle seed targeting 12 Normal / 18 Hernia / 32 SPL
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

    # -------------------------------------------------------------------------
    # 3. Visualization Construction
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.1, 1], hspace=0.4, wspace=0.35)

    # --- Part A: Feature Distributions ---
    feature_titles = [
        'Pelvic Incidence', 'Pelvic Tilt', 'Lumbar Lordosis Angle',
        'Sacral Slope', 'Pelvic Radius', 'Degree of SPL'
    ]
    for i, (col, title) in enumerate(zip(feature_cols, feature_titles)):
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
        ax_a.set_title(title, rotation=90, x=-0.15, y=0.5, va='center', ha='right', fontsize=11)
        ax_a.set_xlabel(''); ax_a.set_ylabel(''); ax_a.set_xticks([]); ax_a.set_yticks([])
        sns.despine(ax=ax_a, left=True, bottom=False)
        ax_a.spines['bottom'].set_color('black')
        
        if i == 0:
            ax_a.text(0.4, 0.90, 'Normal', color='blue', transform=ax_a.transAxes, fontsize=9, fontweight='bold')
            ax_a.text(0.4, 0.82, 'Hernia', color='red', transform=ax_a.transAxes, fontsize=9, fontweight='bold')
            ax_a.text(0.4, 0.74, 'SPL', color='orange', transform=ax_a.transAxes, fontsize=9, fontweight='bold')

    fig.text(0.38, 0.95, 'Feature Distributions', fontsize=14, fontweight='bold')
    fig.text(0.04, 0.94, 'A', fontsize=16, fontweight='bold')

    # --- Part B: Feature Medians Radar Chart ---
    fig.text(0.04, 0.46, 'B', fontsize=16, fontweight='bold')
    fig.text(0.08, 0.46, 'Feature Medians', fontsize=14, fontweight='bold')
    radar_cols = ['PI', 'PT', 'LLA', 'SS', 'PR']
    num_vars = len(radar_cols)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist() + [0]

    ax_b = fig.add_subplot(gs[1, 0:2], polar=True)
    for cls in class_order:
        medians = df[df['Class'] == cls][radar_cols].median().values.tolist()
        medians += medians[:1]
        ax_b.plot(angles, medians, color=color_dict[cls], linewidth=2.0)
    ax_b.set_xticks(angles[:-1])
    ax_b.set_xticklabels(radar_cols, fontsize=9, fontweight='bold')
    ax_b.set_rlabel_position(20)

    # --- Part C: Logistic Regression Confusion Matrix ---
    fig.text(0.36, 0.46, 'C', fontsize=16, fontweight='bold')
    fig.text(0.40, 0.46, 'Logistic Regression', fontsize=14, fontweight='bold')
    ax_c = fig.add_subplot(gs[1, 2:4])
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        cbar=False,
        xticklabels=class_labels, 
        yticklabels=class_labels, 
        ax=ax_c, 
        annot_kws={"size": 11}
    )
    ax_c.set_xlabel('Predicted', fontsize=10)
    ax_c.set_ylabel('Actual', fontsize=10)

    # --- Part D: Parallel Coordinates (Misclassified Cases) ---
    fig.text(0.69, 0.46, 'D', fontsize=16, fontweight='bold')
    fig.text(0.73, 0.46, 'Misclassified Cases', fontsize=14, fontweight='bold')
    ax_d = fig.add_subplot(gs[1, 4:6])

    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_cols)
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
    ax_d.set_xlabel('Features', fontsize=10)
    ax_d.set_ylabel('Feature Value', fontsize=10)
    sns.despine(ax=ax_d)

    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/problem4_recreated_visualization.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()