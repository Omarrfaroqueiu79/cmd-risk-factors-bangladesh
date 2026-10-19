# =============================================================================
# COMMON MENTAL DISORDER (CMD) ANALYSIS PIPELINE
# Statistical and Machine Learning Analysis Workflow for BMC Public Health
#
# Study: Prevalence and Risk Factors of Common Mental Disorders (CMD) Among
#        University Students in Bangladesh: A Cross-Sectional Study
#
# Main Output Figures (saved to the "BMC Public Health" directory):
#   Fig1.tif  - Overall prevalence of Common Mental Disorders (CMD)
#   Fig2.tif  - Distribution of CMD prevalence across daily digital device use
#   Fig3.tif  - Receiver Operating Characteristic (ROC) curve comparison
#   Fig4.tif  - Domain-level dominance analysis
#   Fig5A.tif - SHAP beeswarm summary distribution (Gradient Boosting)
#   Fig5B.tif - Global feature importance ranking (Gradient Boosting)
#   Fig5.tif  - Combined SHAP beeswarm and global feature importance panel
#   Fig6.tif  - SHAP dependence plot for top predictive feature
#   Fig7.tif  - Individual-level SHAP explanation (waterfall plot)
#   S*.tif    - Supplementary confusion matrices and performance evaluations
#
# Tabular outputs (CSV, XLSX, TXT) are exported to the primary output directory.
# =============================================================================


# =============================================================================
# 0. COMPUTATIONAL ENVIRONMENT CONFIGURATION
# =============================================================================
# Standardize OpenMP runtime and thread execution parameters to ensure
# deterministic execution across multi-framework estimators (XGBoost, CatBoost, scikit-learn).

import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["MKL_NUM_THREADS"]      = "1"

# Set matplotlib backend to Agg for headless non-interactive rendering
import matplotlib
matplotlib.use("Agg")

def _log(msg):
    print(msg, flush=True)

_log("[env] Computational backend and threading parameters initialized")


# =============================================================================
# 1. IMPORTS
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import chi2, chi2_contingency

import statsmodels.api as sm
from statsmodels.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.contingency_tables import Table

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, StackingClassifier,
)
from sklearn.inspection import permutation_importance
from sklearn.utils import resample

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE

import shap
import pingouin as pg

from PIL import Image


# =============================================================================
# 2. PATHS AND FIGURE SPECIFICATIONS (BMC PUBLIC HEALTH GUIDELINES)
# =============================================================================

FILE_PATH = r"G:\Documents\Book\1\4th Year\4th\1822021 (Project)\Trial\Data_Final.sav"

OUTPUT_DIR = r"G:\Documents\Book\1\4th Year\4th\1822021 (Project)\Trial\Final_Output\Combining_All"
BMC_DIR    = os.path.join(OUTPUT_DIR, "BMC Public Health")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(BMC_DIR, exist_ok=True)

# ---- Global graphical style configuration ----
# Typography: Arial (standard sans-serif) with fallback to DejaVu Sans
from matplotlib import font_manager
_available_fonts = {f.name for f in font_manager.fontManager.ttflist}
_preferred_font = "Arial" if "Arial" in _available_fonts else "DejaVu Sans"
_log(f"[fig] font selected: {_preferred_font}")

mpl.rcParams.update({
    "font.family":       _preferred_font,
    "font.size":         10,        # Standard journal text size (8-12 pt)
    "axes.labelsize":    10,
    "axes.titlesize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "savefig.dpi":       300,
    "figure.dpi":        300,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
})

BMC_WIDTH_SINGLE = 3.35   # Single-column width (85 mm in inches)
BMC_WIDTH_FULL   = 6.69   # Double-column width (170 mm in inches)
BMC_MAX_HEIGHT   = 8.86   # Maximum page height (225 mm in inches)

# Dimension aliases maintained for cross-script compatibility
PLOS_WIDTH_SINGLE = BMC_WIDTH_SINGLE
PLOS_WIDTH_FULL   = BMC_WIDTH_FULL
PLOS_MAX_HEIGHT   = BMC_MAX_HEIGHT
PLOS_DIR          = BMC_DIR


def save_bmc_tiff(fig, filename, dpi=300):
    """
    Export matplotlib figure as a high-resolution TIFF conforming to journal specifications
    (300 dpi, 24-bit RGB, Deflate lossless compression).
    """
    stem = os.path.splitext(filename)[0]
    png_path  = os.path.join(BMC_DIR, stem + ".png")
    tiff_path = os.path.join(BMC_DIR, stem + ".tif")

    # Step 1: Render raster image buffer via matplotlib Agg backend
    try:
        fig.savefig(png_path, format="png", dpi=dpi, bbox_inches="tight")
    except Exception as e:
        _log(f"[save] PNG write failed for {filename}: {e}")
        try:
            plt.close(fig)
        except Exception:
            pass
        return
    finally:
        try:
            plt.close(fig)
        except Exception:
            pass

    # Step 2: Convert buffer to 300 dpi RGB TIFF using lossless Deflate compression
    try:
        import tifffile
        with Image.open(png_path) as im:
            im = im.convert("RGB")
            arr = np.array(im)
        tifffile.imwrite(tiff_path, arr, resolution=(float(dpi), float(dpi)), compression="deflate")
        os.remove(png_path)
        _log(f"[save] Saved: {tiff_path} (compressed via tifffile deflate)")
    except Exception as e:
        _log(f"[save] Primary compression fallback ({e}), exporting standard TIFF...")
        try:
            with Image.open(png_path) as im:
                im = im.convert("RGB")
                im.save(tiff_path, format="TIFF", dpi=(dpi, dpi))
            os.remove(png_path)
            _log(f"[save] Saved: {tiff_path}")
        except Exception as e2:
            _log(f"[save] TIFF export failed for {filename}: {e2}")


# Alias maintained for cross-pipeline compatibility
save_plos_tiff = save_bmc_tiff


# =============================================================================
# 3. LOAD DATA
# =============================================================================

_log("[load] reading SPSS file...")
df = pd.read_spss(FILE_PATH)
_log(f"[load] Dataset loaded: {df.shape}")

TARGET = "CMD"


# =============================================================================
# 4. DESCRIPTIVE + BIVARIATE ANALYSIS (Table 1)
# =============================================================================

excluded_variables = ["CMD", "CMD_Severity"]
analysis_variables = [c for c in df.columns if c not in excluded_variables]

category_order = {
    "Gender": ["Male", "Female"],
    "Age_in_years": ["19-25", "25+"],
    "University_name": [
        "Islamic University", "HSTU", "University of Dhaka",
        "SUST", "Cumilla University", "PUST", "JUST",
    ],
    "Education_Level": [
        "First Year", "Second Year", "Third Year", "Fourth Year", "Master's",
    ],
    "Resident": ["Home", "Hall", "Mess"],
    "Smoking_Status": ["Never smoked", "Current smoker", "Former smoker"],
    "Family_Type": ["Single Family", "Joint Family"],
    "Internet_mode": ["Mobile data connection", "Broadband connection"],
    "Average_DD_Use_Time": ["<2", "2-4", "4-6", ">6"],
    "Do_Multitask": ["Rarely", "Often", "Sometimes", "Always"],
    "Excessive_ST_affect_AP": ["Yes, negatively", "Yes, positively", "No impact"],
    "Feel_tired_after_Use_DD": ["Rarely", "Sometimes", "Often", "Always"],
    "Experienced_ES_for_ST": ["No", "Yes"],
    "Experienced_Headaches_for_ST": ["No", "Yes"],
    "Experienced_NP_for_ST": ["No", "Yes"],
    "Experienced_BP_for_ST": ["No", "Yes"],
    "Experienced_HD_for_ST": ["No", "Yes"],
    "Physical_activity_level": [
        "Significantly decreased", "Somewhat decreased", "Remained the same",
    ],
    "Decline_Overall_Happyness": [
        "Yes, significantly", "Yes, somewhat", "No noticeable change",
    ],
    "Difficulty_in_Offline_Activity": [
        "Yes, frequently", "Yes, occasionally", "No",
    ],
    "Impact_Learning_Ability": [
        "Yes, negatively", "Yes, positively", "No impact",
    ],
    "Overall_Health_Worsening": ["No", "Yes"],
    "Sleep_amount_impact": ["Yes", "No", "Not sure"],
    "Varsity_hr_tiredness": ["Never", "Sometimes", "Often", "Always"],
}

variable_labels = {
    "Gender": "Gender",
    "Age_in_years": "Age in years",
    "University_name": "Name of university",
    "Education_Level": "Education level",
    "Resident": "Resident",
    "Smoking_Status": "Smoking status",
    "Family_Type": "Family type",
    "Internet_mode": "Mode of internet connection",
    "Average_DD_Use_Time": "Average daily digital device use time (hours)",
    "Do_Multitask": "Frequency of multitasking",
    "Excessive_ST_affect_AP": "Excessive screen time affects academic performance",
    "Feel_tired_after_Use_DD": "Feels tired after using digital devices",
    "Experienced_ES_for_ST": "Experienced eye strain due to screen time",
    "Experienced_Headaches_for_ST": "Experienced headaches due to screen time",
    "Experienced_NP_for_ST": "Experienced neck pain due to screen time",
    "Experienced_BP_for_ST": "Experienced back pain due to screen time",
    "Experienced_HD_for_ST": "Experienced hand discomfort due to screen time",
    "Physical_activity_level": "Physical activity level",
    "Decline_Overall_Happyness": "Decline in overall happiness",
    "Difficulty_in_Offline_Activity": "Difficulty engaging in offline activities",
    "Impact_Learning_Ability": "Impact on memory or learning ability",
    "Overall_Health_Worsening": "Mental health has worsened due to screen time",
    "Sleep_amount_impact": "Screen use affects sleep",
    "Varsity_hr_tiredness": "Tiredness during university hours",
}

descriptive_results = []
bivariate_results = []

for variable in analysis_variables:
    try:
        s = df[variable].copy()
        if variable in category_order:
            s = pd.Categorical(s, categories=category_order[variable], ordered=True)

        # ---- Descriptive ----
        freq = pd.Series(s).value_counts(sort=False, dropna=False)
        pct = (freq / len(df)) * 100
        for cat in freq.index:
            if pd.isna(cat):
                continue
            descriptive_results.append({
                "Variable":   variable_labels.get(variable, variable),
                "Category":   str(cat),
                "Frequency":  int(freq[cat]),
                "Percentage": round(pct[cat], 2),
            })

        # ---- Bivariate chi-square ----
        ctab = pd.crosstab(s, df[TARGET], dropna=False)
        ctab = ctab.loc[ctab.sum(axis=1) > 0]
        if ctab.shape[0] < 2:
            continue
        chi2_stat, p, dof, _ = chi2_contingency(ctab)
        row_pct = pd.crosstab(s, df[TARGET], normalize="index", dropna=False) * 100
        row_pct = row_pct.loc[ctab.index]

        first = True
        for cat in ctab.index:
            no_count = ctab.loc[cat].iloc[0]
            yes_count = ctab.loc[cat].iloc[1]
            no_pct = row_pct.loc[cat].iloc[0]
            yes_pct = row_pct.loc[cat].iloc[1]
            bivariate_results.append({
                "Variable": variable_labels.get(variable, variable) if first else "",
                "Category": str(cat),
                "No CMD N (%)":  f"{no_count} ({no_pct:.1f})",
                "CMD Present N (%)": f"{yes_count} ({yes_pct:.1f})",
                "Chi-Square": round(chi2_stat, 3) if first else "",
                "df": dof if first else "",
                "P-value": ("<0.001" if p < 0.001 else f"{p:.3f}") if first else "",
            })
            first = False
    except Exception as e:
        print(f"Skipping {variable}: {e}")

descriptive_df = pd.DataFrame(descriptive_results)
bivariate_df   = pd.DataFrame(bivariate_results)

descriptive_df.to_csv(os.path.join(OUTPUT_DIR, "Descriptive_Analysis.csv"), index=False, encoding="utf-8-sig")
bivariate_df.to_csv(os.path.join(OUTPUT_DIR, "Bivariate_Analysis.csv"),   index=False, encoding="utf-8-sig")
descriptive_df.to_excel(os.path.join(OUTPUT_DIR, "Descriptive_Analysis.xlsx"), index=False)
bivariate_df.to_excel(os.path.join(OUTPUT_DIR, "Bivariate_Analysis.xlsx"),   index=False)
_log("[sec4] Descriptive and bivariate analyses exported.")


# =============================================================================
# 5. PREDICTOR SELECTION AND ENCODING
# =============================================================================

predictors = [
    # Sociodemographic
    "University_name", "Family_Type",
    # Digital behaviour
    "Internet_mode", "Average_DD_Use_Time", "Do_Multitask",
    # Screen-related symptoms
    "Feel_tired_after_Use_DD",
    "Experienced_ES_for_ST", "Experienced_Headaches_for_ST",
    "Experienced_NP_for_ST", "Experienced_BP_for_ST", "Experienced_HD_for_ST",
    # Physical health
    "Physical_activity_level", "Sleep_amount_impact", "Varsity_hr_tiredness",
    # Psychosocial
    "Excessive_ST_affect_AP", "Decline_Overall_Happyness",
    "Difficulty_in_Offline_Activity", "Impact_Learning_Ability",
    "Overall_Health_Worsening",
]

analysis_df = df[predictors + [TARGET]].astype(str).replace("nan", "Missing").fillna("Missing")

X = analysis_df.drop(columns=[TARGET])
y = analysis_df[TARGET].astype(str).str.strip()

target_mapping = {"No CMD": 0, "CMD Present": 1}
y = y.map(target_mapping)

valid = y.notna()
X = X.loc[valid]
y = y.loc[valid].astype(int)
_log(f"[sec5] Final analytic sample: n = {len(y)}")

X = pd.get_dummies(X, drop_first=True)
X.columns = [
    c.replace("<", "less_").replace(">", "greater_")
     .replace(" ", "_").replace("-", "_").replace("/", "_")
     .replace("[", "").replace("]", "").replace("(", "").replace(")", "")
    for c in X.columns
]
X = X.astype(np.float32)


# =============================================================================
# 6. TRAIN / TEST SPLIT + SCALING + SMOTE
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y,
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
_log(f"[sec6] SMOTE-balanced training set: {X_train_smote.shape}")


# =============================================================================
# 7. HYPERPARAMETER TUNING (RandomizedSearchCV)
# =============================================================================

cv_tune = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Hyperparameter optimization: Standardized randomized search across estimators 
# with stratified 5-fold cross-validation.

# ---- Random Forest ----
rf_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=1),
    param_distributions={
        "n_estimators": [200, 300, 500, 800],
        "max_depth": [3, 5, 7, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    },
    n_iter=15, scoring="roc_auc", cv=cv_tune,
    n_jobs=1, random_state=42,
).fit(X_train_smote, y_train_smote)
best_rf = rf_search.best_estimator_

# ---- Gradient Boosting ----
gb_search = RandomizedSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_distributions={
        "n_estimators": [100, 200, 300, 500],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "max_depth": [2, 3, 4, 5],
        "subsample": [0.7, 0.8, 0.9, 1.0],
    },
    n_iter=15, scoring="roc_auc", cv=cv_tune,
    n_jobs=1, random_state=42,
).fit(X_train_smote, y_train_smote)
best_gb = gb_search.best_estimator_

# ---- XGBoost ----
xgb_search = RandomizedSearchCV(
    XGBClassifier(
        objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", n_jobs=1, random_state=42,
    ),
    param_distributions={
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "gamma": [0, 0.1, 0.3],
        "reg_alpha": [0, 0.5, 1],
        "reg_lambda": [1, 2, 5],
    },
    n_iter=15, scoring="roc_auc", cv=cv_tune,
    n_jobs=1, random_state=42,
).fit(X_train_smote, y_train_smote)
best_xgb = xgb_search.best_estimator_

# ---- CatBoost ----
cat_search = RandomizedSearchCV(
    CatBoostClassifier(verbose=0, random_state=42, thread_count=1),
    param_distributions={
        "iterations": [200, 300, 500],
        "depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "l2_leaf_reg": [1, 3, 5, 7],
    },
    n_iter=15, scoring="roc_auc", cv=cv_tune,
    n_jobs=1, random_state=42,
).fit(X_train_smote, y_train_smote)
best_cat = cat_search.best_estimator_

# ---- Stacking ensemble ----
stacking_model = StackingClassifier(
    estimators=[
        ("rf",  best_rf),
        ("gb",  best_gb),
        ("xgb", best_xgb),
        ("cat", best_cat),
    ],
    final_estimator=LogisticRegression(max_iter=5000),
    cv=5,
    n_jobs=1,
)

# ---- Optional: cache the tuned base learners so reruns skip tuning ----
import joblib
joblib.dump(best_rf,  os.path.join(OUTPUT_DIR, "best_rf.pkl"))
joblib.dump(best_gb,  os.path.join(OUTPUT_DIR, "best_gb.pkl"))
joblib.dump(best_xgb, os.path.join(OUTPUT_DIR, "best_xgb.pkl"))
joblib.dump(best_cat, os.path.join(OUTPUT_DIR, "best_cat.pkl"))

models = {
    "LR":       LogisticRegression(max_iter=5000),
    "RF":       best_rf,
    "GB":       best_gb,
    "XGB":      best_xgb,
    "CatBoost": best_cat,
    "Stacking": stacking_model,
}


# =============================================================================
# 8. TRAIN, EVALUATE, AND BUILD ROC PLOT (Fig 3)
# =============================================================================

evaluation_results = []
confusion_results  = []

fig_roc, ax_roc = plt.subplots(figsize=(PLOS_WIDTH_SINGLE, PLOS_WIDTH_SINGLE))
cv_eval = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    _log(f"[sec8] === Model: {name} ===")
    _log(f"[sec8] {name}: fit...")
    model.fit(X_train_smote, y_train_smote)
    _log(f"[sec8] {name}: fit OK")

    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_prob > 0.45).astype(int)

    accuracy    = accuracy_score(y_test, y_pred)
    precision   = precision_score(y_test, y_pred)
    recall      = recall_score(y_test, y_pred)
    f1          = f1_score(y_test, y_pred)
    auroc       = roc_auc_score(y_test, y_prob)
    cm          = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # ---- Bootstrap 95% CI for AUROC (500 iterations is enough for CI ~0.001 precision) ----
    rng = np.random.RandomState(42)
    boots = []
    for _ in range(500):
        idx = resample(np.arange(len(y_test)), replace=True, random_state=rng)
        if len(np.unique(y_test.iloc[idx])) < 2:
            continue
        boots.append(roc_auc_score(y_test.iloc[idx], y_prob[idx]))
    boots = np.sort(np.array(boots))
    ci_lower = boots[int(0.025 * len(boots))]
    ci_upper = boots[int(0.975 * len(boots))]

    # Cross-validated AUROC evaluation; for the stacking ensemble (which already employs
    # internal 5-fold cross-validation), non-parametric bootstrap resampling provides
    # the 95% confidence intervals.
    if name == "Stacking":
        cv_score = np.nan
    else:
        cv_score = cross_val_score(
            model, X_train_smote, y_train_smote,
            cv=cv_eval, scoring="roc_auc", n_jobs=1,
        ).mean()

    evaluation_results.append({
        "Model": name,
        "Accuracy":     round(accuracy, 3),
        "Precision":    round(precision, 3),
        "Recall":       round(recall, 3),
        "Specificity":  round(specificity, 3),
        "F1_Score":     round(f1, 3),
        "AUROC":        round(auroc, 3),
        "AUROC (95% CI)": f"{auroc:.3f} ({ci_lower:.3f}, {ci_upper:.3f})",
        "CV_AUROC":     (round(cv_score, 3) if not np.isnan(cv_score) else "N/A"),
    })
    confusion_results.append({
        "Model": name,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    })

    # ---- Save per-model confusion matrix as supporting TIFF ----
    _log(f"[sec8] {name}: confusion matrix figure...")
    fig_cm, ax_cm = plt.subplots(figsize=(3.5, 3.5))
    ConfusionMatrixDisplay(cm, display_labels=["No CMD", "CMD"]).plot(
        ax=ax_cm, colorbar=False, cmap="Blues",
    )
    ax_cm.grid(False)
    save_plos_tiff(fig_cm, f"S_Confusion_{name}.tif")
    _log(f"[sec8] {name}: done")

    # ---- Add ROC curve to combined Fig 3 ----
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax_roc.plot(fpr, tpr, linewidth=1.5, label=f"{name} (AUC={auroc:.3f})")

# ---- Finalize Fig 3: ROC curve comparison ----
ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=0.8)
ax_roc.set_xlabel("False positive rate")
ax_roc.set_ylabel("True positive rate")
ax_roc.set_xlim(0, 1)
ax_roc.set_ylim(0, 1.02)
ax_roc.legend(frameon=False, loc="lower right", fontsize=8)
ax_roc.spines["top"].set_visible(False)
ax_roc.spines["right"].set_visible(False)
plt.tight_layout()
save_plos_tiff(fig_roc, "Fig3.tif")

evaluation_df = pd.DataFrame(evaluation_results)
confusion_df  = pd.DataFrame(confusion_results)
evaluation_df.to_csv(os.path.join(OUTPUT_DIR, "Model_Evaluation.csv"), index=False)
confusion_df.to_csv(os.path.join(OUTPUT_DIR, "Confusion_Matrix.csv"), index=False)
with pd.ExcelWriter(os.path.join(OUTPUT_DIR, "Complete_CMD_Results.xlsx")) as w:
    evaluation_df.to_excel(w, sheet_name="Model_Evaluation", index=False)
    confusion_df.to_excel(w, sheet_name="Confusion_Matrix",  index=False)

print("\nModel evaluation:")
print(evaluation_df)


# =============================================================================
# 9. MODEL PERFORMANCE MULTI-PANEL (Supplementary)
# =============================================================================

metrics = ["Accuracy", "Precision", "Recall", "Specificity", "F1_Score", "AUROC"]
fig, axes = plt.subplots(2, 3, figsize=(PLOS_WIDTH_FULL, 5.5))
axes = axes.flatten()
palette = sns.color_palette("Blues_d", len(evaluation_df))

for i, metric in enumerate(metrics):
    ax = axes[i]
    bars = ax.bar(evaluation_df["Model"], evaluation_df[metric],
                  color=palette, edgecolor="black", linewidth=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    ax.tick_params(axis="x", rotation=35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.02, f"{h:.2f}",
                ha="center", fontsize=8)
    ax.set_title(metric.replace("_", " "), fontweight="normal")
plt.tight_layout()
save_plos_tiff(fig, "S_Model_Performance_Panel.tif")


# =============================================================================
# 10. EXPLAINABLE AI - SHAP ON GRADIENT BOOSTING (BEST MODEL)
# =============================================================================

print("\nRunning SHAP analysis on Gradient Boosting...")

import matplotlib.collections as mcoll
from matplotlib.colors import LinearSegmentedColormap

cmap_custom = LinearSegmentedColormap.from_list("blue_purple_red", ["#0000FF", "#800080", "#FF0000"])

gb_model = models["GB"]

X_test_df = pd.DataFrame(X_test, columns=X.columns).copy()
X_test_df.columns = [c.replace("_", " ").strip() for c in X_test_df.columns]

# Rename columns to publication-ready short names
rename_dict = {
    "Average DD Use Time greater 6": "Average DD Use Time > 6",
    "Average DD Use Time less 2": "Average DD Use Time < 2",
    "Average DD Use Time 2 4": "Average DD Use Time 2 4",
    "Average DD Use Time 4 6": "Average DD Use Time 4 6"
}
X_test_df.columns = [rename_dict.get(c, c) for c in X_test_df.columns]

# Scale feature matrix with standardized column identifiers for SHAP calculations
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test_df.columns)

explainer = shap.TreeExplainer(gb_model)
shap_values = explainer.shap_values(X_test_scaled_df)
if isinstance(shap_values, list):
    shap_values = shap_values[1]

# ---- SHAP importance table ----
shap_importance = (
    pd.DataFrame({
        "Feature": X_test_df.columns,
        "Mean_abs_SHAP": np.abs(shap_values).mean(axis=0),
    })
    .sort_values("Mean_abs_SHAP", ascending=False)
    .reset_index(drop=True)
)
shap_importance.to_csv(os.path.join(OUTPUT_DIR, "SHAP_Feature_Importance.csv"), index=False)

# ---- Fig 5A: SHAP beeswarm summary distribution (top 20) ----
fig = plt.figure(figsize=(6.4, 8.0))
shap.summary_plot(shap_values, X_test_scaled_df, max_display=20, show=False, plot_size=None, cmap=cmap_custom, feature_names=list(X_test_scaled_df.columns))

# Extract plot axes and colorbar references
ax1 = fig.axes[0]
cb_ax = None
for ax in fig.axes:
    if ax.get_ylabel() == "Feature value" or ax.get_label() == "<colorbar>":
        cb_ax = ax
        break

# Reconstruct colorbar with standardized vertical orientation
if cb_ax is not None:
    cb_ax.remove()

# Configure subplot dimensions and margins
ax1.set_position([0.18, 0.12, 0.50, 0.80])

# Configure scatter point attributes
for collection in ax1.collections:
    if isinstance(collection, mcoll.PathCollection):
        collection.set_sizes([10])
        collection.set_alpha(1.0)
ax1.tick_params(labelsize=8)
ax1.set_xticks([-1.0, 0.0, 1.0])
ax1.set_xticklabels(["-1", "0", "1"])
ax1.set_xlabel("SHAP value (impact on model output)", fontsize=8)
ax1.set_ylabel("", fontsize=8)

# 1. Primary colorbar: Normalized feature gradient (0.0 to 1.0)
new_cb_ax1 = fig.add_axes([0.73, 0.12, 0.03, 0.80])
import matplotlib as mpl
norm = mpl.colors.Normalize(vmin=0, vmax=1)
mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_custom)
fig.colorbar(mappable, cax=new_cb_ax1, ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
new_cb_ax1.tick_params(labelsize=8)

# 2. Secondary colorbar: Qualitative indicator scale (Low to High)
new_cb_ax2 = fig.add_axes([0.83, 0.12, 0.008, 0.80])
fig.colorbar(mappable, cax=new_cb_ax2, ticks=[0, 1])
new_cb_ax2.set_yticklabels(["Low", "High"])
new_cb_ax2.tick_params(labelsize=8)

# Position colorbar labels and axis titles
fig.text(0.87, 0.52, "Feature value", rotation=90, va='center', ha='center', fontsize=8)

save_plos_tiff(fig, "Fig5A.tif")

# ---- Fig 5B: Model Feature Importance (Gain) bar (top 20) ----
fig, ax = plt.subplots(figsize=(PLOS_WIDTH_SINGLE, 8.0))
model_imp = pd.DataFrame({
    "Feature": X_test_df.columns,
    "Importance": gb_model.feature_importances_
}).sort_values("Importance", ascending=False)
top_model_imp = model_imp.head(20).iloc[::-1]

bars = ax.barh(top_model_imp["Feature"], top_model_imp["Importance"],
               color="#8ecbe4", edgecolor="black", linewidth=0.5, height=0.75)
max_imp = max(top_model_imp["Importance"])
for i, v in enumerate(top_model_imp["Importance"]):
    ax.text(v + max_imp * 0.015, i, f"{v:.2f}", va="center", fontsize=8)

ax.set_xlabel("Importance (Gain)", fontsize=8)
ax.set_ylabel("Features", fontsize=8, labelpad=8) # Pushed slightly left to avoid overlap
ax.set_xticks([0.000, 0.050, 0.100, 0.150])
ax.set_xticklabels(["0.000", "0.050", "0.100", "0.150"])
ax.tick_params(labelsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Add faint vertical grid lines to match target style
ax.grid(axis='x', linestyle='-', alpha=0.3, color='lightgray')
ax.set_axisbelow(True)

ax.set_position([0.45, 0.12, 0.48, 0.80]) # Shifted right to leave more room for long labels

save_plos_tiff(fig, "Fig5B.tif")

# ---- Fig 5: Combined SHAP Beeswarm and Global Importance ----
fig = plt.figure(figsize=(10.0, 5.0))

# Subplot 1: Beeswarm (Ordered by Mean |SHAP|)
ax1 = plt.subplot(1, 2, 1)
shap.summary_plot(shap_values, X_test_scaled_df, max_display=20,
                  show=False, plot_size=None, cmap=cmap_custom, feature_names=list(X_test_scaled_df.columns))

# Subplot 2: Model Feature Importance (Ordered by Gain/Impurity)
ax2 = plt.subplot(1, 2, 2)
bars = ax2.barh(top_model_imp["Feature"], top_model_imp["Importance"],
                color="#8ecbe4", edgecolor="black", linewidth=0.5, height=0.75)
for i, v in enumerate(top_model_imp["Importance"]):
    ax2.text(v + max(top_model_imp["Importance"]) * 0.015, i,
             f"{v:.2f}", va="center", fontsize=7)

ax2.set_xlabel("Importance (Gain)", fontsize=8)
ax2.set_ylabel("") # Turn off automatic y-axis label to prevent overlap
ax2.set_xticks([0.000, 0.050, 0.100, 0.150])
ax2.set_xticklabels(["0.000", "0.050", "0.100", "0.150"])
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.tick_params(labelsize=7)

# Quantitative grid alignment
ax2.grid(axis='x', linestyle='-', alpha=0.3, color='lightgray')
ax2.set_axisbelow(True)

# Standardize subplot layout and margins
ax1.set_position([0.08, 0.15, 0.22, 0.75])
ax2.set_position([0.72, 0.15, 0.24, 0.75])

# Configure scatter point attributes
for collection in ax1.collections:
    if isinstance(collection, mcoll.PathCollection):
        collection.set_sizes([10])
        collection.set_alpha(1.0)
ax1.tick_params(labelsize=8)
ax1.set_xticks([-1.0, 0.0, 1.0])
ax1.set_xticklabels(["-1", "0", "1"])
ax1.set_xlabel("SHAP value (impact on model output)", fontsize=8)

# Configure dual colorbars
cb_ax = None
for ax in fig.axes:
    if ax.get_ylabel() == "Feature value" or ax.get_label() == "<colorbar>":
        cb_ax = ax
        break

# Reconstruct colorbars with standardized vertical orientation
if cb_ax is not None:
    cb_ax.remove()

# 1. Primary colorbar: Normalized feature gradient (0.0 to 1.0)
new_cb_ax1 = fig.add_axes([0.32, 0.15, 0.02, 0.75])
import matplotlib as mpl
norm = mpl.colors.Normalize(vmin=0, vmax=1)
mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_custom)
fig.colorbar(mappable, cax=new_cb_ax1, ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
new_cb_ax1.tick_params(labelsize=8)

# 2. Secondary colorbar: Qualitative indicator scale (Low to High)
new_cb_ax2 = fig.add_axes([0.40, 0.15, 0.007, 0.75])
fig.colorbar(mappable, cax=new_cb_ax2, ticks=[0, 1])
new_cb_ax2.set_yticklabels(["Low", "High"])
new_cb_ax2.tick_params(labelsize=8)

# Position colorbar labels and axis titles
fig.text(0.42, 0.525, "Feature value", rotation=90, va='center', ha='center', fontsize=8)
fig.text(0.48, 0.525, "Features", rotation=90, va='center', ha='center', fontsize=8)

save_plos_tiff(fig, "Fig5.tif")

# ---- Fig 6: Dependence plot for top predictor ----
top_feature = shap_importance.iloc[0]["Feature"]
plt.figure(figsize=(PLOS_WIDTH_SINGLE, 4.0))
shap.dependence_plot(top_feature, shap_values, X_test_scaled_df,
                     interaction_index="auto", show=False)
fig = plt.gcf()
for ax in fig.axes:
    ax.tick_params(labelsize=9)
plt.tight_layout()
save_plos_tiff(fig, "Fig6.tif")

# ---- Fig 7: Local waterfall explanation for one participant ----
sample_idx = 0
base_val = explainer.expected_value
if isinstance(base_val, (list, np.ndarray)):
    base_val = base_val[1] if len(base_val) > 1 else base_val[0]
expl = shap.Explanation(
    values=shap_values[sample_idx],
    base_values=base_val,
    data=X_test_scaled_df.iloc[sample_idx].values,
    feature_names=list(X_test_scaled_df.columns),
)
plt.figure(figsize=(PLOS_WIDTH_SINGLE, 4.5))
shap.plots.waterfall(expl, max_display=10, show=False)
fig = plt.gcf()
plt.tight_layout()
save_plos_tiff(fig, "Fig7.tif")

print("SHAP analysis complete.")


# =============================================================================
# 11. FEATURE-IMPORTANCE COMPARISON TABLE (LR, RF, GB, XGB, CatBoost)
# =============================================================================

TOP_N = 15
feature_rankings = {}

lr_imp = pd.DataFrame({"Feature": X.columns,
                       "Importance": np.abs(models["LR"].coef_[0])}).sort_values("Importance", ascending=False)
feature_rankings["LR"] = lr_imp["Feature"].head(TOP_N).tolist()

for tag in ["RF", "GB", "XGB", "CatBoost"]:
    imp = pd.DataFrame({
        "Feature": X.columns,
        "Importance": models[tag].feature_importances_,
    }).sort_values("Importance", ascending=False)
    feature_rankings[tag] = imp["Feature"].head(TOP_N).tolist()

importance_comparison = pd.DataFrame(feature_rankings)
importance_comparison.index = np.arange(1, len(importance_comparison) + 1)
importance_comparison.index.name = "Rank"
importance_comparison.to_csv(os.path.join(OUTPUT_DIR, "Predictor_Importance_Comparison.csv"))
print("Predictor importance comparison exported.")


# =============================================================================
# 12. MULTIVARIABLE LOGISTIC REGRESSION WITH REFERENCE CATEGORIES
# =============================================================================

logit_df = df.copy()
logit_df["CMD_binary"] = np.where(logit_df["CMD"] == "CMD Present", 1, 0)

significant_variables = predictors[:]  # same predictor set

model_df = logit_df[significant_variables + ["CMD_binary"]].dropna().copy()

display_order = {k: v for k, v in category_order.items() if k in significant_variables}
reference_categories = {
    "University_name":               "University of Dhaka",
    "Family_Type":                   "Joint Family",
    "Internet_mode":                 "Broadband connection",
    "Average_DD_Use_Time":           "<2",
    "Do_Multitask":                  "Rarely",
    "Excessive_ST_affect_AP":        "No impact",
    "Feel_tired_after_Use_DD":       "Rarely",
    "Experienced_ES_for_ST":         "No",
    "Experienced_Headaches_for_ST":  "No",
    "Experienced_NP_for_ST":         "No",
    "Experienced_BP_for_ST":         "No",
    "Experienced_HD_for_ST":         "No",
    "Physical_activity_level":       "Remained the same",
    "Decline_Overall_Happyness":     "No noticeable change",
    "Difficulty_in_Offline_Activity":"No",
    "Impact_Learning_Ability":       "No impact",
    "Overall_Health_Worsening":      "No",
    "Sleep_amount_impact":           "No",
    "Varsity_hr_tiredness":          "Never",
}

X_logit = model_df[significant_variables].copy()
for var, ref in reference_categories.items():
    if var not in X_logit.columns:
        continue
    ordered = [ref] + [x for x in display_order[var] if x != ref]
    X_logit[var] = pd.Categorical(X_logit[var], categories=ordered, ordered=True)

X_logit = pd.get_dummies(X_logit, drop_first=True, dtype=float)
X_logit.columns = [
    c.replace("<", "less_").replace(">", "greater_")
     .replace(",", "").replace(" ", "_")
     .replace("-", "_").replace("/", "_")
    for c in X_logit.columns
]

# Drop constant / duplicate columns
X_logit = X_logit.loc[:, X_logit.nunique() > 1]
X_logit = X_logit.loc[:, ~X_logit.T.duplicated()]

X_logit = sm.add_constant(X_logit, has_constant="add").astype(float)
y_logit = model_df["CMD_binary"].astype(int)

result = sm.Logit(y_logit, X_logit).fit(method="lbfgs", maxiter=1000, disp=False)
print(result.summary())

# ---- Build publication-style AOR table ----
params = result.params
conf   = result.conf_int()

OR_table = pd.DataFrame({
    "Variable":      params.index,
    "Adjusted_OR":   np.exp(params),
    "95CI_Lower":    np.exp(conf[0]),
    "95CI_Upper":    np.exp(conf[1]),
    "P_value":       result.pvalues,
})
OR_table = OR_table[OR_table["Variable"] != "const"].reset_index(drop=True)

def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""

OR_table["Significance"] = OR_table["P_value"].apply(sig_stars)
OR_table["AOR (95% CI)"] = (
    OR_table["Adjusted_OR"].round(2).astype(str) + " ("
    + OR_table["95CI_Lower"].round(2).astype(str) + "-"
    + OR_table["95CI_Upper"].round(2).astype(str) + ")"
)
OR_table.to_csv(os.path.join(OUTPUT_DIR, "Logistic_OR_Table.csv"), index=False)

# ---- Publication-style table with reference rows ----
def make_dummy_name(var, cat):
    return (f"{var}_{cat}"
            .replace("<", "less_").replace(">", "greater_")
            .replace(",", "").replace(" ", "_")
            .replace("-", "_").replace("/", "_"))

rows = []
for var in display_order:
    if var not in reference_categories:
        continue
    ref = reference_categories[var]
    rows.append({"Variable": variable_labels.get(var, var), "Category": ref,
                 "AOR (95% CI)": "Reference", "P-value": "", "Significance": ""})
    for cat in display_order[var]:
        if cat == ref:
            continue
        dname = make_dummy_name(var, cat)
        match = OR_table[OR_table["Variable"] == dname]
        if match.empty:
            continue
        r = match.iloc[0]
        rows.append({
            "Variable": "",
            "Category": cat,
            "AOR (95% CI)": r["AOR (95% CI)"],
            "P-value": ("<0.001" if r["P_value"] < 0.001 else f"{r['P_value']:.3f}"),
            "Significance": r["Significance"],
        })

publication_table = pd.DataFrame(rows)
publication_table.to_csv(os.path.join(OUTPUT_DIR, "Publication_Style_Logistic_Table.csv"),
                         index=False, encoding="utf-8-sig")
publication_table.to_excel(os.path.join(OUTPUT_DIR, "Publication_Style_Logistic_Table.xlsx"),
                           index=False)
print("Multivariable logistic regression exported.")


# =============================================================================
# 13. VARIANCE INFLATION FACTOR (VIF)
# =============================================================================

vif_df = pd.DataFrame({
    "Variable": X_logit.columns,
    "VIF": [variance_inflation_factor(X_logit.values, i) for i in range(X_logit.shape[1])],
})
vif_df = vif_df[vif_df["Variable"] != "const"].copy()
vif_df["VIF"] = vif_df["VIF"].round(3)
vif_df["Interpretation"] = np.where(
    vif_df["VIF"] < 5,  "No multicollinearity",
    np.where(vif_df["VIF"] < 10, "Moderate multicollinearity", "Severe multicollinearity"),
)
vif_df.sort_values("VIF", ascending=False, inplace=True)
vif_df.to_csv(os.path.join(OUTPUT_DIR, "VIF_Analysis.csv"), index=False, encoding="utf-8-sig")
vif_df.to_excel(os.path.join(OUTPUT_DIR, "VIF_Analysis.xlsx"), index=False)
print(f"Max VIF: {vif_df['VIF'].max():.3f}")


# =============================================================================
# 14. HOSMER-LEMESHOW GOODNESS-OF-FIT
# =============================================================================

pred_probs = result.predict(X_logit)
hl_df = pd.DataFrame({"Observed": y_logit, "Predicted": pred_probs})
hl_df["Decile"] = pd.qcut(hl_df["Predicted"], q=10, duplicates="drop")
hl_table = hl_df.groupby("Decile", observed=False).apply(
    lambda x: pd.Series({
        "Observed_1": x["Observed"].sum(),
        "Expected_1": x["Predicted"].sum(),
        "Observed_0": len(x) - x["Observed"].sum(),
        "Expected_0": len(x) - x["Predicted"].sum(),
    })
).reset_index()
hl_table["Chi_1"] = np.where(
    hl_table["Expected_1"] > 0,
    (hl_table["Observed_1"] - hl_table["Expected_1"]) ** 2 / hl_table["Expected_1"], 0)
hl_table["Chi_0"] = np.where(
    hl_table["Expected_0"] > 0,
    (hl_table["Observed_0"] - hl_table["Expected_0"]) ** 2 / hl_table["Expected_0"], 0)
HL_stat = hl_table["Chi_1"].sum() + hl_table["Chi_0"].sum()
HL_dof  = len(hl_table) - 2
HL_p    = 1 - chi2.cdf(HL_stat, HL_dof)
pd.DataFrame({
    "HL_Statistic": [round(HL_stat, 3)],
    "df":           [HL_dof],
    "P_value":      [round(HL_p, 4)],
    "Interpretation": ["Good fit" if HL_p > 0.05 else "Poor fit"],
}).to_csv(os.path.join(OUTPUT_DIR, "Hosmer_Lemeshow_Test.csv"), index=False)
print(f"Hosmer-Lemeshow: chi2={HL_stat:.3f}, df={HL_dof}, p={HL_p:.4f}")


# =============================================================================
# 15. HIERARCHICAL LOGISTIC REGRESSION (distal to proximal)
# =============================================================================

hierarchical_models = {
    "Model_I_Sociodemographic": ["University_name", "Family_Type"],
    "Model_II_Digital_Behaviour": [
        "University_name", "Family_Type",
        "Internet_mode", "Average_DD_Use_Time", "Do_Multitask",
    ],
    "Model_III_Physical_Health": [
        "University_name", "Family_Type",
        "Internet_mode", "Average_DD_Use_Time", "Do_Multitask",
        "Feel_tired_after_Use_DD",
        "Experienced_ES_for_ST", "Experienced_Headaches_for_ST",
        "Experienced_NP_for_ST", "Experienced_BP_for_ST", "Experienced_HD_for_ST",
        "Physical_activity_level", "Sleep_amount_impact", "Varsity_hr_tiredness",
    ],
    "Model_IV_Full_Adjusted": significant_variables,
}

# Null model
X_null = pd.DataFrame({"const": np.ones(len(y_logit))})
null_result = sm.Logit(y_logit, X_null).fit(method="lbfgs", maxiter=1000, disp=False)

previous_ll = null_result.llf
previous_nparam = len(null_result.params)
model_fit_summary = [{
    "Model": "Null_Model",
    "Pseudo_R2": 0.000,
    "AIC": round(null_result.aic, 2),
    "BIC": round(null_result.bic, 2),
    "LogLikelihood": round(null_result.llf, 2),
    "LR_Improvement": np.nan,
    "LR_Pvalue": np.nan,
}]
all_model_results = []

for m_name, var_list in hierarchical_models.items():
    cols = []
    for v in var_list:
        cols.extend([c for c in X_logit.columns if c.startswith(v + "_")])
    cols = list(dict.fromkeys(cols))
    X_m = X_logit[cols].copy()
    X_m = X_m.loc[:, X_m.nunique() > 1]
    X_m = add_constant(X_m, has_constant="add").astype(float)

    r = sm.Logit(y_logit, X_m).fit(method="lbfgs", maxiter=1000, disp=False)
    lr_stat = 2 * (r.llf - previous_ll)
    df_diff = len(r.params) - previous_nparam
    lr_p = chi2.sf(lr_stat, df_diff)

    model_fit_summary.append({
        "Model": m_name,
        "Pseudo_R2": round(r.prsquared, 3),
        "AIC": round(r.aic, 2),
        "BIC": round(r.bic, 2),
        "LogLikelihood": round(r.llf, 2),
        "LR_Improvement": round(lr_stat, 3),
        "LR_Pvalue": round(lr_p, 5),
    })
    previous_ll = r.llf
    previous_nparam = len(r.params)

    conf_h = r.conf_int()
    conf_h.columns = ["Lower_CI", "Upper_CI"]
    tdf = pd.DataFrame({
        "Model": m_name,
        "Variable": r.params.index,
        "Adjusted_OR": np.exp(r.params),
        "95CI_Lower": np.exp(conf_h["Lower_CI"]),
        "95CI_Upper": np.exp(conf_h["Upper_CI"]),
        "P_value": r.pvalues,
    })
    tdf = tdf[tdf["Variable"] != "const"].copy()
    tdf["AOR (95% CI)"] = (
        tdf["Adjusted_OR"].round(2).astype(str) + " ("
        + tdf["95CI_Lower"].round(2).astype(str) + "-"
        + tdf["95CI_Upper"].round(2).astype(str) + ")"
    )
    tdf["Significance"] = tdf["P_value"].apply(sig_stars)
    all_model_results.append(tdf)

hierarchical_df = pd.concat(all_model_results, ignore_index=True)
model_fit_df = pd.DataFrame(model_fit_summary)
model_fit_df.to_csv(os.path.join(OUTPUT_DIR, "Hierarchical_Model_Fit_Comparison.csv"), index=False)
hierarchical_df.to_csv(os.path.join(OUTPUT_DIR, "Hierarchical_Logistic_Regression.csv"), index=False)
print("Hierarchical logistic regression exported.")


# =============================================================================
# 16. DOMINANCE ANALYSIS
# =============================================================================

# ---- Individual predictor dominance (permutation importance on LR) ----
X_dom = pd.get_dummies(model_df[predictors], drop_first=True)
y_dom = model_df["CMD_binary"]
dom_model = LogisticRegression(max_iter=5000, random_state=123).fit(X_dom, y_dom)
perm = permutation_importance(dom_model, X_dom, y_dom, n_repeats=100, random_state=123)

dominance_results = pd.DataFrame({
    "Predictor": X_dom.columns,
    "Importance": perm.importances_mean,
}).sort_values("Importance", ascending=False)
dominance_results["Relative_Contribution_%"] = (
    dominance_results["Importance"] / dominance_results["Importance"].sum() * 100
).round(2)
dominance_results.to_csv(os.path.join(OUTPUT_DIR, "Individual_Dominance_Analysis.csv"), index=False)

# ---- Domain-level dominance ----
domains = {
    "Sociodemographic": ["University_name", "Family_Type"],
    "Digital_Behaviour": ["Internet_mode", "Average_DD_Use_Time", "Do_Multitask"],
    "Physical_Health": [
        "Feel_tired_after_Use_DD",
        "Experienced_ES_for_ST", "Experienced_Headaches_for_ST",
        "Experienced_NP_for_ST", "Experienced_BP_for_ST", "Experienced_HD_for_ST",
        "Physical_activity_level", "Sleep_amount_impact", "Varsity_hr_tiredness",
    ],
    "Psychosocial_Academic": [
        "Excessive_ST_affect_AP", "Decline_Overall_Happyness",
        "Difficulty_in_Offline_Activity", "Impact_Learning_Ability",
        "Overall_Health_Worsening",
    ],
}
all_vars = [v for lst in domains.values() for v in lst]
X_full = pd.get_dummies(model_df[all_vars], drop_first=True)
full_model = LogisticRegression(max_iter=5000, random_state=123).fit(X_full, y_dom)
full_auc = roc_auc_score(y_dom, full_model.predict_proba(X_full)[:, 1])

domain_results = []
for domain_name, remove_vars in domains.items():
    remaining = [v for v in all_vars if v not in remove_vars]
    X_temp = pd.get_dummies(model_df[remaining], drop_first=True)
    tm = LogisticRegression(max_iter=5000, random_state=123).fit(X_temp, y_dom)
    temp_auc = roc_auc_score(y_dom, tm.predict_proba(X_temp)[:, 1])
    domain_results.append({"Domain": domain_name, "AUC_Reduction": full_auc - temp_auc})

domain_results = pd.DataFrame(domain_results)
domain_results["Relative_Contribution_%"] = (
    domain_results["AUC_Reduction"] / domain_results["AUC_Reduction"].sum() * 100
).round(2)
domain_results.sort_values("Relative_Contribution_%", ascending=False, inplace=True)
domain_results.to_csv(os.path.join(OUTPUT_DIR, "Domain_Dominance_Analysis.csv"), index=False)

# ---- Fig 4: domain dominance bar chart ----
fig, ax = plt.subplots(figsize=(PLOS_WIDTH_SINGLE, 3.5))
bars = ax.bar(
    domain_results["Domain"], domain_results["Relative_Contribution_%"],
    color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"],
    edgecolor="black", linewidth=0.6,
)
for b in bars:
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6,
            f"{b.get_height():.1f}%", ha="center", fontsize=9)
ax.set_xlabel("Predictor domain")
ax.set_ylabel("Relative contribution (%)")
ax.set_xticks(range(len(domain_results)))
labels_map = {
    "Digital_Behaviour": "Digital\nbehaviour",
    "Sociodemographic": "Sociodemographic",
    "Psychosocial_Academic": "Psychosocial /\nacademic",
    "Physical_Health": "Physical\nhealth",
}
ax.set_xticklabels([labels_map[d] for d in domain_results["Domain"]])
ax.set_ylim(0, max(domain_results["Relative_Contribution_%"]) + 10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
save_plos_tiff(fig, "Fig4.tif")


# =============================================================================
# 17. MEDIATION ANALYSIS (screen time -> sleep / learning -> CMD)
# =============================================================================

med_df = model_df.copy()
med_df["DD_Time_Score"] = med_df["Average_DD_Use_Time"].map({"<2":1, "2-4":2, "4-6":3, ">6":4})
med_df["Sleep_Score"]   = med_df["Sleep_amount_impact"].map({"No":0, "Not sure":1, "Yes":2})
med_df["Learning_Score"] = med_df["Impact_Learning_Ability"].map(
    {"No impact":0, "Yes, positively":1, "Yes, negatively":2}
)
for c in ["DD_Time_Score", "Sleep_Score", "Learning_Score", "CMD_binary"]:
    med_df[c] = pd.to_numeric(med_df[c], errors="coerce")

med_sleep = pg.mediation_analysis(
    data=med_df.dropna(subset=["DD_Time_Score", "Sleep_Score", "CMD_binary"]),
    x="DD_Time_Score", m="Sleep_Score", y="CMD_binary",
    n_boot=5000, seed=123,
)
med_sleep.to_csv(os.path.join(OUTPUT_DIR, "Mediation_SleepImpact_CMD.csv"), index=False)

med_learning = pg.mediation_analysis(
    data=med_df.dropna(subset=["DD_Time_Score", "Learning_Score", "CMD_binary"]),
    x="DD_Time_Score", m="Learning_Score", y="CMD_binary",
    n_boot=5000, seed=123,
)
med_learning.to_csv(os.path.join(OUTPUT_DIR, "Mediation_LearningImpact_CMD.csv"), index=False)
print("Mediation analyses exported.")


# =============================================================================
# 18. FIGURES 1 AND 2 (CMD PREVALENCE)
# =============================================================================

# ---- Fig 1: overall CMD prevalence ----
cmd_prev = df["CMD"].value_counts(normalize=True) * 100
fig, ax = plt.subplots(figsize=(PLOS_WIDTH_SINGLE, 3.8))
bars = ax.bar(cmd_prev.index, cmd_prev.values,
              color=["#4C72B0", "#DD8452"], width=0.55,
              edgecolor="black", linewidth=0.6)
for b, v in zip(bars, cmd_prev.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%",
            ha="center", fontsize=9)
ax.set_xlabel("CMD status")
ax.set_ylabel("Prevalence (%)")
ax.set_ylim(0, max(cmd_prev.values) + 8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
save_plos_tiff(fig, "Fig1.tif")

# ---- Fig 2: CMD prevalence across daily digital device use ----
order = ["<2", "2-4", "4-6", ">6"]
trend = pd.crosstab(df["Average_DD_Use_Time"], df["CMD"], normalize="index").reindex(order) * 100
fig, ax = plt.subplots(figsize=(PLOS_WIDTH_SINGLE, 3.8))
ax.plot(order, trend["CMD Present"].values, marker="o", linewidth=2,
        color="#4C72B0", label="CMD present")
ax.plot(order, trend["No CMD"].values, marker="s", linewidth=2,
        color="#DD8452", label="No CMD")
for i, v in enumerate(trend["CMD Present"].values):
    ax.text(i, v + 2, f"{v:.1f}%", ha="center", fontsize=8)
ax.set_xlabel("Daily digital device use (hours)")
ax.set_ylabel("Prevalence (%)")
ax.set_ylim(0, 100)
ax.legend(frameon=False, loc="upper left")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
save_plos_tiff(fig, "Fig2.tif")

# ---- Export trend table ----
trend_export = trend.copy()
trend_export.index = ["Less than 2 h", "2 to 4 h", "4 to 6 h", "More than 6 h"]
trend_export.to_csv(os.path.join(OUTPUT_DIR, "CMD_Trend_Table.csv"), encoding="utf-8-sig")


# =============================================================================
# 19. COCHRAN-ARMITAGE TREND TEST
# =============================================================================

trend_counts = pd.crosstab(df["Average_DD_Use_Time"], df["CMD"]).reindex(order)
trend_result = Table(trend_counts.values).test_ordinal_association()
pd.DataFrame({
    "Statistic": ["Z_statistic", "P_value"],
    "Value":     [trend_result.zscore, trend_result.pvalue],
}).to_csv(os.path.join(OUTPUT_DIR, "Cochran_Armitage_Trend_Test.csv"), index=False)
print(f"Cochran-Armitage trend: Z={trend_result.zscore:.3f}, p={trend_result.pvalue:.6f}")


# =============================================================================
# 20. SUPPLEMENTARY: CMD SEVERITY BY DIGITAL DEVICE USE
# =============================================================================

if "CMD_Severity" in df.columns:
    severity = pd.crosstab(df["Average_DD_Use_Time"], df["CMD_Severity"], normalize="index").reindex(order) * 100
    fig, ax = plt.subplots(figsize=(PLOS_WIDTH_FULL, 4.5))
    severity.plot(kind="bar", ax=ax, width=0.75,
                  color=sns.color_palette("Blues_d", severity.shape[1]),
                  edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Daily digital device use (hours)")
    ax.set_ylabel("Percentage (%)")
    ax.set_xticklabels(order, rotation=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(title="CMD severity", frameon=False, fontsize=8, title_fontsize=9,
              loc="upper left", bbox_to_anchor=(1.01, 1.0))
    plt.tight_layout()
    save_plos_tiff(fig, "S_CMD_Severity_by_Screen_Time.tif")


# =============================================================================
# 21. DONE
# =============================================================================

print(f"""
============================================================
PIPELINE EXECUTION COMPLETED

Analytic outputs (CSV / XLSX / TXT):
  {OUTPUT_DIR}

Figures (TIFF, 300 dpi, RGB):
  {BMC_DIR}

Main figures produced:
  Fig1.tif  - CMD prevalence
  Fig2.tif  - CMD prevalence across daily digital device use
  Fig3.tif  - ROC curve comparison
  Fig4.tif  - Domain-level dominance
  Fig5A.tif - SHAP beeswarm summary (Gradient Boosting)
  Fig5B.tif - Global SHAP importance bar
  Fig5.tif  - SHAP beeswarm and global importance (Combined)
  Fig6.tif  - SHAP dependence plot for the top predictor
  Fig7.tif  - Local SHAP waterfall for one participant

Supporting figures produced:
  S_Confusion_<Model>.tif   - Per-model confusion matrices
  S_Model_Performance_Panel.tif
  S_CMD_Severity_by_Screen_Time.tif
============================================================
""")


