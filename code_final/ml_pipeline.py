#!/usr/bin/env python3
"""
Machine Learning Classification Pipeline for Spinal Assessment Data
- Multiple classifiers comparison with statistical testing
- Decision Tree hyperparameter optimization
- Academic-quality visualizations
- External validation + internal test set evaluation
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import (
    StratifiedKFold, cross_val_predict, GridSearchCV, StratifiedShuffleSplit
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier,
    ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score,
    matthews_corrcoef, cohen_kappa_score, brier_score_loss,
    log_loss, balanced_accuracy_score
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from scipy import stats
from scipy.stats import chi2_contingency

def mcnemar_test(table):
    """Manual McNemar's test with continuity correction"""
    n01 = table[0][1]
    n10 = table[1][0]
    if n01 + n10 == 0:
        return type('Result', (), {'pvalue': 1.0})()
    chi2 = (abs(n01 - n10) - 1)**2 / (n01 + n10)
    pval = 1 - stats.chi2.cdf(chi2, df=1)
    return type('Result', (), {'pvalue': pval})()
import graphviz
import json, ast, io, os

OUT = Path('/mnt/user-data/outputs')
OUT.mkdir(exist_ok=True)

# ============================================================
# Publication-quality plot settings
# ============================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.linewidth': 1.2,
    'axes.grid': False,
})

# ============================================================
# 1. DATA LOADING & FEATURE ENGINEERING
# ============================================================
print("=" * 70)
print("STEP 1: Data Loading & Feature Engineering")
print("=" * 70)

df = pd.read_excel('/mnt/user-data/outputs/merged_patient_data.xlsx')
# Remove header artifact row
df = df[~df['group'].isin(['group', None])].copy()
df = df[df['group'].notna()].copy()

# Simplified column name mapping for score columns
SHORT_NAMES = {
    'image_back__ParaspinalTensionAsymmetry__score': 'Back_ParaspinalAsym',
    'image_back__PelvicRotation__score': 'Back_PelvicRot',
    'image_back__ScapularSymmetry__score': 'Back_ScapularSym',
    'image_back__SpinalVerticality__score': 'Back_SpinalVert',
    'image_frontal__CoronalCurvature__score': 'Front_CoronalCurv',
    'image_frontal__GlobalBalance__score': 'Front_GlobalBal',
    'image_frontal__PelvicLeveling__score': 'Front_PelvicLevel',
    'image_frontal__ShoulderSymmetry__score': 'Front_ShoulderSym',
    'image_frontal__SpinalAlignment__score': 'Front_SpinalAlign',
    'image_lateral__HeadPosition__score': 'Lat_HeadPos',
    'image_lateral__LumbarLordosis__score': 'Lat_LumbarLord',
    'image_lateral__SagittalBalance__score': 'Lat_SagittalBal',
    'image_lateral__ThoracicKyphosis__score': 'Lat_ThoracicKyph',
    'image_lateral__TrunkInclination__score': 'Lat_TrunkIncl',
    'video_roll_left__HipShoulderCoordination__score': 'RollL_HipShldCoord',
    'video_roll_left__PainDuringMovement__score': 'RollL_Pain',
    'video_roll_left__RollingSmoothness__score': 'RollL_Smoothness',
    'video_roll_left__SupportHandUsage__score': 'RollL_HandSupport',
    'video_roll_right__HipShoulderCoordination__score': 'RollR_HipShldCoord',
    'video_roll_right__PainDuringMovement__score': 'RollR_Pain',
    'video_roll_right__RollingSmoothness__score': 'RollR_Smoothness',
    'video_roll_right__SupportHandUsage__score': 'RollR_HandSupport',
    'video_sit_to_supine__ArmAssistance__score': 'SitSup_ArmAssist',
    'video_sit_to_supine__ExecutionTime__score': 'SitSup_ExecTime',
    'video_sit_to_supine__MotionCoordination__score': 'SitSup_MotionCoord',
    'video_sit_to_supine__PainResponseLevel__score': 'SitSup_PainResp',
    'video_sit_to_supine__TrunkControl__score': 'SitSup_TrunkCtrl',
    'video_supine_to_sit__ArmAssistance__score': 'SupSit_ArmAssist',
    'video_supine_to_sit__ExecutionTime__score': 'SupSit_ExecTime',
    'video_supine_to_sit__MotionCoordination__score': 'SupSit_MotionCoord',
    'video_supine_to_sit__PainResponseLevel__score': 'SupSit_PainResp',
    'video_supine_to_sit__TrunkControl__score': 'SupSit_TrunkCtrl',
}

# Rename score columns
df = df.rename(columns=SHORT_NAMES)

# ---- Encode categorical features ----
# Sex: M=1, F=0
df['sex'] = df['sex'].map({'M': 1, 'F': 0})

# Smoking: binary (有=1, 无=0)
df['smoking_history'] = df['smoking_history'].apply(
    lambda x: 0 if pd.isna(x) or x in ['无', '信息缺失'] else 1
)

# Drinking: binary
df['drinking_history'] = df['drinking_history'].apply(
    lambda x: 0 if pd.isna(x) or x in ['无', '信息缺失'] else 1
)

# Injury: ordinal (无=0, 低能量=1, 高能量=2)
def encode_injury(x):
    if pd.isna(x) or x in ['无', '无明显外伤', '信息缺失']:
        return 0
    elif '低能量' in str(x):
        return 1
    elif '高能量' in str(x):
        return 2
    return 0
df['injury_history'] = df['injury_history'].apply(encode_injury)

# Ordinal text scores: SupportHandUsage, ArmAssistance
hand_support_map = {'无需辅助': 0, '单手辅助': 1, '双手用力推床': 2}
arm_assist_map = {'无需辅助': 0, '部分支撑': 1, '明显依赖支撑': 2}

for col in ['RollL_HandSupport', 'RollR_HandSupport']:
    df[col] = df[col].map(hand_support_map)
for col in ['SitSup_ArmAssist', 'SupSit_ArmAssist']:
    df[col] = df[col].map(arm_assist_map)

# Target encoding
df['target'] = (df['group'] == 'positive').astype(int)

# Feature columns
demo_features = ['age', 'sex', 'height_cm', 'weight_kg',
                 'smoking_history', 'drinking_history', 'injury_history']
score_features = list(SHORT_NAMES.values())
all_features = demo_features + score_features

# Ensure numeric
for c in all_features:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# Drop rows where target is missing
df = df.dropna(subset=['target'])
print(f"Total samples after cleaning: {len(df)}")
print(f"Class distribution: {df['target'].value_counts().to_dict()}")
print(f"Features: {len(all_features)} ({len(demo_features)} demographic + {len(score_features)} score)")

# ============================================================
# 2. DATA SPLITTING
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: Data Splitting (External 54 → Train/Test 7:3)")
print("=" * 70)

np.random.seed(42)

# Stratified split for external validation (54 samples)
from sklearn.model_selection import train_test_split

df_remain, df_external = train_test_split(
    df, test_size=54, stratify=df['target'], random_state=42
)
# Internal: 70/30
df_train, df_test = train_test_split(
    df_remain, test_size=0.3, stratify=df_remain['target'], random_state=42
)

print(f"Training set:          {len(df_train)} (pos={df_train['target'].sum()}, neg={len(df_train)-df_train['target'].sum()})")
print(f"Internal test set:     {len(df_test)} (pos={df_test['target'].sum()}, neg={len(df_test)-df_test['target'].sum()})")
print(f"External validation:   {len(df_external)} (pos={df_external['target'].sum()}, neg={len(df_external)-df_external['target'].sum()})")

X_train = df_train[all_features].values
y_train = df_train['target'].values
X_test = df_test[all_features].values
y_test = df_test['target'].values
X_ext = df_external[all_features].values
y_ext = df_external['target'].values

# ============================================================
# 3. PREPROCESSING PIPELINE (imputation + scaling)
# ============================================================
from sklearn.impute import KNNImputer

# Use KNN imputation (more sophisticated than mean/median)
imputer = KNNImputer(n_neighbors=5)
X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)
X_ext_imp = imputer.transform(X_ext)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_imp)
X_test_sc = scaler.transform(X_test_imp)
X_ext_sc = scaler.transform(X_ext_imp)

print(f"\nAfter imputation — NaN remaining: train={np.isnan(X_train_imp).sum()}, test={np.isnan(X_test_imp).sum()}, ext={np.isnan(X_ext_imp).sum()}")

# ============================================================
# 4. MODEL DEFINITIONS
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: Model Training & Evaluation")
print("=" * 70)

models = {
    'Decision Tree': DecisionTreeClassifier(max_depth=5, min_samples_split=10,
                                            min_samples_leaf=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=8,
                                            min_samples_leaf=3, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                                     learning_rate=0.1, random_state=42),
    'Logistic Regression': LogisticRegression(C=1.0, penalty='l2', max_iter=1000,
                                               random_state=42, solver='lbfgs'),
    'SVM (RBF)': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True,
                      random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=7),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, learning_rate=0.1,
                                    random_state=42),
    'Extra Trees': ExtraTreesClassifier(n_estimators=200, max_depth=8,
                                         random_state=42, n_jobs=-1),
    'MLP': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                          random_state=42, early_stopping=True),
}

# Models that need scaled data
needs_scaling = {'Logistic Regression', 'SVM (RBF)', 'KNN', 'MLP'}

def evaluate_model(model, X_tr, y_tr, X_te, y_te, scaled=False):
    """Comprehensive model evaluation with bootstrapped CI"""
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_te)[:, 1]
    elif hasattr(model, 'decision_function'):
        y_prob = model.decision_function(X_te)
    else:
        y_prob = y_pred.astype(float)

    metrics = {
        'Accuracy': accuracy_score(y_te, y_pred),
        'Balanced Accuracy': balanced_accuracy_score(y_te, y_pred),
        'Sensitivity': recall_score(y_te, y_pred, pos_label=1),
        'Specificity': recall_score(y_te, y_pred, pos_label=0),
        'Precision': precision_score(y_te, y_pred, zero_division=0),
        'F1-Score': f1_score(y_te, y_pred),
        'AUC-ROC': roc_auc_score(y_te, y_prob),
        'MCC': matthews_corrcoef(y_te, y_pred),
        'Cohen Kappa': cohen_kappa_score(y_te, y_pred),
    }

    # Bootstrap 95% CI for AUC
    n_boot = 1000
    aucs = []
    rng = np.random.RandomState(42)
    for _ in range(n_boot):
        idx = rng.choice(len(y_te), len(y_te), replace=True)
        if len(np.unique(y_te[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_te[idx], y_prob[idx]))
    metrics['AUC 95% CI'] = f"({np.percentile(aucs, 2.5):.3f}-{np.percentile(aucs, 97.5):.3f})"

    return metrics, y_pred, y_prob

# ---- Train and evaluate all models ----
results_internal = {}
results_external = {}
predictions_internal = {}
predictions_external = {}
probas_internal = {}
probas_external = {}

for name, model in models.items():
    print(f"  Training {name}...")
    scaled = name in needs_scaling

    Xtr = X_train_sc if scaled else X_train_imp
    Xte = X_test_sc if scaled else X_test_imp
    Xex = X_ext_sc if scaled else X_ext_imp

    # Internal test
    met_int, pred_int, prob_int = evaluate_model(model, Xtr, y_train, Xte, y_test)
    results_internal[name] = met_int
    predictions_internal[name] = pred_int
    probas_internal[name] = prob_int

    # Re-train for external (same model, same training data)
    model.fit(Xtr, y_train)
    pred_ext = model.predict(Xex)
    prob_ext = model.predict_proba(Xex)[:, 1] if hasattr(model, 'predict_proba') else pred_ext.astype(float)

    met_ext, _, _ = evaluate_model(model, Xtr, y_train, Xex, y_ext)
    results_external[name] = met_ext
    predictions_external[name] = pred_ext
    probas_external[name] = prob_ext

# ---- Statistical comparison (McNemar's test vs best model) ----
print("\n  Computing McNemar's P-values...")
# Find the best model by AUC on internal test
best_model_name = max(results_internal, key=lambda k: results_internal[k]['AUC-ROC'])
print(f"  Reference model (best AUC): {best_model_name}")

pvalues_internal = {}
pvalues_external = {}
for name in models:
    # McNemar's test: compare each model to the best model
    if name == best_model_name:
        pvalues_internal[name] = 1.0
        pvalues_external[name] = 1.0
        continue
    # Internal
    a = predictions_internal[name]
    b = predictions_internal[best_model_name]
    n01 = np.sum((a != y_test) & (b == y_test))
    n10 = np.sum((a == y_test) & (b != y_test))
    if n01 + n10 > 0:
        # Exact McNemar
        pval = stats.binomtest(n01, n01 + n10, 0.5).pvalue if (n01 + n10) < 25 else \
               mcnemar_test([[np.sum((a==y_test)&(b==y_test)), n01],
                        [n10, np.sum((a!=y_test)&(b!=y_test))]]).pvalue
    else:
        pval = 1.0
    pvalues_internal[name] = pval

    # External
    a = predictions_external[name]
    b = predictions_external[best_model_name]
    n01 = np.sum((a != y_ext) & (b == y_ext))
    n10 = np.sum((a == y_ext) & (b != y_ext))
    if n01 + n10 > 0:
        pval = stats.binomtest(n01, n01 + n10, 0.5).pvalue if (n01 + n10) < 25 else \
               mcnemar_test([[np.sum((a==y_ext)&(b==y_ext)), n01],
                        [n10, np.sum((a!=y_ext)&(b!=y_ext))]]).pvalue
    else:
        pval = 1.0
    pvalues_external[name] = pval

# Also compute DeLong-like test for AUC comparison using bootstrap
def bootstrap_auc_pvalue(y_true, prob_a, prob_b, n_boot=2000):
    """Bootstrap test for difference in AUCs"""
    rng = np.random.RandomState(42)
    diffs = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        auc_a = roc_auc_score(y_true[idx], prob_a[idx])
        auc_b = roc_auc_score(y_true[idx], prob_b[idx])
        diffs.append(auc_a - auc_b)
    if len(diffs) == 0:
        return 1.0
    diffs = np.array(diffs)
    p = 2 * min(np.mean(diffs >= 0), np.mean(diffs <= 0))
    return p

auc_pvalues_int = {}
auc_pvalues_ext = {}
for name in models:
    if name == best_model_name:
        auc_pvalues_int[name] = '-'
        auc_pvalues_ext[name] = '-'
        continue
    auc_pvalues_int[name] = bootstrap_auc_pvalue(
        y_test, probas_internal[name], probas_internal[best_model_name])
    auc_pvalues_ext[name] = bootstrap_auc_pvalue(
        y_ext, probas_external[name], probas_external[best_model_name])

# ============================================================
# 5. TABLE 1: MODEL COMPARISON
# ============================================================
print("\n" + "=" * 70)
print("TABLE 1: Model Classification Performance Comparison")
print("=" * 70)

def build_table(results_dict, pvals_mcn, pvals_auc, set_name):
    rows = []
    for name in models:
        r = results_dict[name]
        row = {
            'Model': name,
            'Accuracy': f"{r['Accuracy']:.3f}",
            'Bal. Accuracy': f"{r['Balanced Accuracy']:.3f}",
            'Sensitivity': f"{r['Sensitivity']:.3f}",
            'Specificity': f"{r['Specificity']:.3f}",
            'Precision': f"{r['Precision']:.3f}",
            'F1-Score': f"{r['F1-Score']:.3f}",
            'AUC-ROC': f"{r['AUC-ROC']:.3f}",
            'AUC 95% CI': r['AUC 95% CI'],
            'MCC': f"{r['MCC']:.3f}",
            'Kappa': f"{r['Cohen Kappa']:.3f}",
            'McNemar P': f"{pvals_mcn[name]:.4f}" if isinstance(pvals_mcn[name], float) else pvals_mcn[name],
            'AUC P (bootstrap)': f"{pvals_auc[name]:.4f}" if isinstance(pvals_auc[name], float) else pvals_auc[name],
        }
        rows.append(row)
    return pd.DataFrame(rows)

table1_int = build_table(results_internal, pvalues_internal, auc_pvalues_int, 'Internal Test')
table1_ext = build_table(results_external, pvalues_external, auc_pvalues_ext, 'External Validation')

print("\n--- Internal Test Set ---")
print(table1_int.to_string(index=False))
print(f"\n--- External Validation Set ---")
print(table1_ext.to_string(index=False))

# ============================================================
# 6. DECISION TREE PARAMETER OPTIMIZATION (TABLE 2)
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: Decision Tree Hyperparameter Exploration")
print("=" * 70)

dt_params = []
for depth in [2, 3, 4, 5, 6, 7, 8, 10, None]:
    for min_split in [2, 5, 10, 20]:
        for min_leaf in [1, 3, 5, 10]:
            for criterion in ['gini', 'entropy']:
                dt = DecisionTreeClassifier(
                    max_depth=depth, min_samples_split=min_split,
                    min_samples_leaf=min_leaf, criterion=criterion,
                    random_state=42
                )
                # 5-fold CV on training set
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                cv_scores = []
                for tr_idx, val_idx in cv.split(X_train_imp, y_train):
                    dt.fit(X_train_imp[tr_idx], y_train[tr_idx])
                    prob = dt.predict_proba(X_train_imp[val_idx])[:, 1]
                    cv_scores.append(roc_auc_score(y_train[val_idx], prob))
                cv_auc = np.mean(cv_scores)
                cv_std = np.std(cv_scores)

                # Evaluate on test and external
                dt.fit(X_train_imp, y_train)
                pred_te = dt.predict(X_test_imp)
                prob_te = dt.predict_proba(X_test_imp)[:, 1]
                pred_ex = dt.predict(X_ext_imp)
                prob_ex = dt.predict_proba(X_ext_imp)[:, 1]

                dt_params.append({
                    'max_depth': str(depth) if depth else 'None',
                    'min_samples_split': min_split,
                    'min_samples_leaf': min_leaf,
                    'criterion': criterion,
                    'CV AUC (mean±std)': f"{cv_auc:.3f}±{cv_std:.3f}",
                    'cv_auc': cv_auc,
                    'Test Accuracy': accuracy_score(y_test, pred_te),
                    'Test AUC': roc_auc_score(y_test, prob_te),
                    'Test Sensitivity': recall_score(y_test, pred_te, pos_label=1),
                    'Test Specificity': recall_score(y_test, pred_te, pos_label=0),
                    'Test F1': f1_score(y_test, pred_te),
                    'Ext Accuracy': accuracy_score(y_ext, pred_ex),
                    'Ext AUC': roc_auc_score(y_ext, prob_ex),
                    'Ext Sensitivity': recall_score(y_ext, pred_ex, pos_label=1),
                    'Ext Specificity': recall_score(y_ext, pred_ex, pos_label=0),
                    'Ext F1': f1_score(y_ext, pred_ex),
                    'n_leaves': dt.get_n_leaves(),
                    'tree_depth': dt.get_depth(),
                })

df_dt = pd.DataFrame(dt_params)
df_dt = df_dt.sort_values('cv_auc', ascending=False)

# Table 2: Top 20 configurations
table2 = df_dt.head(20).drop(columns=['cv_auc']).reset_index(drop=True)
table2.index = table2.index + 1  # 1-indexed
print("\nTable 2: Top 20 Decision Tree Configurations")
print(table2[['max_depth','min_samples_split','min_samples_leaf','criterion',
              'CV AUC (mean±std)','Test AUC','Test Accuracy','Test F1',
              'Ext AUC','Ext Accuracy','Ext F1']].to_string())

# Best DT config
best_dt_row = df_dt.iloc[0]
print(f"\nBest DT config: depth={best_dt_row['max_depth']}, "
      f"split={best_dt_row['min_samples_split']}, leaf={best_dt_row['min_samples_leaf']}, "
      f"criterion={best_dt_row['criterion']}")

best_dt = DecisionTreeClassifier(
    max_depth=int(best_dt_row['max_depth']) if best_dt_row['max_depth'] != 'None' else None,
    min_samples_split=int(best_dt_row['min_samples_split']),
    min_samples_leaf=int(best_dt_row['min_samples_leaf']),
    criterion=best_dt_row['criterion'],
    random_state=42
)
best_dt.fit(X_train_imp, y_train)

# ============================================================
# 7. SAVE TABLES TO EXCEL
# ============================================================
with pd.ExcelWriter(str(OUT / 'ML_Results_Tables.xlsx'), engine='openpyxl') as writer:
    table1_int.to_excel(writer, sheet_name='Table1_Internal_Test', index=False)
    table1_ext.to_excel(writer, sheet_name='Table1_External_Valid', index=False)
    table2.to_excel(writer, sheet_name='Table2_DT_Params', index=True)
    # Also save full DT grid
    df_dt.drop(columns=['cv_auc']).to_excel(writer, sheet_name='DT_Full_Grid', index=False)
print(f"\nTables saved to {OUT / 'ML_Results_Tables.xlsx'}")

# ============================================================
# 8. VISUALIZATIONS
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: Generating Academic Figures")
print("=" * 70)

# Color palette
COLORS = {
    'Decision Tree': '#E74C3C',
    'Random Forest': '#2ECC71',
    'Gradient Boosting': '#3498DB',
    'Logistic Regression': '#9B59B6',
    'SVM (RBF)': '#F39C12',
    'KNN': '#1ABC9C',
    'AdaBoost': '#E67E22',
    'Extra Trees': '#34495E',
    'MLP': '#E91E63',
}

# ========== FIGURE 1: ROC Curves (Internal + External) ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, y_true, probas, title in [
    (axes[0], y_test, probas_internal, 'Internal Test Set'),
    (axes[1], y_ext, probas_external, 'External Validation Set')
]:
    for name in models:
        fpr, tpr, _ = roc_curve(y_true, probas[name])
        auc = roc_auc_score(y_true, probas[name])
        ax.plot(fpr, tpr, color=COLORS[name], lw=2, alpha=0.85,
                label=f'{name} (AUC={auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('1 - Specificity (FPR)')
    ax.set_ylabel('Sensitivity (TPR)')
    ax.set_title(title)
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
fig.suptitle('Figure 1. ROC Curves for All Classifiers', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(str(OUT / 'Fig1_ROC_Curves.png'))
fig.savefig(str(OUT / 'Fig1_ROC_Curves.pdf'))
plt.close()
print("  Fig 1: ROC Curves ✓")

# ========== FIGURE 2: Precision-Recall Curves ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, y_true, probas, title in [
    (axes[0], y_test, probas_internal, 'Internal Test Set'),
    (axes[1], y_ext, probas_external, 'External Validation Set')
]:
    for name in models:
        prec, rec, _ = precision_recall_curve(y_true, probas[name])
        ap = average_precision_score(y_true, probas[name])
        ax.plot(rec, prec, color=COLORS[name], lw=2, alpha=0.85,
                label=f'{name} (AP={ap:.3f})')
    baseline = y_true.mean()
    ax.axhline(y=baseline, color='k', ls='--', lw=1, alpha=0.5)
    ax.set_xlabel('Recall (Sensitivity)')
    ax.set_ylabel('Precision (PPV)')
    ax.set_title(title)
    ax.legend(loc='best', fontsize=8, framealpha=0.9)
fig.suptitle('Figure 2. Precision-Recall Curves', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(str(OUT / 'Fig2_PR_Curves.png'))
fig.savefig(str(OUT / 'Fig2_PR_Curves.pdf'))
plt.close()
print("  Fig 2: PR Curves ✓")

# ========== FIGURE 3: Confusion Matrices (Best 4 models) ==========
top4 = sorted(results_internal, key=lambda k: results_internal[k]['AUC-ROC'], reverse=True)[:4]

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
for i, name in enumerate(top4):
    for j, (y_true, preds, title) in enumerate([
        (y_test, predictions_internal[name], 'Internal'),
        (y_ext, predictions_external[name], 'External')
    ]):
        cm = confusion_matrix(y_true, preds)
        ax = axes[j, i]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Control', 'Positive'],
                    yticklabels=['Control', 'Positive'],
                    annot_kws={'size': 14})
        ax.set_xlabel('Predicted' if j == 1 else '')
        ax.set_ylabel('Actual' if i == 0 else '')
        ax.set_title(f'{name}\n({title})', fontsize=11)
fig.suptitle('Figure 3. Confusion Matrices for Top-4 Classifiers',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(str(OUT / 'Fig3_Confusion_Matrices.png'))
fig.savefig(str(OUT / 'Fig3_Confusion_Matrices.pdf'))
plt.close()
print("  Fig 3: Confusion Matrices ✓")

# ========== FIGURE 4: Model Performance Radar Chart ==========
from math import pi

metrics_radar = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1-Score', 'AUC-ROC', 'MCC']
fig, axes = plt.subplots(1, 2, figsize=(16, 7), subplot_kw=dict(polar=True))

for ax, results, title in [(axes[0], results_internal, 'Internal Test'),
                            (axes[1], results_external, 'External Validation')]:
    angles = [n / float(len(metrics_radar)) * 2 * pi for n in range(len(metrics_radar))]
    angles += angles[:1]

    for name in models:
        values = [results[name][m] for m in metrics_radar]
        values += values[:1]
        ax.plot(angles, values, color=COLORS[name], lw=1.5, alpha=0.7, label=name)
        ax.fill(angles, values, color=COLORS[name], alpha=0.05)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_radar, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=13, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=7)

fig.suptitle('Figure 4. Radar Chart: Multi-Metric Model Comparison',
             fontsize=14, fontweight='bold', y=1.05)
plt.tight_layout()
fig.savefig(str(OUT / 'Fig4_Radar_Chart.png'))
fig.savefig(str(OUT / 'Fig4_Radar_Chart.pdf'))
plt.close()
print("  Fig 4: Radar Chart ✓")

# ========== FIGURE 5: Feature Importance (RF + GB + Permutation) ==========
fig, axes = plt.subplots(1, 3, figsize=(20, 8))

# 5a: Random Forest Gini importance
rf_model = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=3,
                                   random_state=42, n_jobs=-1)
rf_model.fit(X_train_imp, y_train)
imp_rf = pd.Series(rf_model.feature_importances_, index=all_features).sort_values(ascending=True)
imp_rf.tail(20).plot.barh(ax=axes[0], color='#2ECC71', edgecolor='#27AE60')
axes[0].set_title('(A) Random Forest\n(Gini Importance)', fontsize=12)
axes[0].set_xlabel('Importance')

# 5b: Gradient Boosting importance
gb_model = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                       random_state=42)
gb_model.fit(X_train_imp, y_train)
imp_gb = pd.Series(gb_model.feature_importances_, index=all_features).sort_values(ascending=True)
imp_gb.tail(20).plot.barh(ax=axes[1], color='#3498DB', edgecolor='#2980B9')
axes[1].set_title('(B) Gradient Boosting\n(Feature Importance)', fontsize=12)
axes[1].set_xlabel('Importance')

# 5c: Permutation importance (model-agnostic) on best model
best_model_for_perm = rf_model
perm_imp = permutation_importance(best_model_for_perm, X_test_imp, y_test,
                                   n_repeats=30, random_state=42, n_jobs=-1)
imp_perm = pd.Series(perm_imp.importances_mean, index=all_features).sort_values(ascending=True)
imp_perm_std = pd.Series(perm_imp.importances_std, index=all_features)
top20_perm = imp_perm.tail(20)
axes[2].barh(range(len(top20_perm)), top20_perm.values, color='#E74C3C', edgecolor='#C0392B',
             xerr=imp_perm_std[top20_perm.index].values, capsize=2)
axes[2].set_yticks(range(len(top20_perm)))
axes[2].set_yticklabels(top20_perm.index)
axes[2].set_title('(C) Permutation Importance\n(Random Forest, Test Set)', fontsize=12)
axes[2].set_xlabel('Mean Accuracy Decrease')

fig.suptitle('Figure 5. Feature Importance Analysis',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(str(OUT / 'Fig5_Feature_Importance.png'))
fig.savefig(str(OUT / 'Fig5_Feature_Importance.pdf'))
plt.close()
print("  Fig 5: Feature Importance ✓")

# ========== FIGURE 6: Decision Tree Visualization ==========
# Build a clean, interpretable tree
dt_viz = DecisionTreeClassifier(
    max_depth=4,
    min_samples_split=int(best_dt_row['min_samples_split']),
    min_samples_leaf=int(best_dt_row['min_samples_leaf']),
    criterion=best_dt_row['criterion'],
    random_state=42
)
dt_viz.fit(X_train_imp, y_train)

fig, ax = plt.subplots(figsize=(28, 14))
plot_tree(dt_viz, feature_names=all_features,
          class_names=['Control', 'Positive'],
          filled=True, rounded=True, fontsize=8,
          proportion=True, impurity=True, ax=ax)
ax.set_title('Figure 6. Optimized Decision Tree Structure (max_depth=4)',
             fontsize=16, fontweight='bold')
plt.tight_layout()
fig.savefig(str(OUT / 'Fig6_Decision_Tree.png'))
fig.savefig(str(OUT / 'Fig6_Decision_Tree.pdf'))
plt.close()
print("  Fig 6: Decision Tree ✓")

# ========== FIGURE 7: DT Depth vs Performance ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Aggregate by depth
depth_perf = df_dt.copy()
depth_perf['max_depth_num'] = depth_perf['max_depth'].apply(lambda x: 15 if x == 'None' else int(x))
depth_agg = depth_perf.groupby('max_depth_num').agg({
    'Test AUC': ['mean', 'std', 'max'],
    'Ext AUC': ['mean', 'std', 'max'],
    'Test Accuracy': ['mean', 'max'],
    'Ext Accuracy': ['mean', 'max'],
}).reset_index()

depths = depth_agg['max_depth_num'].values
labels = [str(d) if d < 15 else '∞' for d in depths]

ax = axes[0]
ax.errorbar(depths, depth_agg[('Test AUC', 'mean')], yerr=depth_agg[('Test AUC', 'std')],
            marker='o', color='#E74C3C', lw=2, capsize=4, label='Internal Test (mean±std)')
ax.errorbar(depths, depth_agg[('Ext AUC', 'mean')], yerr=depth_agg[('Ext AUC', 'std')],
            marker='s', color='#3498DB', lw=2, capsize=4, label='External Valid (mean±std)')
ax.set_xticks(depths)
ax.set_xticklabels(labels)
ax.set_xlabel('Max Depth')
ax.set_ylabel('AUC-ROC')
ax.set_title('(A) AUC vs. Decision Tree Depth')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(depths, depth_agg[('Test Accuracy', 'max')], 'o-', color='#E74C3C', lw=2, label='Internal Test (best)')
ax.plot(depths, depth_agg[('Ext Accuracy', 'max')], 's-', color='#3498DB', lw=2, label='External Valid (best)')
ax.plot(depths, depth_agg[('Test Accuracy', 'mean')], 'o--', color='#E74C3C', lw=1, alpha=0.5, label='Internal Test (mean)')
ax.plot(depths, depth_agg[('Ext Accuracy', 'mean')], 's--', color='#3498DB', lw=1, alpha=0.5, label='External Valid (mean)')
ax.set_xticks(depths)
ax.set_xticklabels(labels)
ax.set_xlabel('Max Depth')
ax.set_ylabel('Accuracy')
ax.set_title('(B) Accuracy vs. Decision Tree Depth')
ax.legend()
ax.grid(True, alpha=0.3)

fig.suptitle('Figure 7. Decision Tree: Effect of Max Depth on Performance',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(str(OUT / 'Fig7_DT_Depth_Performance.png'))
fig.savefig(str(OUT / 'Fig7_DT_Depth_Performance.pdf'))
plt.close()
print("  Fig 7: DT Depth Analysis ✓")

# ========== FIGURE 8: Model Comparison Bar Chart ==========
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax, results, title in [(axes[0], results_internal, 'Internal Test Set'),
                            (axes[1], results_external, 'External Validation')]:
    model_names = list(models.keys())
    metrics_bar = ['Accuracy', 'Sensitivity', 'Specificity', 'F1-Score', 'AUC-ROC']
    x = np.arange(len(model_names))
    width = 0.15
    for i, met in enumerate(metrics_bar):
        vals = [results[m][met] for m in model_names]
        ax.bar(x + i * width, vals, width, label=met, alpha=0.85)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(model_names, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(title)

fig.suptitle('Figure 8. Multi-Metric Comparison Across Classifiers',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(str(OUT / 'Fig8_Model_Comparison_Bar.png'))
fig.savefig(str(OUT / 'Fig8_Model_Comparison_Bar.pdf'))
plt.close()
print("  Fig 8: Model Comparison Bar ✓")

# ========== FIGURE 9: Calibration Curves ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, y_true, probas, title in [
    (axes[0], y_test, probas_internal, 'Internal Test Set'),
    (axes[1], y_ext, probas_external, 'External Validation Set')
]:
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfectly calibrated')
    for name in models:
        if probas[name] is not None:
            try:
                frac_pos, mean_pred = calibration_curve(y_true, probas[name], n_bins=8, strategy='uniform')
                brier = brier_score_loss(y_true, probas[name])
                ax.plot(mean_pred, frac_pos, 's-', color=COLORS[name], lw=1.5,
                        label=f'{name} (BS={brier:.3f})')
            except:
                pass
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title(title)
    ax.legend(loc='best', fontsize=7)
fig.suptitle('Figure 9. Calibration Curves (Reliability Diagrams)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(str(OUT / 'Fig9_Calibration_Curves.png'))
fig.savefig(str(OUT / 'Fig9_Calibration_Curves.pdf'))
plt.close()
print("  Fig 9: Calibration Curves ✓")

# ========== FIGURE 10: Feature Correlation Heatmap ==========
# Combine train data for correlation
df_corr = pd.DataFrame(X_train_imp, columns=all_features)
corr = df_corr.corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

fig, ax = plt.subplots(figsize=(18, 15))
sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.5, ax=ax,
            cbar_kws={"shrink": 0.8, "label": "Pearson r"},
            annot=False, fmt='.2f')
ax.set_title('Figure 10. Feature Correlation Matrix (Training Set)',
             fontsize=14, fontweight='bold')
ax.tick_params(axis='both', labelsize=7)
plt.xticks(rotation=90)
plt.tight_layout()
fig.savefig(str(OUT / 'Fig10_Correlation_Heatmap.png'))
fig.savefig(str(OUT / 'Fig10_Correlation_Heatmap.pdf'))
plt.close()
print("  Fig 10: Correlation Heatmap ✓")

# ========== FIGURE 11: Box Plot of Key Features by Group ==========
top_features_by_imp = imp_rf.tail(12).index.tolist()
df_box = pd.DataFrame(
    np.vstack([X_train_imp, X_test_imp, X_ext_imp]),
    columns=all_features
)
df_box['Group'] = np.concatenate([
    ['Positive' if y == 1 else 'Control' for y in y_train],
    ['Positive' if y == 1 else 'Control' for y in y_test],
    ['Positive' if y == 1 else 'Control' for y in y_ext],
])

n_feat = len(top_features_by_imp)
fig, axes = plt.subplots(3, 4, figsize=(18, 13))
for i, feat in enumerate(top_features_by_imp):
    ax = axes[i // 4, i % 4]
    data_ctrl = df_box[df_box['Group'] == 'Control'][feat].dropna()
    data_pos = df_box[df_box['Group'] == 'Positive'][feat].dropna()

    bp = ax.boxplot([data_ctrl, data_pos], labels=['Control', 'Positive'],
                    patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('#AED6F1')
    bp['boxes'][1].set_facecolor('#F5B7B1')

    # Statistical test
    stat, pval = stats.mannwhitneyu(data_ctrl, data_pos, alternative='two-sided')
    stars = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
    ax.set_title(f'{feat}\n(p={pval:.4f}) {stars}', fontsize=9)
    ax.tick_params(labelsize=8)

fig.suptitle('Figure 11. Distribution of Top-12 Features by Group (Mann-Whitney U Test)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(str(OUT / 'Fig11_Feature_BoxPlots.png'))
fig.savefig(str(OUT / 'Fig11_Feature_BoxPlots.pdf'))
plt.close()
print("  Fig 11: Feature Box Plots ✓")

# ========== FIGURE 12: DT Criterion & Leaf Size Heatmap ==========
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, criterion in [(axes[0], 'gini'), (axes[1], 'entropy')]:
    subset = df_dt[df_dt['criterion'] == criterion].copy()
    subset['max_depth_num'] = subset['max_depth'].apply(lambda x: 15 if x == 'None' else int(x))
    pivot = subset.groupby(['max_depth_num', 'min_samples_leaf'])['Test AUC'].mean().unstack()
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax, vmin=0.5, vmax=1.0)
    ax.set_xlabel('min_samples_leaf')
    ax.set_ylabel('max_depth')
    ax.set_title(f'Criterion: {criterion}')
    # Fix yticklabels
    yticklabels = [str(int(x)) if x < 15 else '∞' for x in sorted(subset['max_depth_num'].unique())]
    ax.set_yticklabels(yticklabels, rotation=0)

fig.suptitle('Figure 12. Decision Tree: AUC Heatmap (Depth × Leaf Size)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(str(OUT / 'Fig12_DT_Heatmap.png'))
fig.savefig(str(OUT / 'Fig12_DT_Heatmap.pdf'))
plt.close()
print("  Fig 12: DT Heatmap ✓")

# ========== FIGURE 13: Learning Curves ==========
from sklearn.model_selection import learning_curve

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
selected_models = {
    'Decision Tree': DecisionTreeClassifier(
        max_depth=int(best_dt_row['max_depth']) if best_dt_row['max_depth'] != 'None' else None,
        min_samples_split=int(best_dt_row['min_samples_split']),
        min_samples_leaf=int(best_dt_row['min_samples_leaf']),
        criterion=best_dt_row['criterion'], random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM (RBF)': SVC(probability=True, random_state=42),
    'MLP': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42, early_stopping=True),
}

for idx, (name, model) in enumerate(selected_models.items()):
    ax = axes[idx // 3, idx % 3]
    data = X_train_sc if name in needs_scaling else X_train_imp

    train_sizes, train_scores, val_scores = learning_curve(
        model, data, y_train, cv=5, scoring='roc_auc',
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1, random_state=42
    )
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='#E74C3C')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color='#3498DB')
    ax.plot(train_sizes, train_mean, 'o-', color='#E74C3C', lw=2, label='Training')
    ax.plot(train_sizes, val_mean, 's-', color='#3498DB', lw=2, label='Validation')
    ax.set_xlabel('Training Samples')
    ax.set_ylabel('AUC-ROC')
    ax.set_title(name)
    ax.legend(loc='lower right', fontsize=8)
    ax.set_ylim(0.4, 1.05)
    ax.grid(True, alpha=0.3)

fig.suptitle('Figure 13. Learning Curves (5-Fold Cross-Validation)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(str(OUT / 'Fig13_Learning_Curves.png'))
fig.savefig(str(OUT / 'Fig13_Learning_Curves.pdf'))
plt.close()
print("  Fig 13: Learning Curves ✓")

# ========== FIGURE 14: Feature Category Importance ==========
# Group features by category
category_groups = {
    'Demographics': demo_features,
    'Image (Back)': [f for f in all_features if f.startswith('Back_')],
    'Image (Frontal)': [f for f in all_features if f.startswith('Front_')],
    'Image (Lateral)': [f for f in all_features if f.startswith('Lat_')],
    'Video (Roll L)': [f for f in all_features if f.startswith('RollL_')],
    'Video (Roll R)': [f for f in all_features if f.startswith('RollR_')],
    'Video (Sit→Sup)': [f for f in all_features if f.startswith('SitSup_')],
    'Video (Sup→Sit)': [f for f in all_features if f.startswith('SupSit_')],
}

cat_importance_rf = {}
cat_importance_gb = {}
for cat, feats in category_groups.items():
    cat_importance_rf[cat] = sum(imp_rf.get(f, 0) for f in feats)
    cat_importance_gb[cat] = sum(imp_gb.get(f, 0) for f in feats)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
cats = list(category_groups.keys())

ax = axes[0]
vals = [cat_importance_rf[c] for c in cats]
colors = plt.cm.Set2(np.linspace(0, 1, len(cats)))
bars = ax.barh(cats, vals, color=colors, edgecolor='gray')
ax.set_xlabel('Cumulative Importance')
ax.set_title('(A) Random Forest')

ax = axes[1]
vals = [cat_importance_gb[c] for c in cats]
bars = ax.barh(cats, vals, color=colors, edgecolor='gray')
ax.set_xlabel('Cumulative Importance')
ax.set_title('(B) Gradient Boosting')

fig.suptitle('Figure 14. Feature Category Importance',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(str(OUT / 'Fig14_Category_Importance.png'))
fig.savefig(str(OUT / 'Fig14_Category_Importance.pdf'))
plt.close()
print("  Fig 14: Category Importance ✓")

# ========== FIGURE 15: Cross-Validation AUC Distribution ==========
fig, ax = plt.subplots(figsize=(12, 6))

cv_results = {}
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
for name, model in models.items():
    data = X_train_sc if name in needs_scaling else X_train_imp
    fold_aucs = []
    for tr_idx, val_idx in cv.split(data, y_train):
        model.fit(data[tr_idx], y_train[tr_idx])
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(data[val_idx])[:, 1]
        else:
            prob = model.decision_function(data[val_idx])
        fold_aucs.append(roc_auc_score(y_train[val_idx], prob))
    cv_results[name] = fold_aucs

bp = ax.boxplot([cv_results[n] for n in models],
                labels=list(models.keys()),
                patch_artist=True, widths=0.5)
for i, (patch, name) in enumerate(zip(bp['boxes'], models)):
    patch.set_facecolor(COLORS[name])
    patch.set_alpha(0.7)

ax.set_ylabel('AUC-ROC')
ax.set_title('Figure 15. 10-Fold Cross-Validation AUC Distribution',
             fontsize=14, fontweight='bold')
ax.tick_params(axis='x', rotation=30)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(str(OUT / 'Fig15_CV_AUC_BoxPlot.png'))
fig.savefig(str(OUT / 'Fig15_CV_AUC_BoxPlot.pdf'))
plt.close()
print("  Fig 15: CV AUC BoxPlot ✓")

# ========== FIGURE 16: DT Feature Importance with Error Bars ==========
# Use multiple random seeds to get importance stability
dt_importances_list = []
for seed in range(50):
    dt_temp = DecisionTreeClassifier(
        max_depth=int(best_dt_row['max_depth']) if best_dt_row['max_depth'] != 'None' else None,
        min_samples_split=int(best_dt_row['min_samples_split']),
        min_samples_leaf=int(best_dt_row['min_samples_leaf']),
        criterion=best_dt_row['criterion'],
        random_state=seed
    )
    # Bootstrap sample
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X_train_imp), len(X_train_imp), replace=True)
    dt_temp.fit(X_train_imp[idx], y_train[idx])
    dt_importances_list.append(dt_temp.feature_importances_)

dt_imp_mean = np.mean(dt_importances_list, axis=0)
dt_imp_std = np.std(dt_importances_list, axis=0)
dt_imp_df = pd.DataFrame({'feature': all_features, 'mean': dt_imp_mean, 'std': dt_imp_std})
dt_imp_df = dt_imp_df.sort_values('mean', ascending=True).tail(20)

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(range(len(dt_imp_df)), dt_imp_df['mean'].values, xerr=dt_imp_df['std'].values,
        color='#E74C3C', edgecolor='#C0392B', capsize=3, alpha=0.85)
ax.set_yticks(range(len(dt_imp_df)))
ax.set_yticklabels(dt_imp_df['feature'].values)
ax.set_xlabel('Importance (Gini/Entropy)')
ax.set_title('Figure 16. Decision Tree Feature Importance\n(Bootstrap Stability, 50 iterations)',
             fontsize=14, fontweight='bold')
ax.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
fig.savefig(str(OUT / 'Fig16_DT_Feature_Importance.png'))
fig.savefig(str(OUT / 'Fig16_DT_Feature_Importance.pdf'))
plt.close()
print("  Fig 16: DT Feature Importance ✓")

# ========== FIGURE 17: AUC Comparison Forest Plot ==========
fig, ax = plt.subplots(figsize=(10, 7))

y_pos = range(len(models))
model_list = list(models.keys())
for i, name in enumerate(model_list):
    auc_int = results_internal[name]['AUC-ROC']
    auc_ext = results_external[name]['AUC-ROC']
    ci_str = results_internal[name]['AUC 95% CI']
    ci_low, ci_high = [float(x) for x in ci_str.strip('()').split('-')]

    ax.errorbar(auc_int, i - 0.12, xerr=[[auc_int - ci_low], [ci_high - auc_int]],
                fmt='o', color='#E74C3C', markersize=8, capsize=5, lw=2, label='Internal' if i == 0 else '')
    ax.plot(auc_ext, i + 0.12, 's', color='#3498DB', markersize=8, label='External' if i == 0 else '')

ax.set_yticks(y_pos)
ax.set_yticklabels(model_list)
ax.set_xlabel('AUC-ROC')
ax.axvline(x=0.5, color='gray', ls='--', lw=1)
ax.set_title('Figure 17. Forest Plot: AUC-ROC with 95% CI',
             fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.set_xlim(0.3, 1.05)
ax.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
fig.savefig(str(OUT / 'Fig17_AUC_Forest_Plot.png'))
fig.savefig(str(OUT / 'Fig17_AUC_Forest_Plot.pdf'))
plt.close()
print("  Fig 17: AUC Forest Plot ✓")

print("\n" + "=" * 70)
print("ALL DONE! Files saved to /mnt/user-data/outputs/")
print("=" * 70)

# List all output files
for f in sorted(OUT.glob('Fig*')):
    print(f"  {f.name}")
print(f"  ML_Results_Tables.xlsx")
