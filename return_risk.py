from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
 
ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
MODELS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)
 
# ---------- helpers ----------
def classification_metrics(y_true, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
    }
 
def sweep_thresholds(y_true, prob, step=0.02):
    rows = []
    for t in np.arange(0.10, 0.9001, step):
        m = classification_metrics(y_true, prob, float(np.round(t, 2)))
        rows.append(m)
    out = pd.DataFrame(rows)
    best = out.loc[out["f1"].idxmax()].to_dict()
    return out, best
 
def subgroup_report(frame, y_true, pred, group_col):
    tmp = frame[[group_col]].copy().reset_index(drop=True)
    tmp["y_true"] = np.asarray(y_true)
    tmp["y_pred"] = np.asarray(pred)
    rows = []
    for group, g in tmp.groupby(group_col):
        rows.append({
            "group_type": group_col,
            "group": str(group),
            "n": int(len(g)),
            "precision": float(precision_score(g.y_true, g.y_pred, zero_division=0)),
            "recall": float(recall_score(g.y_true, g.y_pred, zero_division=0)),
        })
    return pd.DataFrame(rows)
 
# ---------- data ----------
df = pd.read_csv(ROOT / "orders_dataset.csv")
print("Shape:", df.shape)
print("Overall return rate:", df.returned.mean())
print("Missing rating_given %:", 100 * df.rating_given.isna().mean())
print("\nReturn rate by category:\n", df.groupby("product_category").returned.mean())
print("\nReturn rate by payment:\n", df.groupby("payment_method").returned.mean())
 
cod_missing = df.loc[df.payment_method.eq("COD"), "rating_given"].isna().mean()
noncod_missing = df.loc[~df.payment_method.eq("COD"), "rating_given"].isna().mean()
print(f"COD missing rating rate: {cod_missing:.4f}")
print(f"Non-COD missing rating rate: {noncod_missing:.4f}")
print("Missingness classification: MAR, because missingness depends on observed payment_method.")
 
# Do not use order_id as a predictive feature.
TARGET = "returned"
DROP_COLS = ["order_id", TARGET]
X = df.drop(columns=DROP_COLS)
y = df[TARGET]
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
 
categorical = ["product_category", "payment_method"]
numeric = [c for c in X.columns if c not in categorical]
 
numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric),
    ("cat", cat_pipe, categorical),
])
 
# ---------- Dummy baseline ----------
dummy = Pipeline([
    ("prep", preprocessor),
    ("model", DummyClassifier(strategy="most_frequent")),
])
dummy.fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)
dummy_metrics = {
    "accuracy": float(accuracy_score(y_test, dummy_pred)),
    "precision": float(precision_score(y_test, dummy_pred, zero_division=0)),
    "recall": float(recall_score(y_test, dummy_pred, zero_division=0)),
    "f1": float(f1_score(y_test, dummy_pred, zero_division=0)),
}
print("\nDummy:", dummy_metrics)
print("Interpretation: high accuracy with zero recall is misleading for return detection.")
 
# ---------- Logistic Regression ----------
logreg = Pipeline([
    ("prep", preprocessor),
    ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)),
])
logreg.fit(X_train, y_train)
log_prob = logreg.predict_proba(X_test)[:, 1]
log_default = classification_metrics(y_test, log_prob, 0.5)
log_sweep, log_best = sweep_thresholds(y_test, log_prob)
log_sweep.to_csv(RESULTS / "logreg_thresholds.csv", index=False)
print("\nLogistic @ 0.5:", log_default)
print("Best Logistic threshold:", log_best)
print("Business trade-off: lowering threshold generally raises recall but accepts more false positives and lower precision.")
 
# ---------- Random Forest Grid Search ----------
rf_pipe = Pipeline([
    ("prep", preprocessor),
    ("model", RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)),
])
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [6, 10, None],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(
    rf_pipe, param_grid=param_grid, scoring="roc_auc", cv=cv,
    n_jobs=-1, refit=True, return_train_score=False
)
grid.fit(X_train, y_train)
best_rf = grid.best_estimator_
rf_prob = best_rf.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_prob)
print("\nRF best params:", grid.best_params_)
print("RF best CV ROC-AUC:", grid.best_score_)
print("RF test ROC-AUC:", rf_auc)
 
# ---------- RF threshold sweep (THIS is t*_rf used by Part 3) ----------
rf_sweep, rf_best = sweep_thresholds(y_test, rf_prob)
rf_sweep.to_csv(RESULTS / "part1_thresholds.csv", index=False)
t_rf = float(rf_best["threshold"])
rf_pred = (rf_prob >= t_rf).astype(int)
print("RF F1-max threshold t*_rf:", t_rf)
print("RF metrics at t*_rf:", rf_best)
 
# ---------- feature names + impurity importance ----------
prep = best_rf.named_steps["prep"]
model = best_rf.named_steps["model"]
feature_names = prep.get_feature_names_out()
imp_df = pd.DataFrame({
    "feature": feature_names,
    "impurity_importance": model.feature_importances_,
}).sort_values("impurity_importance", ascending=False)
 
# Permutation importance on transformed held-out X so ranking is comparable at encoded-feature level.
X_test_t = prep.transform(X_test)
perm = permutation_importance(
    model, X_test_t, y_test, scoring="roc_auc",
    n_repeats=10, random_state=42, n_jobs=-1
)
perm_df = pd.DataFrame({
    "feature": feature_names,
    "permutation_importance": perm.importances_mean,
})
importance_compare = imp_df.merge(perm_df, on="feature")
importance_compare.to_csv(RESULTS / "feature_importance.csv", index=False)
print("\nTop 5 impurity features:\n", importance_compare.head(5))
 
# ---------- subgroup performance ----------
sub_cat = subgroup_report(X_test, y_test, rf_pred, "product_category")
sub_pay = subgroup_report(X_test, y_test, rf_pred, "payment_method")
subgroups = pd.concat([sub_cat, sub_pay], ignore_index=True)
subgroups.to_csv(RESULTS / "subgroup_metrics.csv", index=False)
print("\nSubgroup metrics:\n", subgroups)
 
# ---------- save final artifacts ----------
joblib.dump(best_rf, MODELS / "return_risk_model.pkl")
with open(MODELS / "return_risk_threshold.json", "w") as f:
    json.dump({
        "t_rf": t_rf,
        "low_if": f"p < {t_rf:.4f}",
        "medium_if": f"{t_rf:.4f} <= p < {min(1.0, t_rf + 0.15):.4f}",
        "high_if": f"p >= {min(1.0, t_rf + 0.15):.4f}",
    }, f, indent=2)
 
summary = {
    "rows": int(len(df)),
    "columns": int(df.shape[1]),
    "overall_return_rate": float(df.returned.mean()),
    "rating_missing_rate": float(df.rating_given.isna().mean()),
    "cod_rating_missing_rate": float(cod_missing),
    "noncod_rating_missing_rate": float(noncod_missing),
    "missingness": "MAR",
    "dummy": dummy_metrics,
    "logistic_default": log_default,
    "logistic_best_threshold": log_best,
    "rf_best_params": grid.best_params_,
    "rf_cv_roc_auc": float(grid.best_score_),
    "rf_test_roc_auc": float(rf_auc),
    "rf_best_threshold": rf_best,
}
with open(RESULTS / "part1_metrics.json", "w") as f:
    json.dump(summary, f, indent=2)
 
print("\nSaved:")
print(MODELS / "return_risk_model.pkl")
print(MODELS / "return_risk_threshold.json")
print(RESULTS / "part1_metrics.json")