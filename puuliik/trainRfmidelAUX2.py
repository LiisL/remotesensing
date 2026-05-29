import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay

# =========================================================
# SISENDFAILID
# =========================================================

TRAIN_FILE = r"C:\Users\liisl\Documents\Kaugseire\Puuliik\DATA\eelmisedkatsed\Subregion2_train_timeseries1704w2.csv"
VAL_FILE   = r"C:\Users\liisl\Documents\Kaugseire\Puuliik\DATA\eelmisedkatsed\Subregion2_val_timeseries1704w2.csv"
TEST_FILE  = r"C:\Users\liisl\Documents\Kaugseire\Puuliik\DATA\eelmisedkatsed\Subregion2_test_timeseries1704w2.csv"

# Kasuta parandatud AUX faili
AUX_FILE = r"C:\Users\liisl\Documents\Kaugseire\Puuliik\DATA\stand_aux_features.csv"

MODEL_OUT = r"C:\Users\liisl\Documents\Kaugseire\Puuliik\DATA\Subregion2_rf_model_with_aux0405.joblib"
FEATURE_IMPORTANCE_OUT = r"C:\Users\liisl\Documents\Kaugseire\Puuliik\DATA\Subregion2_feature_importance_with_aux0405.csv"
VAL_CM_OUT = r"C:\Users\liisl\Documents\Kaugseire\Puuliik\DATA\Subregion2_confusion_matrix_val_with_aux0405.png"
TEST_CM_OUT = r"C:\Users\liisl\Documents\Kaugseire\Puuliik\DATA\Subregion2_confusion_matrix_test_with_aux0405.png"

# =========================================================
# SEADISTUS
# =========================================================

ID_COL = "id"
TARGET_COL = "target"
DATE_COL = "date_from"

TS_FEATURE_COLS = [
    "B2_mean", "B3_mean", "B4_mean", "B5_mean", "B6_mean", "B7_mean",
    "B8_mean", "B8A_mean", "B11_mean", "B12_mean",
    "NDVI_mean", "NDRE_mean", "NDMI_mean", "NBR_mean"
]

# lisa elev_median
AUX_NUMERIC_COLS = [
    "elev_mean", "elev_median", "elev_std", "elev_min", "elev_max"
]

AUX_CATEGORICAL_COLS = ["soil_main"]

# =========================================================
# ABIFUNKTSIOONID
# =========================================================

def load_timeseries(path):
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Tühi fail: {path}")
    return df


def load_aux(path):
    aux = pd.read_csv(path)
    if aux.empty:
        raise ValueError(f"Tühi aux fail: {path}")
    return aux


def long_to_wide(df, dataset_name="data"):
    needed = [ID_COL, TARGET_COL, DATE_COL] + TS_FEATURE_COLS
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name}: puuduvad veerud {missing}")

    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL]).dt.strftime("%Y_%m_%d")

    labels = df[[ID_COL, TARGET_COL]].drop_duplicates()

    wide = df.pivot_table(
        index=ID_COL,
        columns=DATE_COL,
        values=TS_FEATURE_COLS,
        aggfunc="first"
    )

    wide.columns = [f"{feat}_{date}" for feat, date in wide.columns]
    wide = wide.reset_index()

    wide = wide.merge(labels, on=ID_COL, how="left")

    return wide


def merge_aux_features(wide_df, aux_df, dataset_name="data"):
    keep_cols = [ID_COL, TARGET_COL] + AUX_NUMERIC_COLS + AUX_CATEGORICAL_COLS
    missing = [c for c in keep_cols if c not in aux_df.columns]
    if missing:
        raise ValueError(f"{dataset_name}: aux failist puuduvad veerud {missing}")

    aux_sub = aux_df[keep_cols].drop_duplicates(subset=[ID_COL]).copy()

    merged = wide_df.merge(
        aux_sub.drop(columns=[TARGET_COL]),
        on=ID_COL,
        how="left"
    )

    print(f"\nAUX merge kontroll ({dataset_name}):")
    for col in AUX_NUMERIC_COLS + AUX_CATEGORICAL_COLS:
        missing_count = merged[col].isna().sum()
        print(f"  {col}: puudu {missing_count}")

    return merged


def encode_categorical_consistently(train_df, val_df, test_df):
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["__source__"] = "train"
    val_df["__source__"] = "val"
    test_df["__source__"] = "test"

    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)

    for col in AUX_CATEGORICAL_COLS:
        combined[col] = combined[col].fillna("MISSING").astype(str)

    combined = pd.get_dummies(combined, columns=AUX_CATEGORICAL_COLS, prefix=AUX_CATEGORICAL_COLS)

    train_out = combined[combined["__source__"] == "train"].drop(columns="__source__").copy()
    val_out   = combined[combined["__source__"] == "val"].drop(columns="__source__").copy()
    test_out  = combined[combined["__source__"] == "test"].drop(columns="__source__").copy()

    return train_out, val_out, test_out


def align_feature_columns(train_df, val_df, test_df):
    feature_cols_train = [c for c in train_df.columns if c not in [ID_COL, TARGET_COL]]

    for df in [val_df, test_df]:
        for col in feature_cols_train:
            if col not in df.columns:
                df[col] = np.nan

    train_df = train_df[[ID_COL] + feature_cols_train + [TARGET_COL]]
    val_df   = val_df[[ID_COL] + feature_cols_train + [TARGET_COL]]
    test_df  = test_df[[ID_COL] + feature_cols_train + [TARGET_COL]]

    return train_df, val_df, test_df, feature_cols_train


def print_metrics(title, y_true, y_pred):
    print(f"\n================ {title} ================")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, y_pred))


def save_confusion_matrix(y_true, y_pred, title, out_png):
    labels = sorted(pd.Series(y_true).astype(str).unique().tolist())
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix salvestatud: {out_png}")


# =========================================================
# 1. LOE FAILID
# =========================================================

train_ts = load_timeseries(TRAIN_FILE)
val_ts   = load_timeseries(VAL_FILE)
test_ts  = load_timeseries(TEST_FILE)
aux_df   = load_aux(AUX_FILE)

print("Long-form read:")
print("train:", train_ts.shape)
print("val  :", val_ts.shape)
print("test :", test_ts.shape)
print("aux  :", aux_df.shape)

# =========================================================
# 2. WIDE-FORM TIMESERIES
# =========================================================

train_wide = long_to_wide(train_ts, "train")
val_wide   = long_to_wide(val_ts, "val")
test_wide  = long_to_wide(test_ts, "test")

# =========================================================
# 3. LIIDA AUX FEATURE’ID
# =========================================================

train_wide = merge_aux_features(train_wide, aux_df, "train")
val_wide   = merge_aux_features(val_wide, aux_df, "val")
test_wide  = merge_aux_features(test_wide, aux_df, "test")

train_wide, val_wide, test_wide = encode_categorical_consistently(train_wide, val_wide, test_wide)
train_wide, val_wide, test_wide, feature_cols = align_feature_columns(train_wide, val_wide, test_wide)

print("\nWide-form + aux:")
print("train:", train_wide.shape)
print("val  :", val_wide.shape)
print("test :", test_wide.shape)

# =========================================================
# 4. X / y
# =========================================================

X_train = train_wide[feature_cols].copy()
y_train = train_wide[TARGET_COL].copy()

X_val = val_wide[feature_cols].copy()
y_val = val_wide[TARGET_COL].copy()

X_test = test_wide[feature_cols].copy()
y_test = test_wide[TARGET_COL].copy()

# =========================================================
# 5. TRAIN -> VAL
# =========================================================

imputer = SimpleImputer(strategy="median")

X_train_imp = imputer.fit_transform(X_train)
X_val_imp = imputer.transform(X_val)
X_test_imp = imputer.transform(X_test)

rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train_imp, y_train)

val_pred = rf.predict(X_val_imp)
print_metrics("VALIDATION (with corrected elevation + soil, trained on TRAIN)", y_val, val_pred)
save_confusion_matrix(y_val, val_pred, "Validation Confusion Matrix - Corrected Aux", VAL_CM_OUT)

# =========================================================
# 6. RETRAIN TRAIN+VAL -> TEST
# =========================================================

X_trainval = pd.concat([X_train, X_val], axis=0)
y_trainval = pd.concat([y_train, y_val], axis=0)

imputer_final = SimpleImputer(strategy="median")
X_trainval_imp = imputer_final.fit_transform(X_trainval)
X_test_imp_final = imputer_final.transform(X_test)

rf_final = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_final.fit(X_trainval_imp, y_trainval)

test_pred = rf_final.predict(X_test_imp_final)
print_metrics("TEST (with corrected elevation + soil, trained on TRAIN+VAL)", y_test, test_pred)
save_confusion_matrix(y_test, test_pred, "Test Confusion Matrix - Corrected Aux", TEST_CM_OUT)

# =========================================================
# 7. FEATURE IMPORTANCE
# =========================================================

importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": rf_final.feature_importances_
}).sort_values("importance", ascending=False)

print("\n================ FEATURE IMPORTANCE ================")
print(importance_df.head(50))

importance_df.to_csv(FEATURE_IMPORTANCE_OUT, index=False, encoding="utf-8-sig")
print(f"\nFeature importance salvestatud: {FEATURE_IMPORTANCE_OUT}")

# =========================================================
# 8. SALVESTA MUDEL
# =========================================================

joblib.dump(
    {
        "model": rf_final,
        "imputer": imputer_final,
        "feature_cols": feature_cols,
        "id_col": ID_COL,
        "target_col": TARGET_COL
    },
    MODEL_OUT
)

print(f"\nMudeli fail salvestatud: {MODEL_OUT}")