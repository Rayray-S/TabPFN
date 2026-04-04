import os
import time
import joblib
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from tabpfn.finetuning import FinetunedTabPFNClassifier

warnings.filterwarnings("ignore")

WANDB_PROJECT = "tabpfn-finetuning"
WANDB_RUN_NAME = None
WANDB_MODE = "online"  # online / offline / disabled
WANDB_TAGS = ["fraud", "tabpfn", "finetuning"]

# ========================= MOD-4 =========================
# 减少 CUDA 内存碎片导致的 OOM 概率
# =========================================================
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ==========================================
# 1. 全局配置
# ==========================================
DATA_FILE = "/root/autodl-fs/data-based/data/data_with_expert_features.csv"
SAVE_MODEL_DIR = "/root/autodl-fs/data-based/TabMethod/model/tabpfn_finetuned"
os.makedirs(SAVE_MODEL_DIR, exist_ok=True)

TARGET_COL = "misstate"
YEAR_COL = "fyear"
FINETUNE_YEAR_END = 2002
DROP_COLS = [TARGET_COL, YEAR_COL, "gvkey", "datadate", "p_aaer"]

# ========================= MOD-2 =========================
# 验证集最大样本数（早停监控用，不必全量）
# 建议先 2000，若仍 OOM 再降到 1000
# =========================================================
MAX_VAL_SAMPLES_FOR_EVAL = 1000


def subsample_val_keep_all_pos(X_val, y_val, max_samples=2000, seed=42):
    y_val = np.asarray(y_val).astype(int)
    n = len(y_val)
    if n <= max_samples:
        return X_val, y_val

    rng = np.random.default_rng(seed)
    pos_idx = np.where(y_val == 1)[0]
    neg_idx = np.where(y_val == 0)[0]

    if len(pos_idx) >= max_samples:
        keep_idx = rng.choice(pos_idx, size=max_samples, replace=False)
    else:
        n_neg_keep = max_samples - len(pos_idx)
        n_neg_keep = min(n_neg_keep, len(neg_idx))
        neg_keep = (
            rng.choice(neg_idx, size=n_neg_keep, replace=False)
            if n_neg_keep > 0
            else np.array([], dtype=int)
        )
        keep_idx = np.concatenate([pos_idx, neg_keep])

    rng.shuffle(keep_idx)
    return X_val[keep_idx], y_val[keep_idx]


# ==========================================
# 2. 数据准备与预处理
# ==========================================
def load_and_prepare_data():
    print(f"Loading data from: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE, low_memory=False)

    df_finetune = df[df[YEAR_COL] <= FINETUNE_YEAR_END].copy()
    df_finetune = df_finetune.dropna(subset=[TARGET_COL, YEAR_COL])

    print(f"Finetuning data spans years: {sorted(df_finetune[YEAR_COL].unique())}")
    print(f"Total samples for finetuning: {len(df_finetune)}")
    print(f"Fraud samples: {int(df_finetune[TARGET_COL].sum())}")

    feature_cols = [c for c in df_finetune.columns if c not in DROP_COLS]

    kept_cols = []
    for col in feature_cols:
        if df_finetune[col].nunique(dropna=True) > 1:
            kept_cols.append(col)

    X = df_finetune[kept_cols].copy()
    y = df_finetune[TARGET_COL].astype(int).values

    for col in X.columns:
        if pd.api.types.is_object_dtype(X[col]) or pd.api.types.is_categorical_dtype(X[col]):
            X[col] = X[col].astype("category")

    X = np.ascontiguousarray(np.asarray(X, dtype=np.float32))

    val_mask = df_finetune[YEAR_COL] == FINETUNE_YEAR_END
    X_train, y_train = X[~val_mask], y[~val_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    # ========================= MOD-2 =========================
    # 只对 early-stopping 的验证集做下采样，降低 eval OOM 风险
    # =========================================================
    X_val_small, y_val_small = subsample_val_keep_all_pos(
        X_val, y_val, max_samples=MAX_VAL_SAMPLES_FOR_EVAL, seed=42
    )

    print(
        f"Train size: {len(y_train)}, Val size(full): {len(y_val)}, Val size(used): {len(y_val_small)}"
    )
    print(f"Val fraud(full): {int(y_val.sum())}, Val fraud(used): {int(y_val_small.sum())}")

    return X_train, y_train, X_val_small, y_val_small, kept_cols


# ==========================================
# 3. 执行微调
# ==========================================
def main():
    X_train, y_train, X_val, y_val, feature_names = load_and_prepare_data()

    wandb_config = {
        "data_file": DATA_FILE,
        "save_model_dir": SAVE_MODEL_DIR,
        "target_col": TARGET_COL,
        "year_col": YEAR_COL,
        "finetune_year_end": FINETUNE_YEAR_END,
        "drop_cols": DROP_COLS,
        "max_val_samples_for_eval": MAX_VAL_SAMPLES_FOR_EVAL,
        "train_size": int(len(y_train)),
        "val_size": int(len(y_val)),
        "train_positive_count": int(y_train.sum()),
        "val_positive_count": int(y_val.sum()),
        "n_features": int(len(feature_names)),
    }

    clf = FinetunedTabPFNClassifier(
        device="cuda",
        epochs=40,
        learning_rate=1e-5,
        weight_decay=0.01,
        early_stopping=True,
        early_stopping_patience=5,
        validation_split_ratio=0.0,
        n_estimators_finetune=1,
        n_estimators_validation=1,

        # ========================= MOD-1 =========================
        # 验证/推理 eval 配置的 subsample 上限，默认 50_000 太大
        # 先试 5000；仍 OOM 可降到 3000/2000
        # =========================================================
        n_inference_subsample_samples=3000,
        n_finetune_ctx_plus_query_samples=2000,

        # ========================= MOD-3 =========================
        # 训练阶段先用较小推理 estimator，省显存
        # 最终部署时可另开脚本设大一点
        # =========================================================
        n_estimators_final_inference=1,
        optimizer_name="adamw",    # 可试: adam / adamw / rmsprop
        loss_type="weighted_ce",   # 可试: ce / weighted_ce / focal
        focal_gamma=2.0,           # focal 时生效
        weight_clamp_max=20.0,
        eval_metric="roc_auc",
        use_lr_scheduler=True,
        grad_clip_value=1.0,
        wandb_project=WANDB_PROJECT,
        wandb_run_name=WANDB_RUN_NAME,
        wandb_tags=WANDB_TAGS,
        wandb_config=wandb_config,
        wandb_mode=WANDB_MODE,
        extra_classifier_kwargs={
            "model_path": "/root/autodl-fs/data-based/TabMethod/model/tabpfn/tabpfn-v2.5-classifier-v2.5_default.ckpt"
        },
    )

    print("\nStarting TabPFN fine-tuning...")
    start_time = time.time()

    ckpt_dir = os.path.join(SAVE_MODEL_DIR, "checkpoints")
    clf.fit(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        output_dir=Path(ckpt_dir),
    )

    print(f"\nFine-tuning completed in {(time.time() - start_time)/60:.2f} minutes.")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    final_model_path = os.path.join(SAVE_MODEL_DIR, f"tabpfn_finetuned5_{timestamp}.joblib")

    model_bundle = {
        "model": clf,
        "features": feature_names,
        "finetune_year_end": FINETUNE_YEAR_END,
        "wandb": {
            "project": WANDB_PROJECT,
            "run_name": WANDB_RUN_NAME,
            "mode": WANDB_MODE,
            "tags": WANDB_TAGS,
            "config": wandb_config,
        },
    }
    joblib.dump(model_bundle, final_model_path)
    print(f"Final fine-tuned model bundle saved at:\n{final_model_path}")


if __name__ == "__main__":
    main()
