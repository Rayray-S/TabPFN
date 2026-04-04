from pathlib import Path

path = Path(r"d:\Desktop\GradProject\project\TabPFN\TabPFN-tuning.py")
text = path.read_text(encoding="utf-8")

text = text.replace(
    'warnings.filterwarnings("ignore")\n',
    'warnings.filterwarnings("ignore")\n\nWANDB_PROJECT = "tabpfn-finetuning"\nWANDB_RUN_NAME = None\nWANDB_MODE = "online"  # online / offline / disabled\nWANDB_TAGS = ["fraud", "tabpfn", "finetuning"]\n',
    1,
)

text = text.replace(
    'def main():\n    X_train, y_train, X_val, y_val, feature_names = load_and_prepare_data()\n\n    clf = FinetunedTabPFNClassifier(\n',
    'def main():\n    X_train, y_train, X_val, y_val, feature_names = load_and_prepare_data()\n\n    wandb_config = {\n        "data_file": DATA_FILE,\n        "save_model_dir": SAVE_MODEL_DIR,\n        "target_col": TARGET_COL,\n        "year_col": YEAR_COL,\n        "finetune_year_end": FINETUNE_YEAR_END,\n        "drop_cols": DROP_COLS,\n        "max_val_samples_for_eval": MAX_VAL_SAMPLES_FOR_EVAL,\n        "train_size": int(len(y_train)),\n        "val_size": int(len(y_val)),\n        "train_positive_count": int(y_train.sum()),\n        "val_positive_count": int(y_val.sum()),\n        "n_features": int(len(feature_names)),\n    }\n\n    clf = FinetunedTabPFNClassifier(\n',
    1,
)

text = text.replace(
    '        eval_metric="roc_auc",\n        use_lr_scheduler=True,\n        grad_clip_value=1.0,\n\n        extra_classifier_kwargs={\n',
    '        eval_metric="roc_auc",\n        use_lr_scheduler=True,\n        grad_clip_value=1.0,\n\n        wandb_project=WANDB_PROJECT,\n        wandb_run_name=WANDB_RUN_NAME,\n        wandb_tags=WANDB_TAGS,\n        wandb_config=wandb_config,\n        wandb_mode=WANDB_MODE,\n\n        extra_classifier_kwargs={\n',
    1,
)

text = text.replace(
    '    model_bundle = {\n        "model": clf,\n        "features": feature_names,\n        "finetune_year_end": FINETUNE_YEAR_END,\n    }\n',
    '    model_bundle = {\n        "model": clf,\n        "features": feature_names,\n        "finetune_year_end": FINETUNE_YEAR_END,\n        "wandb": {\n            "project": WANDB_PROJECT,\n            "run_name": WANDB_RUN_NAME,\n            "mode": WANDB_MODE,\n            "tags": WANDB_TAGS,\n            "config": wandb_config,\n        },\n    }\n',
    1,
)

path.write_text(text, encoding="utf-8")
