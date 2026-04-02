# TabPFN 代码解读报告（src/tabpfn 自动扫描版）

说明：每个文件列出模块级函数/类；类中列出类内方法名（包含 @staticmethod/@classmethod 定义在同一函数节点的函数名）。为便于修改模型细节，建议先关注 architecture、encoder steps、inference engines、preprocessing/torch preprocessing。

## src/tabpfn\__init__.py
用途：—
模块级函数：—
类：—

## src/tabpfn\architectures\__init__.py
用途：Contains a collection of different model architectures.
模块级函数：register_architecture
类：—

## src/tabpfn\architectures\base\__init__.py
用途：The base architecture.
模块级函数：parse_config, get_architecture, get_encoder, get_y_encoder, _legacy_normalize_features_no_op
类：—

## src/tabpfn\architectures\base\attention\__init__.py
用途：—
模块级函数：—
类：
- Attention（方法：forward）

## src/tabpfn\architectures\base\attention\full_attention.py
用途：Implements standard quadratic attention.
模块级函数：—
类：
- MultiHeadAttention（方法：w_q, w_k, w_v, w_qkv, w_kv, w_out, has_cached_kv, empty_kv_cache, set_parameters, newly_initialized_input_weight, __init__, forward, compute_qkv, _compute, _rearrange_inputs_to_flat_batch, broadcast_kv_across_heads, scaled_dot_product_attention_chunked, compute_attention_heads, convert_torch_nn_multihead_attention_state_dict）

## src/tabpfn\architectures\base\bar_distribution.py
用途：—
模块级函数：get_bucket_limits
类：
- BarDistribution（方法：__init__, has_equal_borders, bucket_widths, num_bars, cdf, get_probs_for_different_borders, average_bar_distributions_into_this, __setstate__, map_to_bucket_idx, ignore_init, compute_scaled_log_probs, full_ce, forward, mean_loss, mean, median, cdf_temporary, icdf, quantile, ucb, mode, ei, pi, mean_of_square, variance, plot）
- FullSupportBarDistribution（方法：__init__, assert_support, halfnormal_with_p_weight_before, forward, pdf, sample, mean, mean_of_square, pi, ei_for_halfnormal, ei）

## src/tabpfn\architectures\base\config.py
用途：—
模块级函数：—
类：
- ModelConfig（方法：upgrade_config, _get_default, validate_consistent）

## src/tabpfn\architectures\base\layer.py
用途：—
模块级函数：_get_feature_attn_constructor, _get_item_attn_constructor
类：
- LayerNorm（方法：__init__, _compute, forward）
- PerFeatureEncoderLayer（方法：__init__, __setstate__, forward, empty_trainset_representation_cache）

## src/tabpfn\architectures\base\memory.py
用途：—
模块级函数：support_save_peak_mem_factor, should_save_peak_mem, _should_save_peak_mem_cpu, _should_save_peak_mem_cuda, _get_free_cuda_memory_bytes, _get_num_cells
类：—

## src/tabpfn\architectures\base\mlp.py
用途：—
模块级函数：—
类：
- Activation（方法：—）
- MLP（方法：__init__, _compute, forward）

## src/tabpfn\architectures\base\thinking_tokens.py
用途：—
模块级函数：—
类：
- AddThinkingTokens（方法：__init__, forward, reset_parameters）

## src/tabpfn\architectures\base\transformer.py
用途：—
模块级函数：_networkx_add_direct_connections, _add_pos_emb
类：
- LayerStack（方法：__init__, of_repeated_layer, forward）
- PerFeatureTransformer（方法：__init__, __setstate__, forward, forward, forward, add_embeddings, empty_trainset_representation_cache）

## src/tabpfn\architectures\encoders\__init__.py
用途：DEPRECATION WARNING: Please note that this module will be deprecated in future
模块级函数：—
类：—

## src/tabpfn\architectures\encoders\pipeline_interfaces.py
用途：Interfaces for encoders.
模块级函数：—
类：
- TorchPreprocessingPipeline（方法：__init__, forward）
- TorchPreprocessingStep（方法：__init__, _fit, _transform, _validate_input_keys, _validate_output_keys, forward）

## src/tabpfn\architectures\encoders\steps\__init__.py
用途：—
模块级函数：—
类：—

## src/tabpfn\architectures\encoders\steps\_ops.py
用途：Custom implementations of PyTorch functions. Needed for ONNX export.
模块级函数：torch_nansum, torch_nanmean, torch_nanmean, torch_nanmean, torch_nanstd, normalize_data, normalize_data, normalize_data, select_features, remove_outliers
类：—

## src/tabpfn\architectures\encoders\steps\feature_group_projections_encoder_step.py
用途：DEPRECATED: Projections from cell-level tensors to embedding space.
模块级函数：—
类：
- LinearInputEncoderStep（方法：__init__, _fit, _transform）
- MLPInputEncoderStep（方法：__init__, _fit, _transform）

## src/tabpfn\architectures\encoders\steps\feature_transform_encoder_step.py
用途：Encoder step to normalize the input in different ways.
模块级函数：—
类：
- FeatureTransformEncoderStep（方法：__init__, _fit, _transform）

## src/tabpfn\architectures\encoders\steps\frequency_feature_encoder_step.py
用途：Encoder step to add frequency-based features to the input.
模块级函数：—
类：
- FrequencyFeatureEncoderStep（方法：__init__, _fit, _transform）

## src/tabpfn\architectures\encoders\steps\multiclass_classification_target_encoder_step.py
用途：Encoder step to encode multiclass classification targets.
模块级函数：—
类：
- MulticlassClassificationTargetEncoderStep（方法：__init__, _apply, _fit, flatten_targets, _transform）

## src/tabpfn\architectures\encoders\steps\nan_handling_encoder_step.py
用途：Encoder step to handle NaN and infinite values in the input.
模块级函数：—
类：
- NanHandlingEncoderStep（方法：__init__, _fit, _transform, _validate_keys）

## src/tabpfn\architectures\encoders\steps\normalize_feature_groups_encoder_step.py
用途：Encoder step to handle variable number of features.
模块级函数：—
类：
- NormalizeFeatureGroupsEncoderStep（方法：__init__, _fit, _transform）

## src/tabpfn\architectures\encoders\steps\remove_duplicate_features_encoder_step.py
用途：Encoder step to remove duplicate features. Note, this is a No-op currently.
模块级函数：—
类：
- RemoveDuplicateFeaturesEncoderStep（方法：__init__, _fit, _transform）

## src/tabpfn\architectures\encoders\steps\remove_empty_features_encoder_step.py
用途：Encoder step to remove empty (constant) features.
模块级函数：—
类：
- RemoveEmptyFeaturesEncoderStep（方法：__init__, _fit, _transform）

## src/tabpfn\architectures\interface.py
用途：Defines the interface for modules containing architectures.
模块级函数：_get_unused_items
类：
- ArchitectureConfig（方法：get_unused_config）
- ArchitectureModule（方法：parse_config, get_architecture）
- Architecture（方法：forward, forward, forward）

## src/tabpfn\architectures\shared\__init__.py
用途：Shared code between the different architectures.
模块级函数：—
类：—

## src/tabpfn\architectures\shared\attention_gqa_check.py
用途：Check for the enable_gqa parameter of PyTorch scaled_dot_product_attention.
模块级函数：gqa_is_supported
类：—

## src/tabpfn\architectures\shared\chunked_evaluate.py
用途：Function for evaluating a function on a Tensor in chunks.
模块级函数：chunked_evaluate_maybe_inplace
类：—

## src/tabpfn\architectures\shared\column_embeddings.py
用途：Function for loading the pre-generated column embeddings.
模块级函数：load_column_embeddings
类：—

## src/tabpfn\architectures\tabpfn_v2_5.py
用途：The TabPFN v2.5 architecture.
模块级函数：_batched_scaled_dot_product_attention, parse_config, get_architecture, _prepare_targets, _replace_keys_from_base_architecture, _remove_constant_features, _pad_and_reshape_feature_groups, _impute_nan_and_inf_with_mean, _impute_target_nan_and_inf, _normalize_feature_groups, _generate_nan_and_inf_indicator
类：
- TabPFNV2p5Config（方法：—）
- AddThinkingRows（方法：__init__, forward, reset_parameters）
- Attention（方法：__init__）
- AlongRowAttention（方法：forward）
- AlongColumnAttention（方法：forward）
- LowerPrecisionLayerNorm（方法：forward）
- TabPFNBlock（方法：__init__, forward）
- TabPFNV2p5（方法：__init__, _get_feature_group_embedder, _add_column_embeddings, forward, _preprocess_and_embed_features, _preprocess_and_embed_targets, load_state_dict）

## src/tabpfn\architectures\tabpfn_v2_6.py
用途：The TabPFN v2.6 architecture.
模块级函数：_batched_scaled_dot_product_attention, parse_config, get_architecture, _prepare_targets, _remove_constant_features, _pad_and_reshape_feature_groups, _impute_nan_and_inf_with_mean, _impute_target_nan_and_inf, _normalize_feature_groups, _generate_nan_and_inf_indicator
类：
- TabPFNV2p6Config（方法：—）
- AddThinkingRows（方法：__init__, forward, reset_parameters）
- Attention（方法：__init__）
- AlongRowAttention（方法：forward）
- AlongColumnAttention（方法：forward）
- LowerPrecisionRMSNorm（方法：forward）
- TabPFNBlock（方法：__init__, forward）
- TabPFNV2p6（方法：__init__, _get_feature_group_embedder, _add_column_embeddings, forward, _preprocess_and_embed_features, _preprocess_and_embed_targets）

## src/tabpfn\base.py
用途：Common logic for TabPFN models.
模块级函数：initialize_tabpfn_model, _assert_inference_configs_equal, determine_precision, create_inference_engine, initialize_model_variables_helper, estimator_to_device, initialize_telemetry, get_embeddings
类：
- BaseModelSpecs（方法：__init__）
- ClassifierModelSpecs（方法：—）
- RegressorModelSpecs（方法：__init__）

## src/tabpfn\classifier.py
用途：TabPFNClassifier class.
模块级函数：_validate_eval_metric
类：
- TabPFNClassifier（方法：__init__, create_default_for_version, estimator_type, model_, _more_tags, __sklearn_tags__, _initialize_model_variables, _initialize_for_differentiable_input, _initialize_dataset_preprocessing, _get_tuning_classifier, fit, fit_from_preprocessed, fit_with_differentiable_input, _maybe_calibrate_temperature_and_tune_decision_thresholds, _compute_holdout_validation_data, _raw_predict, predict, predict_logits, predict_raw_logits, predict_proba, _predict_proba, _get_calibrated_softmax_temperature, _maybe_reweight_probas, _apply_temperature, _average_across_estimators, _apply_softmax, _apply_balancing, logits_to_probabilities, forward, get_embeddings, save_fit_state, load_from_fit_state, to）

## src/tabpfn\constants.py
用途：Various constants used throughout the library.
模块级函数：—
类：
- ModelVersion（方法：—）

## src/tabpfn\errors.py
用途：Custom exception classes for TabPFN.
模块级函数：handle_oom_errors
类：
- TabPFNError（方法：—）
- TabPFNUserError（方法：—）
- TabPFNValidationError（方法：—）
- TabPFNHuggingFaceGatedRepoError（方法：__init__）
- TabPFNOutOfMemoryError（方法：__init__）
- TabPFNCUDAOutOfMemoryError（方法：—）
- TabPFNMPSOutOfMemoryError（方法：—）

## src/tabpfn\finetuning\__init__.py
用途：Single-dataset fine-tuning wrappers for TabPFN models.
模块级函数：—
类：—

## src/tabpfn\finetuning\_torch_compat.py
用途：PyTorch compatibility helpers.
模块级函数：—
类：—

## src/tabpfn\finetuning\data_util.py
用途：Utilities for data preparation used in fine-tuning wrappers.
模块级函数：_take, _chunk_data_non_stratified, _chunk_data_stratified, _collate_list_field, _collate_tensor_field, _collate_cat_indices, meta_dataset_collator, shuffle_and_chunk_data, get_preprocessed_dataset_chunks
类：
- ClassifierBatch（方法：—）
- RegressorBatch（方法：—）
- BaseDatasetConfig（方法：—）
- ClassifierDatasetConfig（方法：—）
- RegressorDatasetConfig（方法：bardist_, bardist_）
- DatasetCollectionWithPreprocessing（方法：__init__, __len__, __getitem__）

## src/tabpfn\finetuning\finetuned_base.py
用途：Abstract base class for fine-tuning TabPFN models.
模块级函数：_init_distributed_if_needed, _maybe_setup_ddp, _move_tabpfn_cached_contexts_to_device
类：
- _TabPFNDDPWrapper（方法：__init__, forward）
- EvalResult（方法：—）
- FinetunedTabPFNBase（方法：__init__, _build_estimator_config, _build_eval_config, _training_forward, _estimator_kwargs, _model_type, _metric_name, _create_estimator, _setup_estimator, _setup_batch, _should_skip_batch, _forward_with_loss, _evaluate_model, _is_improvement, _get_initial_best_metric, _get_checkpoint_metrics, _log_epoch_evaluation, _setup_inference_model, predict, _get_train_val_split, _get_valid_finetuning_query_size, fit, _fit）

## src/tabpfn\finetuning\finetuned_classifier.py
用途：A TabPFN classifier that finetunes the underlying model for a single task.
模块级函数：_compute_classification_loss
类：
- FinetunedTabPFNClassifier（方法：__init__, _estimator_kwargs, _model_type, _metric_name, _create_estimator, _setup_estimator, _setup_batch, _should_skip_batch, _forward_with_loss, _evaluate_model, _is_improvement, _get_initial_best_metric, _get_checkpoint_metrics, _get_valid_finetuning_query_size, _log_epoch_evaluation, _setup_inference_model, fit, predict_proba, predict）

## src/tabpfn\finetuning\finetuned_regressor.py
用途：A TabPFN regressor that finetunes the underlying model for a single task.
模块级函数：_compute_regression_loss, _ranked_probability_score_loss_from_bar_logits
类：
- FinetunedTabPFNRegressor（方法：__init__, _estimator_kwargs, _model_type, _metric_name, _create_estimator, _setup_estimator, _should_skip_batch, _setup_batch, _forward_with_loss, _evaluate_model, _is_improvement, _get_initial_best_metric, _get_checkpoint_metrics, _get_valid_finetuning_query_size, _log_epoch_evaluation, _setup_inference_model, fit, predict）

## src/tabpfn\finetuning\train_util.py
用途：Some utility functions for training.
模块级函数：_format_train_size, clone_model_for_evaluation, get_checkpoint_name, save_checkpoint, get_checkpoint_path_and_epoch_from_output_dir, get_and_init_optimizer, get_cosine_schedule_with_warmup
类：—

## src/tabpfn\inference.py
用途：Module that defines different ways to run inference with TabPFN.
模块级函数：_model_expectes_task_type_arg, _raise_if_kv_cache_enabled_on_save_or_load, _prepare_model_inputs, _move_and_squeeze_output, _maybe_run_gpu_preprocessing, _get_current_device
类：
- InferenceEngine（方法：__init__, iter_outputs, use_torch_inference_mode, save_state_except_model_weights, _create_copy_for_pickling, load_state, _set_models, to, _move_models_to_devices）
- SingleDeviceInferenceEngine（方法：__init__, _create_copy_for_pickling, _set_models）
- MultiDeviceInferenceEngine（方法：__init__, _create_copy_for_pickling, _set_models, _move_models_to_devices, get_devices）
- InferenceEngineOnDemand（方法：__init__, iter_outputs, _call_model）
- InferenceEngineBatchedNoPreprocessing（方法：__init__, iter_outputs, use_torch_inference_mode, _move_models_to_devices）
- InferenceEngineCachePreprocessing（方法：__init__, iter_outputs, _call_model, use_torch_inference_mode）
- InferenceEngineCacheKV（方法：__init__, iter_outputs, _move_models_to_devices）
- _PerDeviceModelCache（方法：__init__, to, get, set_dtype, get_devices）

## src/tabpfn\inference_config.py
用途：Additional configuration options for inference.
模块级函数：_get_v2_config, _get_v2_5_config, _get_v2_6_config
类：
- InferenceConfig（方法：override_with_user_input_and_resolve_auto, get_resolved_outlier_removal_std, get_default）

## src/tabpfn\inference_tuning.py
用途：Inference tuning helpers for TabPFN fit/predict calls.
模块级函数：compute_metric_to_minimize, get_tuning_splits, find_optimal_classification_thresholds, find_optimal_classification_threshold_single_class, select_robust_optimal_threshold, find_optimal_temperature, get_default_tuning_holdout_frac, get_default_tuning_n_folds, resolve_tuning_config
类：
- TuningConfig（方法：resolve）
- ClassifierTuningConfig（方法：—）
- ClassifierEvalMetrics（方法：—）

## src/tabpfn\misc\_sklearn_compat.py
用途：Ease developer experience to support multiple versions of scikit-learn.
模块级函数：_dataclass_args, get_tags, _to_new_tags
类：—

## src/tabpfn\misc\debug_versions.py
用途：This file is taken from PyTorch and modified to work with TabPFN, also
模块级函数：_run, _run_and_read_all, _run_and_parse_first_match, _run_and_return_first_line, _get_conda_packages, _get_gcc_version, _get_clang_version, _get_cmake_version, _get_nvidia_driver_version, _get_gpu_info, _get_running_cuda_version, _get_cudnn_version, _get_nvidia_smi, _get_cpu_info, _get_platform, _get_mac_version, _get_windows_version, _get_lsb_version, _check_release_file, _get_os, _get_python_platform, _get_libc_version, _get_pip_packages, _get_cachingallocator_config, _get_cuda_module_loading_config, _is_xnnpack_available, _get_env_info, _replace_nones, _replace_bools, _prepend, _replace_if_empty, _maybe_start_on_next_line, _pretty_str, _get_deps_info, display_debug_info
类：—

## src/tabpfn\model\__init__.py
用途：Contains references to the base architecture, for backwards compatability.
模块级函数：—
类：—

## src/tabpfn\model\attention.py
用途：DEPRECATED: Please import tabpfn.architectures.base.attention instead.
模块级函数：—
类：—

## src/tabpfn\model\bar_distribution.py
用途：DEPRECATED: Please import tabpfn.architectures.base.bar_distribution instead.
模块级函数：—
类：—

## src/tabpfn\model\config.py
用途：DEPRECATED: Please import tabpfn.architectures.base.config instead.
模块级函数：—
类：—

## src/tabpfn\model\encoders.py
用途：DEPRECATED: Please import tabpfn.architectures.encoders instead.
模块级函数：—
类：—

## src/tabpfn\model\layer.py
用途：DEPRECATED: Please import tabpfn.architectures.base.layer instead.
模块级函数：—
类：—

## src/tabpfn\model\loading.py
用途：DEPRECATED: Please import from `tabpfn.model_loading` instead.
模块级函数：—
类：—

## src/tabpfn\model\memory.py
用途：DEPRECATED: Please import tabpfn.architectures.base.memory instead.
模块级函数：—
类：—

## src/tabpfn\model\mlp.py
用途：DEPRECATED: Please import tabpfn.architectures.base.mlp instead.
模块级函数：—
类：—

## src/tabpfn\model\preprocessing.py
用途：DEPRECATED: Please import tabpfn.preprocessing instead.
模块级函数：—
类：—

## src/tabpfn\model\transformer.py
用途：DEPRECATED: Please import tabpfn.architectures.base.transformer instead.
模块级函数：—
类：—

## src/tabpfn\model_loading.py
用途：Functions for downloading and loading model checkpoints.
模块级函数：_get_model_source, _try_huggingface_downloads, _try_direct_downloads, download_all_models, _version_has_direct_download_option, get_cache_dir, download_model, _download_model, prepend_cache_path, load_model_criterion_config, load_model_criterion_config, load_model_criterion_config, _log_model_config, log_model_init_params, _resolve_model_version, resolve_model_version, resolve_model_path, get_loss_criterion, _load_checkpoint, load_model, _get_inference_config_from_checkpoint, save_tabpfn_model, save_fitted_tabpfn_model, _extract_archive, load_fitted_tabpfn_model, _resolve_architecture_name
类：
- ModelType（方法：—）
- ModelSource（方法：get_classifier_v2, get_regressor_v2, get_classifier_v2_5, get_regressor_v2_5, get_classifier_v2_6, get_regressor_v2_6）

## src/tabpfn\parallel_execute.py
用途：Parallel evaluation of a set of functions across multiple PyTorch devices.
模块级函数：parallel_execute, _execute_in_current_thread, _execute_with_multithreading, _execute_function_in_thread
类：
- ParallelFunction（方法：__call__）

## src/tabpfn\preprocessing\__init__.py
用途：—
模块级函数：—
类：—

## src/tabpfn\preprocessing\clean.py
用途：Module for cleaning the data.
模块级函数：clean_data, fix_dtypes, process_text_na_dataframe
类：—

## src/tabpfn\preprocessing\configs.py
用途：Preprocessor and ensemble config objects.
模块级函数：—
类：
- PreprocessorConfig（方法：__str__）
- EnsembleConfig（方法：—）
- ClassifierEnsembleConfig（方法：—）
- RegressorEnsembleConfig（方法：—）

## src/tabpfn\preprocessing\datamodel.py
用途：Data model for the preprocessing pipeline.
模块级函数：_validate_permutation
类：
- FeatureModality（方法：—）
- Feature（方法：—）
- FeatureSchema（方法：from_only_categorical_indices, feature_names, num_columns, indices_for, indices_for_modalities, append_columns, slice_for_indices, update_from_preprocessing_step_result, remove_columns, apply_permutation）

## src/tabpfn\preprocessing\ensemble.py
用途：Module for generating ensemble configurations.
模块级函数：_balance, _generate_index_permutations, _get_subsample_indices_for_estimators, _generate_class_permutations, generate_classification_ensemble_configs, generate_regression_ensemble_configs
类：
- TabPFNEnsembleMember（方法：transform_X_test）
- TabPFNEnsemblePreprocessor（方法：__init__, next_static_seed, fit_transform_ensemble_members_iterator, fit_transform_ensemble_members）

## src/tabpfn\preprocessing\label_encoder.py
用途：Label encoding utilities for TabPFN classifier.
模块级函数：—
类：
- LabelMetadata（方法：—）
- TabPFNLabelEncoder（方法：__init__, fit_transform, inverse_transform）

## src/tabpfn\preprocessing\modality_detection.py
用途：Module to infer feature modalities: numerical, categorical, text, etc.
模块级函数：detect_feature_modalities, _detect_feature_modality, _is_numeric_pandas_series, _detect_numeric_as_categorical, _get_unique_with_sklearn_compatible_error
类：—

## src/tabpfn\preprocessing\pipeline_factory.py
用途：Methods to generate a preprocessing pipeline from ensemble configurations.
模块级函数：_polynomial_feature_settings, create_preprocessing_pipeline
类：—

## src/tabpfn\preprocessing\pipeline_interface.py
用途：Interfaces for creating preprocessing pipelines.
模块级函数：—
类：
- PreprocessingStepResult（方法：__post_init__）
- PreprocessingPipelineResult（方法：—）
- PreprocessingStep（方法：_fit, _transform, fit_transform, transform, _validate_added_data）
- PreprocessingPipeline（方法：__init__, fit_transform, transform, _process_steps, _validate_expected_dtype, _maybe_append_added_columns, _validate_steps, __len__）

## src/tabpfn\preprocessing\presets.py
用途：Predefined preprocessor configurations for different model versions.
模块级函数：v2_classifier_preprocessor_configs, v2_regressor_preprocessor_configs, v2_5_classifier_preprocessor_configs, v2_5_regressor_preprocessor_configs, v2_6_classifier_preprocessor_configs, v2_6_regressor_preprocessor_configs
类：—

## src/tabpfn\preprocessing\steps\__init__.py
用途：—
模块级函数：—
类：—

## src/tabpfn\preprocessing\steps\adaptive_quantile_transformer.py
用途：Adaptive Quantile Transformer.
模块级函数：—
类：
- AdaptiveQuantileTransformer（方法：__init__, fit）

## src/tabpfn\preprocessing\steps\add_fingerprint_features_step.py
用途：Add Fingerprint Features Step.
模块级函数：_hash_row_bytes
类：
- AddFingerprintFeaturesStep（方法：__init__, _fit, _transform）

## src/tabpfn\preprocessing\steps\add_svd_features_step.py
用途：Adds SVD features to the data.
模块级函数：get_svd_features_transformer
类：
- AddSVDFeaturesStep（方法：__init__, num_added_features, _fit, _transform）

## src/tabpfn\preprocessing\steps\differentiable_z_norm_step.py
用途：Differentiable Z-Norm Step.
模块级函数：—
类：
- DifferentiableZNormStep（方法：__init__, _fit, _transform）

## src/tabpfn\preprocessing\steps\encode_categorical_features_step.py
用途：Encode Categorical Features Step.
模块级函数：_get_all_cat_indices_after_onehot, _get_least_common_category_count
类：
- EncodeCategoricalFeaturesStep（方法：__init__, _get_transformer, _fit, _fit_transform_internal, fit_transform, _transform）

## src/tabpfn\preprocessing\steps\kdi_transformer.py
用途：KDI Transformer with NaN.
模块级函数：get_all_kdi_transformers
类：
- KDITransformerWithNaN（方法：__init__, _more_tags, fit, transform, fit_transform）

## src/tabpfn\preprocessing\steps\nan_handling_polynomial_features_step.py
用途：Nan Handling Polynomial Features Step.
模块级函数：—
类：
- NanHandlingPolynomialFeaturesStep（方法：__init__, _fit, _transform）

## src/tabpfn\preprocessing\steps\preprocessing_helpers.py
用途：Feature Preprocessing Transformer Step.
模块级函数：get_ordinal_encoder
类：
- OrderPreservingColumnTransformer（方法：__init__, transform, fit_transform, _preserve_order）

## src/tabpfn\preprocessing\steps\remove_constant_features_step.py
用途：Remove Constant Features Step.
模块级函数：—
类：
- RemoveConstantFeaturesStep（方法：__init__, _fit, _transform）

## src/tabpfn\preprocessing\steps\reshape_feature_distribution_step.py
用途：Reshape the feature distributions using different transformations.
模块级函数：_exp_minus_1, _make_box_cox_safe, _skew, get_adaptive_preprocessors, get_all_reshape_feature_distribution_preprocessors
类：
- ReshapeFeatureDistributionsStep（方法：get_column_types, __init__, _create_transformers_and_new_schema, _fit, _transform）

## src/tabpfn\preprocessing\steps\safe_power_transformer.py
用途：Safe Power Transformer.
模块级函数：_yeojohnson, _yeojohnson_transform, _yeojohnson_llf, _yeojohnson_normmax, _yeojohnson_inverse_transform
类：
- SafePowerTransformer（方法：__init__, _yeo_johnson_optimize, _yeo_johnson_transform, _yeo_johnson_inverse_transform）

## src/tabpfn\preprocessing\steps\shuffle_features_step.py
用途：Shuffle Features Step.
模块级函数：—
类：
- ShuffleFeaturesStep（方法：__init__, _fit, _transform）

## src/tabpfn\preprocessing\steps\squashing_scaler_transformer.py
用途：Implementation of the SquashingScaler, adapted from skrub.
模块级函数：_validate_data, _mask_inf, _set_zeros, _soft_clip
类：
- _MinMaxScaler（方法：fit, transform）
- SquashingScaler（方法：__init__, fit, fit_transform, transform）

## src/tabpfn\preprocessing\steps\utils.py
用途：Utility functions for preprocessing steps.
模块级函数：make_scaler_safe, wrap_with_safe_standard_scaler, _identity, _replace_inf_with_nan, _make_finite_steps
类：
- _NoInverseImputer（方法：inverse_transform）

## src/tabpfn\preprocessing\torch\__init__.py
用途：Torch-based preprocessing utilities.
模块级函数：—
类：—

## src/tabpfn\preprocessing\torch\factory.py
用途：Factory for creating torch preprocessing pipelines.
模块级函数：create_gpu_preprocessing_pipeline
类：—

## src/tabpfn\preprocessing\torch\ops.py
用途：Torch operations for preprocessing with NaN handling.
模块级函数：torch_nansum, torch_nanmean, torch_nanstd
类：—

## src/tabpfn\preprocessing\torch\pipeline_interface.py
用途：Interfaces for torch preprocessing pipeline.
模块级函数：—
类：
- TorchPreprocessingStep（方法：fit_transform, __repr__, _fit, _transform）
- TorchPreprocessingPipeline（方法：__init__, __call__, _maybe_update_fitted_cache, __repr__, _validate_steps, _validate_use_fitted_cache, _validate_metadata）
- TorchPreprocessingStepResult（方法：—）
- TorchPreprocessingPipelineOutput（方法：—）

## src/tabpfn\preprocessing\torch\steps.py
用途：Pipeline step wrappers for torch preprocessing operations.
模块级函数：—
类：
- TorchStandardScalerStep（方法：__init__, _fit, _transform）
- TorchSoftClipOutliersStep（方法：__init__, _fit, _transform）

## src/tabpfn\preprocessing\torch\torch_soft_clip_outliers.py
用途：Torch implementation of outlier clipping with NaN handling.
模块级函数：—
类：
- TorchSoftClipOutliers（方法：__init__, fit, transform, __call__）

## src/tabpfn\preprocessing\torch\torch_standard_scaler.py
用途：Torch implementation of StandardScaler with NaN handling.
模块级函数：—
类：
- TorchStandardScaler（方法：fit, transform, __call__）

## src/tabpfn\preprocessing\transform.py
用途：Module for fitting and transforming preprocessing pipelines.
模块级函数：_fit_preprocessing_one, _transform_labels_one, fit_preprocessing
类：—

## src/tabpfn\preprocessing\type_detection.py
用途：Module to infer feature types.
模块级函数：infer_categorical_features
类：—

## src/tabpfn\regressor.py
用途：TabPFNRegressor class.
模块级函数：_logits_to_output
类：
- MainOutputDict（方法：—）
- FullOutputDict（方法：—）
- TabPFNRegressor（方法：__init__, create_default_for_version, estimator_type, model_, norm_bardist_, norm_bardist_, bardist_, bardist_, _more_tags, __sklearn_tags__, _initialize_model_variables, _initialize_dataset_preprocessing, fit_from_preprocessed, fit, predict, predict, predict, predict, predict, forward, _handle_constant_target, get_embeddings, save_fit_state, load_from_fit_state, to）

## src/tabpfn\settings.py
用途：Settings module for TabPFN configuration.
模块级函数：—
类：
- TabPFNSettings（方法：model_post_init）
- PytorchSettings（方法：—）
- TestingSettings（方法：_parse_ci）
- Settings（方法：—）

## src/tabpfn\utils.py
用途：A collection of random utilities for the TabPFN models.
模块级函数：get_autocast_context, _repair_borders, _cancel_nan_borders, infer_devices, _parse_device, _is_mps_supported, is_autocast_available, infer_fp16_inference_mode, infer_random_state, _map_to_bucket_ix, _cdf, translate_probs_across_borders, remove_non_differentiable_preprocessing_from_models, transform_borders_one, pad_tensors, balance_probas_by_class_counts, convert_batch_of_cat_ix_to_schema
类：—

## src/tabpfn\validation.py
用途：Module for validation logic.
模块级函数：ensure_compatible_fit_inputs, ensure_compatible_predict_input_sklearn, validate_dataset_size, ensure_compatible_fit_inputs_sklearn, validate_num_classes, _validate_num_samples_and_features, _validate_num_samples_for_cpu
类：—
