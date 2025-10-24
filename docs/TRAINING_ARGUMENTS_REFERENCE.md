# Transformers `TrainingArguments` Reference

This reference compiles every field exposed by `transformers.TrainingArguments` (and the additions from `Seq2SeqTrainingArguments`) as of 🤗 Transformers **v4.57.1**. Defaults and behaviours may change in later releases, so keep an eye on the release notes if you upgrade the library.

## Check Your Local Setup
- Confirm the installed version:
  ```bash
  python3 - <<'PY'
  import transformers
  print(transformers.__version__)
  PY
  ```
- Inspect the live dataclass with Python help:
  ```bash
  python3 - <<'PY'
  from transformers import TrainingArguments
  help(TrainingArguments)
  PY
  ```
- When using an argument parser, `python3 my_trainer.py --help` prints the same information converted to CLI flags.

## Reading This Reference
- Types and defaults are taken from the dataclass signature. `None` means "disabled" or "infer automatically".
- Many arguments accept either strings or enums. Strings shown here must match the enum values (for example `"steps"` or `"epoch"`).
- Values marked as experimental may change behaviour across PyTorch or Transformers versions.

## Argument Catalogue

### Run Management & Checkpoints
- `output_dir` *(type: Optional[str], default: None)* — Folder where checkpoints, predictions, and logs are written. It is created if it doesn’t exist.
- `resume_from_checkpoint` *(type: Optional[str], default: None)* — Path to an existing checkpoint to resume from; restores model weights plus optimizer/scheduler RNG states when present.
- `restore_callback_states_from_checkpoint` *(type: bool, default: False)* — When resuming, also reload callback state (for example early stopping counters); overrides callbacks supplied when you instantiate the trainer.
- `save_strategy` *(type: Union[SaveStrategy, str], default: steps)* — Controls when checkpoints are written: `"no"`, `"steps"`, `"epoch"`, or `"best"` (whenever the monitored metric improves).
- `save_steps` *(type: float, default: 500)* — Step interval or step ratio (if <1) used when `save_strategy="steps"`.
- `save_total_limit` *(type: Optional[int], default: None)* — Maximum number of checkpoints kept on disk; older ones are deleted in chronological order.
- `save_safetensors` *(type: bool, default: True)* — Write model weights to the `.safetensors` format instead of PyTorch `.bin`.
- `save_on_each_node` *(type: bool, default: False)* — In multi-node training, let every node write its own checkpoint (set to `False` if nodes share storage).
- `save_only_model` *(type: bool, default: False)* — Store only model weights; skips optimizer, scheduler, and RNG—useful for packaging but prevents resuming training.
- `push_to_hub` *(type: bool, default: False)* — Automatically push checkpoints to the Hugging Face Hub following `hub_strategy`.
- `hub_model_id` *(type: Optional[str], default: None)* — Target repository on the Hub (e.g. `"username/project-name"`). If omitted the repo name defaults to `output_dir`'s basename.
- `hub_strategy` *(type: Union[HubStrategy, str], default: every_save)* — Push cadence: `"end"`, `"every_save"`, or `"checkpoint"` (only best checkpoints).
- `hub_token` *(type: Optional[str], default: None)* — Personal access token used for Hub authentication; if `None`, falls back to `huggingface-cli login`.
- `hub_private_repo` *(type: Optional[bool], default: None)* — Set `True` to create a private repository on first push; leave `None` to follow Hub defaults.
- `hub_always_push` *(type: bool, default: False)* — Force a push even if nothing changed on disk (useful when metadata is updated without new weights).
- `hub_revision` *(type: Optional[str], default: None)* — Branch or tag to push to; defaults to the Hub repository’s main branch.

### Workflow Toggles
- `do_train` *(type: bool, default: False)* — Gate that decides whether the training loop runs; convenient when wiring together multi-stage scripts.
- `do_eval` *(type: bool, default: False)* — Enable validation passes; often set automatically when `eval_strategy` is not `"no"`.
- `do_predict` *(type: bool, default: False)* — Request a final prediction loop on the test split after training.

### Evaluation & Metrics
- `eval_strategy` *(type: Union[IntervalStrategy, str], default: no)* — `"no"`, `"steps"`, or `"epoch"`; controls how often evaluation runs during training.
- `eval_steps` *(type: Optional[float], default: None)* — Evaluation frequency when `eval_strategy="steps"`; use an integer step count or a ratio (<1) of total steps.
- `eval_delay` *(type: float, default: 0)* — Skip the first evaluation until this many steps or epochs have completed (interpreted like `eval_strategy`).
- `eval_accumulation_steps` *(type: Optional[int], default: None)* — Accumulate evaluation outputs on-device before moving to CPU; useful to avoid host memory spikes.
- `eval_do_concat_batches` *(type: bool, default: True)* — Concatenate evaluation batches into contiguous tensors; disable to keep per-batch structures if your metrics expect lists.
- `eval_use_gather_object` *(type: bool, default: False)* — Use `accelerate.gather_object` to merge arbitrary Python objects across devices; enable only if you return non-tensor metadata.
- `eval_on_start` *(type: bool, default: False)* — Run one evaluation pass before the first training step (helpful for sanity-checking metrics).
- `prediction_loss_only` *(type: bool, default: False)* — When `True`, evaluation returns only the scalar loss instead of logits and labels.
- `batch_eval_metrics` *(type: bool, default: False)* — Call `compute_metrics` each batch to incrementally accumulate statistics instead of buffering all logits; requires your metric function to accept a `compute_result` flag.
- `include_for_metrics` *(type: list, default: <factory>)* — Extra payload forwarded to `compute_metrics`; supported entries are `"inputs"` and `"loss"`.
- `include_num_input_tokens_seen` *(type: Union[str, bool], default: no)* — Track token counts during training; choose `"all"`, `"non_padding"`, `"no"` (or use booleans mapping to `"all"`/`"no"`). Adds cross-device reductions.
- `load_best_model_at_end` *(type: bool, default: False)* — After training finishes, reload the checkpoint with the best recorded metric.
- `metric_for_best_model` *(type: Optional[str], default: None)* — Name of the metric value to monitor when `load_best_model_at_end=True` (for example `"eval_loss"` or `"accuracy"`).
- `greater_is_better` *(type: Optional[bool], default: None)* — Direction of improvement for the monitored metric; set `False` for losses, `True` for accuracy-like scores.
- `ignore_data_skip` *(type: bool, default: False)* — When resuming, skip recomputing how many samples were already consumed; speeds up restarts at the cost of slightly different sampling.
- `average_tokens_across_devices` *(type: bool, default: True)* — Average `num_tokens_in_batch` across ranks before computing the loss, so distributed metrics match single-device behaviour.
- `use_cache` *(type: bool, default: False)* — Keep `past_key_values` or other caches during eval/generation; leave disabled for standard training unless a PEFT method requires it.

### Batching & Data Pipeline
- `per_device_train_batch_size` *(type: int, default: 8)* — Micro-batch size on each accelerator; global batch equals this value × number of devices × `gradient_accumulation_steps`.
- `per_device_eval_batch_size` *(type: int, default: 8)* — Per-device batch size used during evaluation and prediction passes.
- `gradient_accumulation_steps` *(type: int, default: 1)* — Number of forward/backward passes to accumulate before taking an optimizer step.
- `auto_find_batch_size` *(type: bool, default: False)* — Shrinks the batch size automatically on CUDA OOM using exponential backoff (requires `accelerate`).
- `torch_empty_cache_steps` *(type: Optional[int], default: None)* — Call `torch.cuda.empty_cache()` every N steps to reduce peak VRAM at the expense of ~10% throughput.
- `num_train_epochs` *(type: float, default: 3.0)* — Number of passes over the dataset. Fractions run a partial final epoch.
- `max_steps` *(type: int, default: -1)* — Total optimizer steps to run; overrides `num_train_epochs` when set to a positive value.
- `group_by_length` *(type: bool, default: False)* — Bucket examples of similar length together when dynamically padding to reduce wasted tokens.
- `length_column_name` *(type: str, default: length)* — Dataset column holding precomputed sequence lengths when `group_by_length=True`.
- `dataloader_drop_last` *(type: bool, default: False)* — Drop the last incomplete batch to keep shapes consistent (useful when using certain collective ops).
- `dataloader_num_workers` *(type: int, default: 0)* — Number of background worker processes for data loading; `0` loads on the main process.
- `dataloader_prefetch_factor` *(type: Optional[int], default: None)* — Batches each worker prefetches; falls back to PyTorch default (`2`) when `None`.
- `dataloader_pin_memory` *(type: bool, default: True)* — Pin CPU memory before transferring to GPU; set `False` on CPU-only or image-heavy pipelines to reduce memory pressure.
- `dataloader_persistent_workers` *(type: bool, default: False)* — Keep dataloader workers alive across epochs (only effective when `num_workers > 0`).
- `remove_unused_columns` *(type: bool, default: True)* — Drop dataset columns not consumed by the model forward signature to avoid unexpected keyword errors.
- `label_names` *(type: Optional[list[str]], default: None)* — Explicit list of dataset keys treated as labels; required when you have multiple label tensors (e.g. encoder–decoder with auxiliary targets).

### Optimization & Scheduling
- `learning_rate` *(type: float, default: 5e-05)* — Peak learning rate passed to the optimizer.
- `weight_decay` *(type: float, default: 0.0)* — L2 weight decay applied to all parameters except LayerNorm and bias terms.
- `adam_beta1` *(type: float, default: 0.9)* — β₁ coefficient for AdamW-style optimizers.
- `adam_beta2` *(type: float, default: 0.999)* — β₂ coefficient for AdamW-style optimizers.
- `adam_epsilon` *(type: float, default: 1e-08)* — Epsilon term for numerical stability in Adam variants.
- `max_grad_norm` *(type: float, default: 1.0)* — Gradient clipping threshold (set `0` or `None` to disable clipping).
- `optim` *(type: Union[OptimizerNames, str], default: adamw_torch_fused)* — Optimizer backend, e.g. `"adamw_torch"`, `"adamw_torch_fused"`, `"adamw_anyprecision"`, `"adamw_bnb_8bit"`, `"adafactor"`, `"sgd"`, `"lamb"`, `"apollo_adamw"`, or `"galore_adamw"`. Defaults to the fused PyTorch AdamW when available.
- `optim_args` *(type: Optional[str], default: None)* — Free-form argument string forwarded to certain optimizers (AnyPrecision, AdEMAMix, GaLore, APOLLO).
- `optim_target_modules` *(type: Union[NoneType, str, list[str]], default: None)* — Restrict some optimizers (GaLore/APOLLO) to specific module names (typically `nn.Linear` layers).
- `lr_scheduler_type` *(type: Union[SchedulerType, str], default: linear)* — Scheduler policy; choose from `"linear"`, `"cosine"`, `"cosine_with_restarts"`, `"polynomial"`, `"constant"`, `"constant_with_warmup"`, `"inverse_sqrt"`, etc.
- `lr_scheduler_kwargs` *(type: Union[dict, str, NoneType], default: None)* — Extra keyword arguments forwarded to the scheduler factory (JSON string or dict).
- `warmup_ratio` *(type: Optional[float], default: None)* — Fraction of total training steps used for warmup; overrides `warmup_steps` when provided.
- `warmup_steps` *(type: float, default: 0)* — Absolute warmup steps or ratio (<1) if you prefer to specify the schedule manually.
- `neftune_noise_alpha` *(type: Optional[float], default: None)* — When set, enable NEFTune noise injection (typical values 5–15) to improve instruction tuning stability.
- `label_smoothing_factor` *(type: float, default: 0.0)* — Apply label smoothing during loss computation (use small values like 0.1 for classification tasks).

### Precision & Performance
- `bf16` *(type: bool, default: False)* — Enable bfloat16 mixed precision (supported on NVIDIA Ampere+, Intel XPU, some CPUs, and Ascend NPU).
- `fp16` *(type: bool, default: False)* — Enable float16 mixed precision; use on hardware with reliable FP16 support.
- `bf16_full_eval` *(type: bool, default: False)* — Evaluate in bfloat16 instead of float32 to save memory (may slightly change metrics).
- `fp16_full_eval` *(type: bool, default: False)* — Evaluate in float16 instead of float32.
- `tf32` *(type: Optional[bool], default: None)* — Control TensorFloat-32 matrix math on Ampere GPUs; leave `None` to follow PyTorch’s global default.
- `use_cpu` *(type: bool, default: False)* — Force CPU execution even if CUDA/XPU is available (handy for debugging or staged training).
- `gradient_checkpointing` *(type: bool, default: False)* — Trade compute for memory by recomputing intermediate activations during backward passes.
- `gradient_checkpointing_kwargs` *(type: Union[dict[str, Any], str, NoneType], default: None)* — Extra arguments for gradient checkpointing (e.g. `{"use_reentrant": False}`).
- `torch_compile` *(type: bool, default: False)* — Wrap the model with `torch.compile` to use PyTorch 2.x compile-time optimisations (experimental).
- `torch_compile_backend` *(type: Optional[str], default: None)* — Backend indicator for `torch.compile` (for example `"inductor"`, `"eager"`, `"aot_eager"`); setting a value enables compilation.
- `torch_compile_mode` *(type: Optional[str], default: None)* — Compile mode such as `"default"`, `"reduce-overhead"`, or `"max-autotune"` (refer to PyTorch docs).
- `use_liger_kernel` *(type: bool, default: False)* — Enable the Liger fused kernels for supported LLMs to gain throughput and reduce memory usage.
- `liger_kernel_config` *(type: Optional[dict[str, bool]], default: None)* — Fine-grained configuration for Liger (select which ops like `"rope"` or `"swiglu"` should be replaced).
- `skip_memory_metrics` *(type: bool, default: True)* — Skip probing GPU/CPU memory statistics during logging; set to `False` when you need memory telemetry.

### Logging & Tracking
- `log_level` *(type: str, default: passive)* — Logging level on the main process (`"debug"`, `"info"`, `"warning"`, `"error"`, `"critical"`, or `"passive"` to leave transformers’ default).
- `log_level_replica` *(type: str, default: warning)* — Logging level on non-main processes for distributed runs.
- `log_on_each_node` *(type: bool, default: True)* — When using multiple nodes, log once per node (`True`) or only from rank 0 (`False`).
- `logging_strategy` *(type: Union[IntervalStrategy, str], default: steps)* — `"no"`, `"steps"`, or `"epoch"`; mirrors `eval_strategy` but for logging.
- `logging_first_step` *(type: bool, default: False)* — Emit a log entry at step 0 before updates begin.
- `logging_steps` *(type: float, default: 500)* — Log interval (or ratio when <1) used with `logging_strategy="steps"`.
- `logging_nan_inf_filter` *(type: bool, default: True)* — Drop NaN/Inf losses from logs to keep averaged loss curves meaningful.
- `run_name` *(type: Optional[str], default: None)* — Human-readable run identifier consumed by experiment trackers (W&B, MLflow, Trackio, etc.).
- `disable_tqdm` *(type: Optional[bool], default: None)* — Set `True` to silence TQDM progress bars; `None` lets the trainer decide (disabled on non-main processes).
- `report_to` *(type: Union[NoneType, str, list[str]], default: none)* — Integrations to send metrics to. Supported choices include `"azure_ml"`, `"clearml"`, `"codecarbon"`, `"comet_ml"`, `"dagshub"`, `"dvclive"`, `"flyte"`, `"mlflow"`, `"neptune"`, `"swanlab"`, `"tensorboard"`, `"trackio"`, and `"wandb"`. Use `"all"` or `"none"`.
- `project` *(type: str, default: huggingface)* — Project namespace for logging platforms that need one (currently used by Trackio).
- `trackio_space_id` *(type: Optional[str], default: trackio)* — Target Hugging Face Space to deploy Trackio dashboards (format `"namespace/space-name"`).

### Distributed & Parallel
- `local_rank` *(type: int, default: -1)* — Set by torch.distributed launchers; negative values indicate single-process execution.
- `ddp_backend` *(type: Optional[str], default: None)* — Torch distributed backend (`"nccl"`, `"mpi"`, `"ccl"`, `"gloo"`, `"hccl"`). Leave `None` for PyTorch defaults.
- `ddp_find_unused_parameters` *(type: Optional[bool], default: None)* — Toggle `find_unused_parameters` in DDP; set `False` for models with conditional branches.
- `ddp_bucket_cap_mb` *(type: Optional[int], default: None)* — Override gradient bucket size (in MB) used by DDP for gradient all-reduce.
- `ddp_broadcast_buffers` *(type: Optional[bool], default: None)* — Control whether BatchNorm buffers are broadcast across ranks each step.
- `ddp_timeout` *(type: int, default: 1800)* — Timeout (in seconds) for distributed collectives before torch raises an error (default is 30 minutes).
- `fsdp` *(type: Union[list[FSDPOption], str, NoneType], default: None)* — Configure Fully Sharded Data Parallel; accepts options like `"full_shard"`, `"shard_grad_op"`, `"hybrid_shard"`, `"hybrid_shard_zero2"`, `"offload"`, `"auto_wrap"`.
- `fsdp_config` *(type: Union[dict[str, Any], str, NoneType], default: None)* — Path to or dictionary with advanced FSDP settings (e.g. `fsdp_version`, `min_num_params`, `transformer_layer_cls_to_wrap`).
- `accelerator_config` *(type: Union[dict, str, NoneType], default: None)* — Pass an Accelerate configuration (file path, dict, or `AcceleratorConfig` instance) to fine-tune process launch, batch splitting, or dispatching.
- `parallelism_config` *(type: Optional[accelerate.parallelism_config.ParallelismConfig], default: None)* — Inject a prebuilt Accelerate parallelism plan (requires `accelerate>=1.10.1`).
- `deepspeed` *(type: Union[dict, str, NoneType], default: None)* — Activate DeepSpeed via config file path or dictionary; supports ZeRO stages, offloading, and other optimisations.

### Reproducibility & Debug
- `seed` *(type: int, default: 42)* — Base random seed passed to `set_seed` for repeatable experiments.
- `data_seed` *(type: Optional[int], default: None)* — Separate seed for data shuffling/sampling; defaults to `seed` when omitted.
- `full_determinism` *(type: bool, default: False)* — Call `enable_full_determinism()` to enforce deterministic CUDA/cuDNN behaviour; use only when reproducibility outweighs throughput.
- `debug` *(type: Union[str, list[debug_utils.DebugOption]], default: )* — Space-separated debug switches: `"underflow_overflow"` to track NaNs/Infs and `"tpu_metrics_debug"` for TPU logging (experimental).

## `Seq2SeqTrainingArguments` Additions
The sequence-to-sequence subclass inherits every field above and adds a few generation-specific toggles:
- `sortish_sampler` *(type: bool, default: False)* — Use the sortish sampler that shuffles batches while keeping examples of similar length together, improving padding efficiency for generation tasks.
- `predict_with_generate` *(type: bool, default: False)* — Call `model.generate()` during evaluation/prediction so your metrics can use generated text (ROUGE, BLEU, etc.).
- `generation_max_length` *(type: Optional[int], default: None)* — Override the `max_length` used when `predict_with_generate=True`; otherwise uses the model config.
- `generation_num_beams` *(type: Optional[int], default: None)* — Override beam width during generation; falls back to the model config if unset.
- `generation_config` *(type: Union[str, Path, GenerationConfig, NoneType], default: None)* — Load a custom `GenerationConfig` (Hub ID, local directory, or object) to control decoding parameters.

## Additional Resources
- Official docs: https://huggingface.co/docs/transformers/main/en/main_classes/trainer
- Source code for defaults: [`transformers/training_args.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py)
- Accelerate docs: https://huggingface.co/docs/accelerate
- DeepSpeed docs: https://huggingface.co/docs/transformers/main/en/main_classes/deepspeed

Use this reference as a companion to your configs or CLI runs and adjust values according to the version of Transformers installed in your environment.
