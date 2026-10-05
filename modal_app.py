"""
Impersona-Env — Modal Training & Inference (LlamaFactory)
=========================================================

Modal app for GPU-accelerated LoRA fine-tuning and inference.
Uses LlamaFactory for proper chat template handling, loss masking,
and a unified interface across all model architectures.

Usage:
    # Deploy (makes functions callable from pipeline.py):
    modal deploy modal_app.py

    # Run the iterative DPO loop entirely on Modal:
    modal run modal_app.py::run_pipeline_local --config-path configs/dpo_pipeline.yaml
"""

import json
import os
from pathlib import Path

import modal


def _local_persona_name() -> str:
    """Persona first name from persona_config.json (local entrypoints only)."""
    config_path = Path(__file__).resolve().parent / "persona_config.json"
    if config_path.exists():
        return json.loads(config_path.read_text()).get("name") or "User"
    return "User"


# ───────────────────────────────────────────────
# Modal App & Infrastructure
# ───────────────────────────────────────────────

MODAL_APP_NAME = "impersona"
MODAL_VOLUME_NAME = "impersona-data"

app = modal.App(MODAL_APP_NAME)

volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")
VOLUME_PATH = "/data"
LF_WORKSPACE = f"{VOLUME_PATH}/lf_workspace"
DDP_TIMEOUT_SECONDS = 7200

# Library versions are pinned to the ones the original runs used, so a fresh
# build gets the same environment.
LLAMAFACTORY = "git+https://github.com/hiyouga/LLaMA-Factory.git@436d26bc28b7c6422c89b63064c5a87e258ed73e"
TRANSFORMERS_GIT = "git+https://github.com/huggingface/transformers.git@a553395766001116a719c82870171f8d6b458c98"
TRAIN_STACK = (
    "torch==2.11.0", "torchvision==0.26.0", "torchaudio==2.11.0",
    "peft==0.18.1", "trl==0.24.0", "accelerate==1.11.0", "datasets==4.0.0",
)

lf_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        f"llamafactory @ {LLAMAFACTORY}",
        "huggingface_hub==1.10.1",
        *TRAIN_STACK,
    )
    .run_commands(
        f"pip install --no-deps 'transformers @ {TRANSFORMERS_GIT}'",
        "pip install bitsandbytes==0.49.2",
    )
    .env({"DISABLE_VERSION_CHECK": "1"})
)

lf_dpo_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu24.04", add_python="3.11")
    .apt_install("git")
    .pip_install(
        f"llamafactory[torch,metrics] @ {LLAMAFACTORY}",
        "transformers==5.2.0",
        "huggingface_hub==1.10.2",
        "bitsandbytes==0.49.2",
        "deepspeed==0.18.9",
        "pyyaml==6.0.3",
        *TRAIN_STACK,
    )
    .env({
        "DISABLE_VERSION_CHECK": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
)

lf_vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu24.04", add_python="3.11")
    .apt_install("git")
    .pip_install("vllm==0.19.0", "httpx", "pyyaml==6.0.3")
    .run_commands(
        f"pip install 'llamafactory @ {LLAMAFACTORY}' transformers==5.2.0 huggingface_hub==1.10.1"
        " peft==0.18.1 trl==0.24.0 accelerate==1.11.0 datasets==4.0.0",
    )
    .env({"DISABLE_VERSION_CHECK": "1"})
)

eval_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("anthropic==1.11.0", "openai==3.24.0", "python-dotenv==1.2.4", "pyyaml==6.0.3")
    .add_local_dir("IMPersona2", remote_path="/root/IMPersona2")
    .add_local_file("configs/models.yaml", remote_path="/root/configs/models.yaml")
)

# Provider API keys for the Modal-side judge / translate calls (see README → Setup).
api_keys_secret = modal.Secret.from_name("impersona-api-keys")


# ───────────────────────────────────────────────
# Template mapping
# ───────────────────────────────────────────────

def _get_lf_template(model_name: str) -> str:
    """Map a HuggingFace model name to a LlamaFactory template name."""
    m = model_name.lower()
    if "llama-3" in m or "llama3" in m:
        return "llama3"
    elif "gemma" in m:
        return "gemma"
    elif "qwen3.5" in m or "qwen3-5" in m or "qwen3_5" in m:
        # Prefer the `qwen3_5` template (ReasoningTemplate). Callers that want
        # no-think behavior should pass `enable_thinking=False` on the inference
        # class — this is the LF-maintainer-recommended approach and produces
        # the official Qwen nothink format (`<|im_start|>assistant\n<think></think>\n\n`).
        return "qwen3_5"
    elif "qwen" in m:
        return "qwen"
    elif "gpt-oss" in m or "gpt_oss" in m:
        return "gpt_oss"
    elif "mistral" in m or "mixtral" in m:
        return "mistral"
    elif "phi" in m:
        return "phi"
    return "default"


# ───────────────────────────────────────────────
# Training Function (LlamaFactory)
# ───────────────────────────────────────────────

def _sft_config(
    model_name: str,
    adapter_name: str,
    template: str,
    num_epochs: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    lora_rank: int,
    lora_alpha: int,
    cutoff_len: int,
    quantization_bit: int | None,
    lora_dropout: float,
    data_dir: str,
    gpus: int,
) -> dict:
    """LlamaFactory LoRA SFT config. Multi-GPU runs add a DDP timeout; 4+ GPUs shard with FSDP."""
    config = {
        "model_name_or_path": model_name,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_target": "all",
        "lora_dropout": lora_dropout,
        "dataset": "impersona",
        "dataset_dir": data_dir,
        "cutoff_len": cutoff_len,
        "preprocessing_num_workers": 16,
        "output_dir": f"{VOLUME_PATH}/{adapter_name}",
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "num_train_epochs": float(num_epochs),
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "bf16": True,
        "max_grad_norm": 1.0,
        "logging_steps": 10,
        "save_strategy": "steps",
        "save_steps": 200,
        "save_total_limit": 3,
        "val_size": 0.02,
        "per_device_eval_batch_size": batch_size,
        "eval_strategy": "steps",
        "eval_steps": 200,
        # Save the eval-best checkpoint as the final adapter so we don't
        # accidentally ship an overfit late-epoch checkpoint. HF Trainer
        # restores this checkpoint at end of training.
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": "none",
        "overwrite_output_dir": True,
        "plot_loss": True,
    }
    if gpus > 1:
        config["ddp_timeout"] = DDP_TIMEOUT_SECONDS
    if gpus >= 4:
        config["fsdp"] = "full_shard auto_wrap"
        # Layer class to wrap is taken from the model's _no_split_modules.
        config["fsdp_config"] = {
            "fsdp_cpu_ram_efficient_loading": True,
            "fsdp_sync_module_states": True,
        }
    if template:
        config["template"] = template
    if quantization_bit is not None:
        config["quantization_bit"] = quantization_bit
        config["upcast_layernorm"] = True
    return config


def _run_lf_training(
    dataset_json: str,
    model_name: str,
    adapter_name: str,
    template: str = "",
    num_epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    lora_rank: int = 64,
    lora_alpha: int = 128,
    cutoff_len: int = 4096,
    quantization_bit: int | None = None,
    lora_dropout: float = 0.05,
) -> dict:
    """LoRA SFT on however many GPUs the container has (in-process on 1, torchrun on more)."""
    import json as _json
    import subprocess
    import torch

    # Empty string = auto-detect from tokenizer; None triggers the mapping
    effective_template = template if template is not None else _get_lf_template(model_name)
    gpus = torch.cuda.device_count()

    print(f"Model: {model_name}")
    print(f"Template: {effective_template or '(auto-detect from tokenizer)'}")
    print(f"GPUs: {gpus}")

    examples = _json.loads(dataset_json)
    print(f"Dataset: {len(examples)} examples")

    data_dir = f"{LF_WORKSPACE}/data"
    os.makedirs(data_dir, exist_ok=True)
    with open(f"{data_dir}/impersona.json", "w") as f:
        _json.dump(examples, f)
    with open(f"{data_dir}/dataset_info.json", "w") as f:
        _json.dump({"impersona": {"file_name": "impersona.json"}}, f, indent=2)

    config = _sft_config(
        model_name, adapter_name, effective_template, num_epochs, batch_size,
        gradient_accumulation_steps, learning_rate, lora_rank, lora_alpha,
        cutoff_len, quantization_bit, lora_dropout, data_dir, gpus,
    )
    output_dir = config["output_dir"]

    if gpus <= 1:
        print(f"Starting LlamaFactory training...")
        print(f"  Output: {output_dir}")
        from llamafactory.train.tuner import run_exp
        run_exp(args=config)
    else:
        # LlamaFactory needs torchrun for multi-GPU training.
        config_path = f"{LF_WORKSPACE}/train_config.json"
        with open(config_path, "w") as f:
            _json.dump(config, f, indent=2)
        os.environ["DISABLE_VERSION_CHECK"] = "1"
        train_script = f"{LF_WORKSPACE}/train_script.py"
        with open(train_script, "w") as f:
            f.write(
                f"import json; from llamafactory.train.tuner import run_exp; "
                f"run_exp(args=json.load(open('{config_path}')))\n"
            )
        cmd = [
            "torchrun",
            f"--nproc_per_node={gpus}",
            "--master_port=29500",
            "--rdzv_backend=c10d",
            train_script,
        ]
        print(f"Launching{' with FSDP full_shard' if gpus >= 4 else ''}: {' '.join(cmd)}", flush=True)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise RuntimeError(f"Training failed with exit code {result.returncode}")

    volume.commit()

    print(f"Training complete. Adapter saved to {output_dir}")
    return {
        "status": "done",
        "examples": len(examples),
        "adapter_path": output_dir,
        "model": model_name,
        "template": effective_template,
        "gpus": gpus,
        "llamafactory_config": config,
    }


# The same training job on 1, 2 or 4 H100s; run_sft_worker.py picks one by model size.
_TRAIN_KWARGS = dict(image=lf_image, timeout=24 * 3600, volumes={VOLUME_PATH: volume}, secrets=[hf_secret])


@app.function(gpu="H100", **_TRAIN_KWARGS)
def train_llamafactory(
    dataset_json: str,
    model_name: str,
    adapter_name: str = "adapter",
    template: str = "",
    num_epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    lora_rank: int = 64,
    lora_alpha: int = 128,
    cutoff_len: int = 4096,
    quantization_bit: int | None = None,
    lora_dropout: float = 0.05,
) -> dict:
    """LoRA SFT on 1 H100."""
    return _run_lf_training(**locals())


@app.function(gpu="H100:2", **_TRAIN_KWARGS)
def train_llamafactory_multi_gpu(
    dataset_json: str,
    model_name: str,
    adapter_name: str = "adapter",
    template: str = "",
    num_epochs: int = 3,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-4,
    lora_rank: int = 64,
    lora_alpha: int = 128,
    cutoff_len: int = 4096,
    quantization_bit: int | None = None,
    lora_dropout: float = 0.05,
) -> dict:
    """LoRA SFT on 2 H100s (torchrun)."""
    return _run_lf_training(**locals())


@app.function(gpu="H100:4", **_TRAIN_KWARGS)
def train_llamafactory_4gpu(
    dataset_json: str,
    model_name: str,
    adapter_name: str = "adapter",
    template: str = "",
    num_epochs: int = 3,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    learning_rate: float = 2e-4,
    lora_rank: int = 64,
    lora_alpha: int = 128,
    cutoff_len: int = 4096,
    quantization_bit: int | None = None,
    lora_dropout: float = 0.05,
) -> dict:
    """LoRA SFT on 4 H100s (torchrun + FSDP), for the largest models."""
    return _run_lf_training(**locals())


# ───────────────────────────────────────────────
# DPO Training (LlamaFactory, single GPU)
# ───────────────────────────────────────────────

@app.function(
    image=lf_dpo_image,
    gpu="H100:2",
    timeout=24 * 3600,
    volumes={VOLUME_PATH: volume},
    secrets=[hf_secret],
)
def train_dpo(
    dataset_json: str,
    model_name: str,
    adapter_name: str = "adapter-dpo",
    init_adapter_name: str = "",
    template: str = "",
    num_epochs: int = 2,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 3e-6,
    lora_rank: int = 32,
    lora_alpha: int = 64,
    cutoff_len: int = 8192,
    pref_beta: float = 0.1,
    pref_loss: str = "sigmoid",
) -> dict:
    """DPO training on 2× H100 (DeepSpeed ZeRO-3, 4-bit reference model)."""
    import json as _json

    effective_template = template if template else _get_lf_template(model_name)

    print(f"Model: {model_name}")
    print(f"Template: {effective_template or '(auto-detect from tokenizer)'}")

    examples = _json.loads(dataset_json)
    print(f"Dataset: {len(examples)} preference pairs")

    data_dir = f"{LF_WORKSPACE}/data"
    os.makedirs(data_dir, exist_ok=True)

    with open(f"{data_dir}/impersona_dpo.json", "w") as f:
        _json.dump(examples, f)

    dataset_info = {
        "impersona_dpo": {
            "file_name": "impersona_dpo.json",
            "ranking": True,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "chosen": "chosen",
                "rejected": "rejected",
            },
        }
    }
    with open(f"{data_dir}/dataset_info.json", "w") as f:
        _json.dump(dataset_info, f, indent=2)

    output_dir = f"{VOLUME_PATH}/{adapter_name}"

    config = {
        "model_name_or_path": model_name,
        "stage": "dpo",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_target": "all",
        "lora_dropout": 0.1,
        "pref_beta": pref_beta,
        "pref_loss": pref_loss,
        "ref_model_quantization_bit": 4,
        "dataset": "impersona_dpo",
        "dataset_dir": data_dir,
        "cutoff_len": cutoff_len,
        "preprocessing_num_workers": 16,
        "output_dir": output_dir,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "num_train_epochs": float(num_epochs),
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "bf16": True,
        "gradient_checkpointing": True,
        "max_grad_norm": 1.0,
        "logging_steps": 10,
        "save_steps": 25,
        "save_total_limit": 3,
        "report_to": "none",
        "overwrite_output_dir": True,
        "plot_loss": True,
        "val_size": 0.05,
        "per_device_eval_batch_size": 1,
        "eval_strategy": "steps",
        "eval_steps": 25,
    }
    if effective_template:
        config["template"] = effective_template

    # Continued DPO: load an existing adapter as initialization rather than
    # creating a fresh one. Required for "round 2" training that builds on
    # round 1's learned weights.
    if init_adapter_name:
        init_adapter_path = f"{VOLUME_PATH}/{init_adapter_name}"
        config["adapter_name_or_path"] = init_adapter_path
        config["create_new_adapter"] = False
        print(f"Continued DPO: initializing from {init_adapter_path}")

    ds_config = {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "zero_allow_untested_optimizer": True,
        "fp16": {
            "enabled": "auto",
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1,
        },
        "bf16": {"enabled": "auto"},
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": False,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True,
        },
    }
    ds_config_path = f"{LF_WORKSPACE}/ds_z3_config.json"
    with open(ds_config_path, "w") as f:
        _json.dump(ds_config, f, indent=2)

    config["deepspeed"] = ds_config_path
    config["ddp_timeout"] = DDP_TIMEOUT_SECONDS

    import yaml
    config_path = f"{LF_WORKSPACE}/train_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    os.environ["DISABLE_VERSION_CHECK"] = "1"

    import subprocess
    print(f"Launching DPO training (2× H100, DeepSpeed ZeRO-3, 4-bit reference model)...", flush=True)
    proc = subprocess.run(["llamafactory-cli", "train", config_path])
    if proc.returncode != 0:
        raise RuntimeError(f"llamafactory-cli train failed (exit {proc.returncode})")

    volume.commit()

    print(f"DPO training complete. Adapter saved to {output_dir}")
    return {
        "status": "done",
        "examples": len(examples),
        "adapter_path": output_dir,
        "model": model_name,
        "template": effective_template,
        "llamafactory_config": config,
    }


# ───────────────────────────────────────────────
# Inference (vLLM — batched, high-throughput)
# ───────────────────────────────────────────────

@app.cls(
    image=lf_vllm_image,
    gpu="H100",
    timeout=1200,
    startup_timeout=1800,
    volumes={VOLUME_PATH: volume},
    secrets=[hf_secret],
    # Long window so the faster IMPersona container doesn't scale down while
    # waiting for a slower (larger) partner model's cold start to finish.
    # Cold starts can differ by 7-10 min.
    scaledown_window=1200,
    max_containers=1,
)
@modal.concurrent(max_inputs=75)
class VLLMInference:
    """LlamaFactory API server with vLLM backend for batched inference.

    Uses llamafactory-cli api with infer_backend=vllm for template-consistent,
    high-throughput inference with automatic continuous batching.

    `enable_thinking` controls LlamaFactory's ReasoningTemplate for reasoning
    models — False suppresses thinking at inference; other templates ignore it.
    """
    model_name: str = modal.parameter()
    adapter_name: str = modal.parameter(default="")
    template: str = modal.parameter(default="")  # "" = detect from model_name
    enable_thinking: bool = modal.parameter(default=True)

    @modal.enter()
    def start_server(self):
        import subprocess
        import sys
        import time
        import httpx
        import yaml

        adapter_path = f"{VOLUME_PATH}/{self.adapter_name}"
        has_adapter = bool(self.adapter_name) and os.path.exists(adapter_path) and os.path.isdir(adapter_path)

        print(f"Starting LlamaFactory API (vLLM backend): {self.model_name}", flush=True)
        template = self.template or _get_lf_template(self.model_name)
        print(f"  Template: {template}", flush=True)
        print(f"  Adapter: {adapter_path} (exists={has_adapter})", flush=True)
        print(f"  enable_thinking: {self.enable_thinking}", flush=True)

        config = {
            "model_name_or_path": self.model_name,
            "template": template,
            "enable_thinking": bool(self.enable_thinking),
            "infer_backend": "vllm",
            "vllm_maxlen": 32768,
            "vllm_max_lora_rank": 64,
            "trust_remote_code": True,
        }
        if has_adapter:
            config["adapter_name_or_path"] = adapter_path
            config["finetuning_type"] = "lora"

        config_path = "/tmp/lf_api_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        env = {**os.environ, "API_PORT": "8000", "DISABLE_VERSION_CHECK": "1"}
        self._server_proc = subprocess.Popen(
            ["llamafactory-cli", "api", config_path],
            stdout=sys.stdout, stderr=sys.stderr, env=env,
        )

        self._client = httpx.Client(base_url="http://localhost:8000", timeout=120)
        for i in range(3600):
            try:
                r = self._client.get("/v1/models")
                if r.status_code == 200:
                    print(f"LlamaFactory API server ready after {i}s", flush=True)
                    return
            except httpx.ConnectError:
                pass
            time.sleep(1)
        raise RuntimeError("LlamaFactory API server did not start within 3600s")

    @modal.exit()
    def stop_server(self):
        if hasattr(self, "_server_proc"):
            self._server_proc.terminate()

    @modal.method()
    def generate(
        self,
        system_prompt: str,
        conversation_input: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ) -> str:
        body = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": conversation_input},
            ],
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "chat_template_kwargs": {"enable_thinking": bool(self.enable_thinking)},
        }

        resp = self._client.post("/v1/chat/completions", json=body)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


# ───────────────────────────────────────────────
# vLLM Partner Inference (for batched eval)
# ───────────────────────────────────────────────

@app.cls(
    image=lf_vllm_image,
    gpu="H100",
    timeout=7200,
    startup_timeout=1800,
    volumes={VOLUME_PATH: volume},
    secrets=[hf_secret],
    # Long window so the container doesn't idle-down between warmup and
    # the first real inference call, or during long batches of eval work.
    scaledown_window=1200,
    max_containers=2,
)
@modal.concurrent(max_inputs=75)
class VLLMPartner:
    """LlamaFactory API server with vLLM backend for partner model.

    Accepts an OpenAI-style messages list to match the AIPartnerClient
    interface used by run_conversation. Template handled by LlamaFactory.
    Optionally loads a LoRA adapter from the Modal volume.

    `enable_thinking` controls the ReasoningTemplate behavior: False prefills
    empty `<think></think>` on the prompt side (suppresses generation of new
    thinking content at inference); True (default) lets the model think.
    """
    model_name: str = modal.parameter()
    template: str = modal.parameter(default="")  # "" = detect from model_name
    quantization_bit: int = modal.parameter(default=0)
    adapter_name: str = modal.parameter(default="")
    enable_thinking: bool = modal.parameter(default=True)

    @modal.enter()
    def start_server(self):
        import subprocess
        import sys
        import time
        import httpx
        import yaml

        adapter_path = f"{VOLUME_PATH}/{self.adapter_name}"
        has_adapter = (
            bool(self.adapter_name)
            and os.path.exists(adapter_path)
            and os.path.isdir(adapter_path)
        )

        print(f"Starting LlamaFactory API (vLLM backend) for partner: {self.model_name}", flush=True)
        template = self.template or _get_lf_template(self.model_name)
        print(f"  Template: {template}", flush=True)
        print(f"  Adapter:  {adapter_path if has_adapter else '(none)'}", flush=True)
        print(f"  enable_thinking: {self.enable_thinking}", flush=True)

        config = {
            "model_name_or_path": self.model_name,
            "template": template,
            "enable_thinking": bool(self.enable_thinking),
            "infer_backend": "vllm",
            "vllm_maxlen": 32768,
            "vllm_max_lora_rank": 64,
            "trust_remote_code": True,
        }
        if self.quantization_bit > 0:
            config["quantization_bit"] = self.quantization_bit
        if has_adapter:
            config["adapter_name_or_path"] = adapter_path
            config["finetuning_type"] = "lora"

        config_path = "/tmp/lf_partner_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        env = {**os.environ, "API_PORT": "8000", "DISABLE_VERSION_CHECK": "1"}
        self._server_proc = subprocess.Popen(
            ["llamafactory-cli", "api", config_path],
            stdout=sys.stdout, stderr=sys.stderr, env=env,
        )

        self._client = httpx.Client(base_url="http://localhost:8000", timeout=120)
        for i in range(3600):
            try:
                r = self._client.get("/v1/models")
                if r.status_code == 200:
                    print(f"LlamaFactory partner API ready after {i}s", flush=True)
                    return
            except httpx.ConnectError:
                pass
            time.sleep(1)
        raise RuntimeError("LlamaFactory partner API did not start within 3600s")

    @modal.exit()
    def stop_server(self):
        if hasattr(self, "_server_proc"):
            self._server_proc.terminate()

    @modal.method()
    def generate(
        self,
        system_prompt: str,
        messages_json: str,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> str:
        openai_messages = [{"role": "system", "content": system_prompt}]
        openai_messages += json.loads(messages_json)

        resp = self._client.post("/v1/chat/completions", json={
            "model": self.model_name,
            "messages": openai_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": bool(self.enable_thinking)},
        })
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    @modal.method()
    def stream_generate(
        self,
        system_prompt: str,
        messages_json: str,
        max_tokens: int = 16384,
        temperature: float = 0.7,
    ):
        """Stream tokens from vLLM via OpenAI-style SSE. Yields content chunks."""
        import httpx

        openai_messages = [{"role": "system", "content": system_prompt}]
        openai_messages += json.loads(messages_json)

        with httpx.Client(base_url="http://localhost:8000", timeout=600) as client:
            with client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": self.model_name,
                    "messages": openai_messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": True,
                    "chat_template_kwargs": {"enable_thinking": bool(self.enable_thinking)},
                },
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    s = line.strip()
                    if not s.startswith("data:"):
                        continue
                    payload = s[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        obj = json.loads(payload)
                        delta = obj.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except Exception:
                        continue

    @modal.method()
    def warmup(self) -> str:
        """Trivial call used to force container + model load from outside."""
        return "ok"


# ───────────────────────────────────────────────
# vLLM Batched Eval Orchestrator
# ───────────────────────────────────────────────

@app.function(
    image=eval_image,
    secrets=[api_keys_secret, hf_secret],
    volumes={VOLUME_PATH: volume},
    timeout=86400,
)
def run_eval_vllm(config_json: str):
    """Run eval using two separate vLLM containers (IMPersona + partner).

    Each conversation runs in its own thread; vLLM batches concurrent
    requests automatically.
    """
    import asyncio
    import threading
    import time
    from datetime import datetime
    from pathlib import Path

    import sys
    if "/root" not in sys.path:
        sys.path.insert(0, "/root")

    from IMPersona2.eval import (
        load_scenarios,
        evaluate_coherence,
        evaluate_disagreements,
        generate_viewer_html,
        run_conversation,
    )

    config = json.loads(config_json)
    persona_name = config.get("persona_name") or "User"
    impersona_model_name = config["impersona_model_name"]
    impersona_adapter_name = config.get("impersona_adapter_name", "")
    impersona_template = config.get("impersona_template") or _get_lf_template(impersona_model_name)
    partner_model = config["ai_partner_modal_model"]
    partner_template = config.get("partner_template") or _get_lf_template(partner_model)
    partner_thinking = bool(config.get("ai_partner_enable_thinking", False))
    partner_quantization_bit = config.get("partner_quantization_bit", 0)
    partner_adapter = config.get("partner_adapter", "")
    category = config.get("category")
    quick = config.get("quick", False)
    num_runs = config.get("num_runs", 1)
    max_messages = config.get("max_messages", 20)
    run_label = config.get("run_label", "eval-vllm")
    max_concurrent = config.get("max_concurrent", 50)
    critic_mode = config.get("critic_mode", False)
    include_prior_cot = config.get("include_prior_cot", False)

    scenarios_path = "/root/IMPersona2/eval_scenarios.json"
    scenarios = load_scenarios(scenarios_path, scenario_ids=None, category=category, quick=quick)
    if not scenarios:
        print("ERROR: No scenarios found", flush=True)
        return {"status": "error", "error": "No scenarios found"}

    volume.reload()

    run_dir = Path(f"{VOLUME_PATH}/eval_runs/{run_label}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    volume.commit()

    # Resume
    partial_path = run_dir / "results_partial.json"
    completed_conversations = []
    completed_ids = set()
    if partial_path.exists():
        try:
            partial = json.loads(partial_path.read_text())
            completed_conversations = partial.get("conversations", [])
            completed_ids = {c["scenario"]["id"] for c in completed_conversations}
            print(f"Resuming: {len(completed_ids)} scenarios already completed", flush=True)
        except Exception:
            pass

    remaining = [(s, ri) for s in scenarios for ri in range(num_runs)
                 if s["id"] not in completed_ids]

    if not remaining:
        print("All scenarios already completed!", flush=True)
    else:
        total = len(remaining)
        print(f"\nEval (vLLM batched): {total} conversations to run "
              f"({len(completed_ids)} already done)", flush=True)
        print(f"  IMPersona: {impersona_model_name} + {impersona_adapter_name} (vLLM)", flush=True)
        print(f"  Partner:   {partner_model} (vLLM)", flush=True)
        print(f"  Concurrency: {max_concurrent} conversations", flush=True)
        if critic_mode:
            print(f"  Critic mode: ON (authenticity addendum appended to relationship context)", flush=True)
        print("", flush=True)

        # Instantiate vLLM containers
        impersona_vllm = VLLMInference(
            model_name=impersona_model_name,
            adapter_name=impersona_adapter_name,
            template=impersona_template,
        )
        partner_vllm = VLLMPartner(
            model_name=partner_model,
            template=partner_template,
            quantization_bit=partner_quantization_bit,
            adapter_name=partner_adapter,
            enable_thinking=partner_thinking,
        )

        # ── Sequential warmup: partner first (slow), then IMPersona ──
        # A large partner can take far longer to cold start than the IMPersona.
        # If both fire in parallel, the IMPersona container idles and can scale
        # down before the partner is ready, forcing a second cold start.
        # Firing them sequentially avoids the wasted IMPersona compute.
        print(f"Pre-warming partner ({partner_model}; large models can take 20+ min to cold start)...", flush=True)
        partner_warmup = partner_vllm.generate.spawn(
            system_prompt="warmup",
            messages_json=json.dumps([{"role": "user", "content": "hi"}]),
            max_tokens=1,
        )
        partner_warmup.get()
        print(f"Partner warm. Pre-warming IMPersona ({impersona_model_name})...", flush=True)
        impersona_warmup = impersona_vllm.generate.spawn(
            system_prompt="warmup",
            conversation_input="hi",
            max_new_tokens=1,
        )
        impersona_warmup.get()
        print("Both models warm. Starting conversations.", flush=True)

        # Wrapper classes that call .remote() on the vLLM containers
        class _VLLMIMPersona:
            def __init__(self):
                self.model = impersona_model_name
            def generate(self, system_prompt, conversation_input,
                         max_new_tokens=512, temperature=0.7,
                         top_p=0.9, repetition_penalty=1.1):
                return impersona_vllm.generate.remote(
                    system_prompt=system_prompt,
                    conversation_input=conversation_input,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )

        class _VLLMPartnerWrapper:
            def __init__(self):
                self.model = partner_model
                self._num_calls = 0
            def generate(self, system_prompt, messages, max_tokens=4096):
                self._num_calls += 1
                return partner_vllm.generate.remote(
                    system_prompt=system_prompt,
                    messages_json=json.dumps(messages),
                    max_tokens=max_tokens,
                )
            def get_usage(self):
                return {"model": self.model, "num_calls": self._num_calls}

        save_lock = threading.Lock()

        def _run_one(scenario, run_idx):
            chat_model = _VLLMIMPersona()
            ai_partner = _VLLMPartnerWrapper()
            result = run_conversation(
                scenario=scenario,
                chat_model=chat_model,
                ai_partner=ai_partner,
                persona_name=persona_name,
                max_messages=max_messages,
                show_live=False,
                critic_mode=critic_mode,
                include_prior_cot=include_prior_cot,
            )
            if num_runs > 1:
                result["run_index"] = run_idx
            return result

        semaphore = asyncio.Semaphore(max_concurrent)

        async def _run_one_async(scenario, run_idx, idx):
            async with semaphore:
                try:
                    result = await asyncio.to_thread(_run_one, scenario, run_idx)
                    with save_lock:
                        completed_conversations.append(result)
                        partial_path.write_text(
                            json.dumps({"conversations": completed_conversations},
                                       indent=2, ensure_ascii=False))
                        n = len(completed_conversations)
                    await volume.commit.aio()
                    print(f"  [{n}/{len(scenarios)}] {scenario['name']} ✓ "
                          f"({result['stats']['total_messages']} msgs, "
                          f"{result['stats']['elapsed_seconds']:.0f}s)", flush=True)
                    return result
                except Exception as e:
                    print(f"  [?/{len(scenarios)}] {scenario['name']} FAILED: "
                          f"{type(e).__name__}: {e}", flush=True)
                    return None

        async def _run_all():
            tasks = [
                _run_one_async(s, ri, i)
                for i, (s, ri) in enumerate(remaining)
            ]
            return await asyncio.gather(*tasks)

        conv_start = time.time()
        results = asyncio.run(_run_all())
        failed_count = sum(1 for r in results if r is None)
        elapsed = time.time() - conv_start
        print(f"\nConversations done: {len(completed_conversations)} ok, "
              f"{failed_count} failed in {elapsed:.1f}s", flush=True)

    # LLM judges
    conversations = completed_conversations
    print("Running LLM judges (coherence + disagreement)...", flush=True)
    coherence_summary = None
    disagreement_summary = None
    if conversations:
        coherence_summary = evaluate_coherence(conversations, persona_name=persona_name)
        disagreement_summary = evaluate_disagreements(conversations, persona_name=persona_name)

    results_bundle = {
        "metadata": {
            "run_label": run_label,
            "persona_name": persona_name,
            "impersona_model": impersona_model_name,
            "impersona_adapter": impersona_adapter_name,
            "ai_partner_model": partner_model,
            "ai_partner_adapter": partner_adapter,
            "max_messages": max_messages,
            "num_scenarios": len(scenarios),
            "num_runs": num_runs,
            "num_completed": len(conversations),
            "critic_mode": critic_mode,
            "include_prior_cot": include_prior_cot,
            "eval_mode": "vllm_batched",
            "timestamp": datetime.now().isoformat(),
        },
        "evaluations": {
            "coherence": coherence_summary,
            "disagreements": disagreement_summary,
        },
        "conversations": conversations,
    }

    (run_dir / "results.json").write_text(
        json.dumps(results_bundle, indent=2, ensure_ascii=False))
    viewer_html = generate_viewer_html(results_bundle)
    (run_dir / "viewer.html").write_text(viewer_html)

    if partial_path.exists():
        partial_path.unlink()

    volume.commit()
    print(f"\nDone — results saved to {run_dir}", flush=True)
    return {"status": "completed", "num_completed": len(conversations)}


# ───────────────────────────────────────────────
# DPO data generation (server-side)
# ───────────────────────────────────────────────
# Wraps IMPersona2.generate_dpo_dataset.run_generation so the orchestrator
# (`run_pipeline`) can build DPO datasets entirely in the cloud — no local
# Python needed. Output is written in sharegpt format ready for `train_dpo`.

@app.function(
    image=eval_image,
    timeout=6 * 3600,
    volumes={VOLUME_PATH: volume},
    secrets=[api_keys_secret],
)
def generate_dpo_data(
    eval_run_label: str,
    output_path: str,
    partner_model: str,
    partner_adapter: str = "",
    persona_name: str = "User",
    max_workers: int = 20,
    filter_persona_name: bool = True,
) -> dict:
    """Build a DPO dataset from an eval run, on-policy via VLLMPartner rewrites.

    Args:
        eval_run_label: Run label of a prior eval (results.json must already
            exist on the volume at /data/eval_runs/<eval_run_label>/results.json).
        output_path: Absolute volume path where the sharegpt-formatted DPO JSON
            will be written (e.g. /data/dpo_data/pipeline-r1.json). A sibling
            *_raw.json with the full alpaca-+-metadata view is also written
            for debugging/inspection.
        partner_model: HuggingFace model name of the rewrite/partner model.
        partner_adapter: Adapter loaded on top of partner_model. Empty = vanilla
            base (round 1); later rounds pass the previous round's DPO adapter
            for true on-policy rewrites.
        persona_name: Used by the translator prompt and the contamination
            filter below. The rewrite prompt itself stays anonymized.
        max_workers: Translate+rewrite thread-pool size.
        filter_persona_name: If True, drop pairs whose `chosen` text literally
            contains the persona name. Rewrites sometimes address the user by
            name while `rejected` never does, so DPO would learn the name as
            a preference signal.

    Returns: summary dict with output paths and counts.
    """
    import argparse as _argparse
    import json as _json
    import os as _os
    import re as _re
    from IMPersona2.generate_dpo_dataset import run_generation
    from IMPersona2.prepare_dpo_dataset import convert_to_sharegpt_dpo

    results_path = f"{VOLUME_PATH}/eval_runs/{eval_run_label}/results.json"
    if not _os.path.exists(results_path):
        raise FileNotFoundError(
            f"Eval results not found at {results_path}. "
            f"Did the eval step for run '{eval_run_label}' actually complete?"
        )

    # Raw alpaca-format output (with full metadata) — kept for debugging
    raw_path = output_path.replace(".json", "_raw.json")
    _os.makedirs(_os.path.dirname(raw_path) or ".", exist_ok=True)

    # Mimic the CLI argparse namespace expected by run_generation
    args = _argparse.Namespace(
        run=_os.path.dirname(results_path),
        output=raw_path,
        rewrite_modal=True,
        rewrite_modal_model=partner_model,
        rewrite_modal_adapter=partner_adapter,
        rewrite_api_model=None,
        rewrite_reasoning=None,
        rewrite_local_model=None,
        rewrite_template=None,
        rewrite_preset=None,
        rewrite_quantization_bit=None,
        rewrite_local_label=None,
        scenario_ids=None,
        limit_conversations=None,
        max_examples=None,
        max_workers=max_workers,
        allow_identical=False,
        empty_think_prefix="no",
    )

    adapter_note = f" + adapter '{partner_adapter}'" if partner_adapter else " (no adapter)"
    print(f"Generating DPO data from eval '{eval_run_label}'", flush=True)
    print(f"  Partner model:        {partner_model}{adapter_note}", flush=True)
    print(f"  Persona-name filter:  {filter_persona_name} (name='{persona_name}')", flush=True)
    print(f"  Raw output:           {raw_path}", flush=True)
    print(f"  Sharegpt output:      {output_path}", flush=True)

    summary = run_generation(args)
    volume.commit()

    # Load alpaca examples back, apply persona-name filter, convert to sharegpt
    with open(raw_path, "r", encoding="utf-8") as f:
        examples = _json.load(f)

    n_before_filter = len(examples)
    if filter_persona_name and persona_name:
        name_re = _re.compile(rf"\b{_re.escape(persona_name)}\b")
        examples = [ex for ex in examples if not name_re.search(ex.get("chosen", ""))]
    n_filtered = n_before_filter - len(examples)
    if n_filtered:
        print(
            f"  Filter dropped {n_filtered}/{n_before_filter} pairs whose "
            f"chosen mentioned '{persona_name}'",
            flush=True,
        )

    sharegpt = convert_to_sharegpt_dpo(examples)
    _os.makedirs(_os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        _json.dump(sharegpt, f, indent=2, ensure_ascii=False)
        f.write("\n")
    volume.commit()

    print(f"DPO generation complete:", flush=True)
    print(f"  raw alpaca:  {raw_path} ({n_before_filter} pairs)", flush=True)
    print(f"  sharegpt:    {output_path} ({len(sharegpt)} pairs after filter)", flush=True)

    return {
        "eval_run_label": eval_run_label,
        "output_path": output_path,
        "raw_path": raw_path,
        "num_pairs_raw": n_before_filter,
        "num_pairs_filtered_out": n_filtered,
        "num_pairs_final": len(sharegpt),
        **summary,
    }


# ───────────────────────────────────────────────
# Iterative DPO pipeline orchestrator (durable, cloud-side)
# ───────────────────────────────────────────────
# Loops N rounds of:  eval  →  generate DPO data  →  train DPO.
# Per-round identifiers are auto-derived from `pipeline_name`. State is
# checkpointed to the volume so re-running after a crash resumes from the last
# completed step. Round-N>1 always continues DPO from round-(N−1)'s adapter
# via train_dpo's `init_adapter_name`.

@app.function(
    image=eval_image,
    timeout=24 * 3600,
    volumes={VOLUME_PATH: volume},
    secrets=[api_keys_secret],
)
def run_pipeline(config_json: str) -> dict:
    """Orchestrate iterative DPO across N rounds.

    See ``configs/dpo_pipeline.yaml`` for the expected config schema.

    Per-round naming:
        eval run label:  {pipeline_name}-r{N}-eval
        DPO dataset:     /data/dpo_data/{pipeline_name}-r{N}.json
        adapter:         {pipeline_name}-r{N}-adapter
        state file:      /data/pipelines/{pipeline_name}/state.json

    Resume semantics: re-running the same pipeline_name skips any step the
    state file marks as "done". Steps that crashed are retried from scratch
    (the eval step itself has scenario-level resume via results_partial.json).
    """
    import json as _json
    import os as _os

    cfg = _json.loads(config_json)
    pipeline_name = cfg["pipeline_name"]
    num_rounds = int(cfg["num_rounds"])

    state_dir = f"{VOLUME_PATH}/pipelines/{pipeline_name}"
    state_path = f"{state_dir}/state.json"
    _os.makedirs(state_dir, exist_ok=True)

    def _load_state():
        if _os.path.exists(state_path):
            with open(state_path) as f:
                return _json.load(f)
        return {"pipeline_name": pipeline_name, "rounds": {}}

    def _save_state(s):
        with open(state_path, "w") as f:
            _json.dump(s, f, indent=2)
        volume.commit()

    state = _load_state()
    completed_rounds = [k for k, v in state.get("rounds", {}).items()
                        if v.get("train") == "done"]
    print(f"╔═══ Pipeline: {pipeline_name} ═══", flush=True)
    print(f"║ Rounds: {num_rounds}", flush=True)
    print(f"║ State:  {state_path}", flush=True)
    print(f"║ Resume: {len(completed_rounds)} round(s) already complete: {completed_rounds}", flush=True)

    persona_name = cfg.get("persona_name") or "User"
    partner_model = cfg["partner"]["model"]
    base_adapter = cfg["partner"].get("base_adapter", "")
    eval_cfg = cfg["eval"]
    gen_cfg = cfg.get("dpo_gen", {})
    train_cfg = cfg["training"]
    impersona = cfg["impersona"]

    for n in range(1, num_rounds + 1):
        round_key = f"round_{n}"
        rstate = state["rounds"].setdefault(round_key, {})
        prev_adapter = base_adapter if n == 1 else f"{pipeline_name}-r{n-1}-adapter"

        print(f"\n┌─── Round {n}/{num_rounds} ───", flush=True)
        print(f"│ partner adapter for this round: '{prev_adapter or '(base / none)'}'", flush=True)

        # ─── Step 1: Eval ───
        eval_label = f"{pipeline_name}-r{n}-eval"
        if rstate.get("eval") == "done":
            print(f"│ [eval]  ✓ already done ({eval_label}) — skipping", flush=True)
        else:
            print(f"│ [eval]  running ({eval_label})...", flush=True)
            eval_config = {
                "modal_app_name": MODAL_APP_NAME,
                "persona_name": persona_name,
                "ai_partner_modal_model": partner_model,
                "partner_template": eval_cfg.get("partner_template", ""),
                "ai_partner_enable_thinking": bool(eval_cfg.get("partner_thinking", False)),
                "partner_quantization_bit": int(eval_cfg.get("partner_quantization_bit", 0)),
                "partner_adapter": prev_adapter,
                "impersona_model_name": impersona["model"],
                "impersona_adapter_name": impersona["adapter"],
                "impersona_template": impersona.get("template", ""),
                "num_runs": int(eval_cfg["num_trials"]),
                "max_messages": int(eval_cfg.get("max_messages", 20)),
                "max_concurrent": int(eval_cfg.get("max_concurrent", 50)),
                "run_label": eval_label,
                "category": eval_cfg.get("category"),
                "quick": False,
                "critic_mode": bool(eval_cfg.get("critic_mode", False)),
                "include_prior_cot": bool(eval_cfg.get("include_prior_cot", False)),
            }
            run_eval_vllm.remote(config_json=_json.dumps(eval_config))
            rstate["eval"] = "done"
            rstate["eval_label"] = eval_label
            _save_state(state)
            print(f"│ [eval]  ✓ completed", flush=True)

        # ─── Step 2: Generate DPO data ───
        dpo_path = f"{VOLUME_PATH}/dpo_data/{pipeline_name}-r{n}.json"
        if rstate.get("gen") == "done":
            n_pairs = rstate.get("num_pairs", "?")
            print(f"│ [gen]   ✓ already done ({n_pairs} pairs at {dpo_path}) — skipping", flush=True)
        else:
            print(f"│ [gen]   running...", flush=True)
            gen_result = generate_dpo_data.remote(
                eval_run_label=eval_label,
                output_path=dpo_path,
                partner_model=partner_model,
                partner_adapter=prev_adapter,
                persona_name=persona_name,
                max_workers=int(gen_cfg.get("max_workers", 20)),
                filter_persona_name=bool(gen_cfg.get("filter_persona_name", True)),
            )
            rstate["gen"] = "done"
            rstate["dpo_path"] = dpo_path
            rstate["num_pairs"] = gen_result.get("num_pairs_final", 0)
            _save_state(state)
            print(f"│ [gen]   ✓ produced {rstate['num_pairs']} pairs", flush=True)

        # ─── Step 3: Train DPO ───
        adapter_name = f"{pipeline_name}-r{n}-adapter"
        if rstate.get("train") == "done":
            print(f"│ [train] ✓ already done ({adapter_name}) — skipping", flush=True)
        else:
            print(f"│ [train] running ({adapter_name})...", flush=True)
            # Re-read DPO data so the JSON string is fresh
            with open(dpo_path, "r", encoding="utf-8") as f:
                dataset_json = f.read()
            train_dpo.remote(
                dataset_json=dataset_json,
                model_name=partner_model,
                adapter_name=adapter_name,
                init_adapter_name=prev_adapter,
                template=eval_cfg.get("partner_template", ""),
                num_epochs=int(train_cfg.get("epochs", 2)),
                batch_size=int(train_cfg.get("batch_size", 1)),
                gradient_accumulation_steps=int(train_cfg.get("grad_accum", 8)),
                learning_rate=float(train_cfg["lr"]),
                lora_rank=int(train_cfg["lora_rank"]),
                lora_alpha=int(train_cfg["lora_alpha"]),
                cutoff_len=int(train_cfg.get("cutoff_len", 4096)),
                pref_beta=float(train_cfg.get("pref_beta", 0.1)),
                pref_loss=train_cfg.get("pref_loss", "sigmoid"),
            )
            rstate["train"] = "done"
            rstate["adapter"] = adapter_name
            _save_state(state)
            print(f"│ [train] ✓ adapter saved to /data/{adapter_name}", flush=True)

        print(f"└─── Round {n} complete ───", flush=True)

    final_adapter = f"{pipeline_name}-r{num_rounds}-adapter"
    print(f"\n╔═══ Pipeline complete: {pipeline_name} ═══", flush=True)
    print(f"║ Final adapter:  /data/{final_adapter}", flush=True)
    print(f"║ All {num_rounds} round(s): ✓", flush=True)

    return {
        "pipeline_name": pipeline_name,
        "rounds_completed": num_rounds,
        "final_adapter": final_adapter,
        "state": state,
    }


# ───────────────────────────────────────────────
# CLI entry points
# ───────────────────────────────────────────────

@app.local_entrypoint()
def start_eval_vllm(
    run_label: str = "eval-vllm",
    partner_model: str = "",
    partner_template: str = "",
    partner_quantization_bit: int = 0,
    partner_adapter: str = "",
    impersona_model: str = "",
    adapter: str = "adapter-user-model",
    template: str = "",
    max_messages: int = 20,
    num_runs: int = 1,
    category: str = "",
    max_concurrent: int = 50,
    critic_mode: bool = False,
    include_prior_cot: bool = False,
):
    """Fire-and-forget eval using LlamaFactory + vLLM batched inference.

    Runs two LlamaFactory API containers (IMPersona + partner) with vLLM
    backend for automatic continuous batching. Template-consistent.

    When ``critic_mode`` is True, a neutral addendum is appended to the
    IMPersona's relationship context to encourage more authentic internal
    reactions (denser DPO signal). See
    ``IMPersona2.eval._EVAL_RELATIONSHIP_CRITIC_ADDENDUM``.

    When ``include_prior_cot`` is True, each generated ``<think>cot</think>
    response`` is stored in conversation history *with* the CoT block (not
    CoT-stripped). Required when the IMPersona adapter was trained with
    prior CoTs exposed in history (see IMPersona2/add_prior_cot.py).
    """
    from IMPersona2.model_config import model_for, template_for
    partner_model = partner_model or model_for("assistant", "base")
    impersona_model = impersona_model or model_for("user_model", "base")
    config = {
        "modal_app_name": MODAL_APP_NAME,
        "persona_name": _local_persona_name(),
        "ai_partner_modal_model": partner_model,
        "ai_partner_enable_thinking": bool(model_for("assistant", "thinking")),
        "partner_template": partner_template or template_for("assistant", partner_model),
        "partner_quantization_bit": partner_quantization_bit,
        "partner_adapter": partner_adapter,
        "impersona_model_name": impersona_model,
        "impersona_adapter_name": adapter,
        "impersona_template": template or template_for("user_model", impersona_model),
        "num_runs": num_runs,
        "max_messages": max_messages,
        "max_concurrent": max_concurrent,
        "run_label": run_label,
        "category": category or None,
        "quick": False,
        "critic_mode": critic_mode,
        "include_prior_cot": include_prior_cot,
    }
    app_name = MODAL_APP_NAME
    try:
        deployed_fn = modal.Function.from_name(app_name, "run_eval_vllm")
        fc = deployed_fn.spawn(config_json=json.dumps(config))
    except modal.exception.NotFoundError:
        print(f"ERROR: App '{app_name}' is not deployed. Run this first:")
        print(f"  modal deploy modal_app.py")
        print(f"\nThen re-run this command.")
        return

    print(f"Eval started on Modal (vLLM batched)!")
    print(f"  Function call ID: {fc.object_id}")
    print(f"  Run label: {run_label}")
    print(f"  Partner: {partner_model} (vLLM)")
    print(f"  IMPersona: {impersona_model} + {adapter} (vLLM)")
    print(f"  Concurrency: {max_concurrent} conversations")
    if critic_mode:
        print(f"  Critic mode: ON (authenticity addendum active)")
    if include_prior_cot:
        print(f"  Prior CoT: EXPOSED (history includes <think>cot</think>response)")
    print(f"  Results: /data/eval_runs/{run_label}/")
    print(f"\nYou can safely close your laptop now.")
    print(f"Check progress in the Modal dashboard or run:")
    print(f"  modal run modal_app.py::download_eval --run-label {run_label}")


@app.local_entrypoint()
def download_eval(run_label: str = "eval"):
    """Download eval results from Modal volume to local eval_output/."""
    import shutil

    remote_dir = f"{VOLUME_PATH}/eval_runs/{run_label}"
    local_dir = Path("eval_output") / run_label
    local_dir.mkdir(parents=True, exist_ok=True)

    for fname in ["results.json", "viewer.html", "config.json", "results_partial.json"]:
        remote_path = f"{remote_dir}/{fname}"
        try:
            data = b""
            for chunk in volume.read_file(remote_path):
                data += chunk
            (local_dir / fname).write_bytes(data)
            print(f"  Downloaded {fname} ({len(data)} bytes)")
        except Exception as e:
            if "results_partial" not in fname:
                print(f"  {fname}: not found ({e})")

    print(f"\nResults saved to {local_dir}/")
    if (local_dir / "viewer.html").exists():
        print(f"Open viewer: file://{local_dir.resolve()}/viewer.html")


@app.local_entrypoint()
def run_pipeline_local(config_path: str = "configs/dpo_pipeline.yaml"):
    """Load a YAML pipeline config and spawn the cloud `run_pipeline` orchestrator.

    Fire-and-forget: prints the function-call ID and exits. The orchestrator
    survives a closed laptop because it runs as a deployed Modal function.
    Re-run this same command to resume a partially-completed pipeline (state
    is checkpointed to the volume by `pipeline_name`).
    """
    try:
        import yaml as _yaml
    except ImportError:
        print(
            "ERROR: pyyaml is not installed. Install it with:\n"
            "  pip install pyyaml\n"
            "(or use `requirements.txt` which already lists it)."
        )
        return

    cfg_path = Path(config_path)
    if not cfg_path.exists():
        print(f"ERROR: config file not found: {cfg_path}")
        return

    cfg = _yaml.safe_load(cfg_path.read_text())
    if not isinstance(cfg, dict):
        print(f"ERROR: config did not parse as a mapping (got {type(cfg).__name__})")
        return

    # Model ids default to configs/models.yaml when the pipeline config omits them.
    from IMPersona2.model_config import model_for, template_for
    if isinstance(cfg.get("impersona"), dict):
        cfg["impersona"].setdefault("model", model_for("user_model", "base"))
        if not cfg["impersona"].get("template"):
            cfg["impersona"]["template"] = template_for("user_model", cfg["impersona"]["model"])
    if isinstance(cfg.get("partner"), dict):
        cfg["partner"].setdefault("model", model_for("assistant", "base"))
    if isinstance(cfg.get("eval"), dict):
        if isinstance(cfg.get("partner"), dict) and not cfg["eval"].get("partner_template"):
            cfg["eval"]["partner_template"] = template_for("assistant", cfg["partner"]["model"])
        cfg["eval"].setdefault("partner_thinking", bool(model_for("assistant", "thinking")))

    # Schema check — friendly errors before we spawn anything in the cloud
    required = {
        "pipeline_name": str,
        "num_rounds": int,
        "impersona": dict,
        "partner": dict,
        "eval": dict,
        "training": dict,
    }
    for key, typ in required.items():
        if key not in cfg:
            print(f"ERROR: config missing required field '{key}'")
            return
        if not isinstance(cfg[key], typ):
            print(
                f"ERROR: config field '{key}' must be {typ.__name__}, "
                f"got {type(cfg[key]).__name__}"
            )
            return
    for sect, keys in {
        "impersona": ("model", "adapter"),
        "partner": ("model",),
        "eval": ("num_trials",),
        "training": ("lr", "lora_rank", "lora_alpha"),
    }.items():
        for k in keys:
            if k not in cfg[sect]:
                print(f"ERROR: config['{sect}'] missing required field '{k}'")
                return

    cfg.setdefault("persona_name", _local_persona_name())

    app_name = MODAL_APP_NAME
    print(f"╔═══ Submitting iterative DPO pipeline ═══")
    print(f"║ pipeline_name: {cfg['pipeline_name']}")
    print(f"║ num_rounds:    {cfg['num_rounds']}")
    print(f"║ persona_name:  {cfg['persona_name']}")
    print(f"║ partner:       {cfg['partner']['model']} "
          f"(base adapter: '{cfg['partner'].get('base_adapter', '')}')")
    print(f"║ impersona:     {cfg['impersona']['model']} + {cfg['impersona']['adapter']}")
    print(f"║ eval:          {cfg['eval']['num_trials']} trial(s)/scenario, "
          f"max_concurrent={cfg['eval'].get('max_concurrent', 50)}")
    print(f"║ training:      LR {cfg['training']['lr']}, rank {cfg['training']['lora_rank']}, "
          f"β {cfg['training'].get('pref_beta', 0.1)}, "
          f"{cfg['training'].get('epochs', 2)} epoch(s)")
    print(f"║ Modal app:     {app_name}")

    try:
        deployed_fn = modal.Function.from_name(app_name, "run_pipeline")
    except modal.exception.NotFoundError:
        print(f"\nERROR: '{app_name}' / `run_pipeline` not deployed. Run:")
        print(f"  modal deploy modal_app.py")
        return

    fc = deployed_fn.spawn(config_json=json.dumps(cfg))
    print(f"║ Function call: {fc.object_id}")
    print(f"╚═══════════════════════════════════════════")
    print(f"\nPipeline launched. You can close your laptop.")
    print(f"State (resumable):  /data/pipelines/{cfg['pipeline_name']}/state.json")
    print(f"Final adapter:      /data/{cfg['pipeline_name']}-r{cfg['num_rounds']}-adapter")
    print(f"\nMonitor:")
    print(f"  modal app logs {app_name}")
    print(f"  # or re-run this command to print the resume status\n")
