"""Model / tokenizer loading for prompting, LoRA training and inference."""
from __future__ import annotations

from pathlib import Path

from .utils import get_device, inference_dtype, pick_dtypes, resolve_inference_dtype


def load_for_eval(cfg: dict, adapter_dir: str | Path | None = None):
    """(model, tokenizer) for evaluation, honouring `model.inference_dtype`."""
    device = get_device()
    dtype = resolve_inference_dtype(cfg["model"].get("inference_dtype", "auto"), device)
    return load_inference_model(cfg["model"]["name"], adapter_dir, device, dtype), load_tokenizer(cfg["model"]["name"])


def load_tokenizer(name: str, padding_side: str = "left"):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = padding_side
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def stop_token_ids(tok) -> list[int]:
    ids = {tok.eos_token_id}
    for t in ("<|im_end|>", "<|endoftext|>"):
        tid = tok.convert_tokens_to_ids(t)
        if isinstance(tid, int) and tid != tok.unk_token_id:
            ids.add(tid)
    return sorted(i for i in ids if i is not None)


def _from_pretrained(name: str, dtype):
    from transformers import AutoModelForCausalLM

    try:
        return AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    except TypeError:  # transformers < 4.56
        return AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)


def load_inference_model(name: str, adapter_dir: str | Path | None = None, device=None, dtype=None):
    """Base model (optionally with a LoRA adapter merged in) in eval mode."""
    device = device or get_device()
    dtype = dtype or inference_dtype(device)
    model = _from_pretrained(name, dtype)
    if adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload()
    model.to(device=device, dtype=dtype)
    model.eval()
    model.generation_config.pad_token_id = model.generation_config.pad_token_id or model.config.eos_token_id
    return model


def _ignore_incompatible_torchao() -> None:
    """Recent peft raises (instead of skipping) when an old torchao is installed, as on Colab.

    We never use torchao-quantised layers, so treat an incompatible torchao as absent.
    """
    import sys

    import peft.import_utils as iu

    original = iu.is_torchao_available
    try:
        original()
    except ImportError:
        # patch every peft module that imported the function by name
        for mod in list(sys.modules.values()):
            if getattr(mod, "__name__", "").startswith("peft") and getattr(mod, "is_torchao_available", None) is original:
                mod.is_torchao_available = lambda: False


def load_lora_model(name: str, lora_cfg: dict, device=None, gradient_checkpointing: bool = False):
    from peft import LoraConfig, get_peft_model

    _ignore_incompatible_torchao()
    device = device or get_device()
    weight_dtype, _ = pick_dtypes(device)
    model = _from_pretrained(name, weight_dtype)
    model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=list(lora_cfg["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    # keep trainable LoRA weights in fp32 even when the frozen base is bf16
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.to(device)
    return model


def count_params(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
