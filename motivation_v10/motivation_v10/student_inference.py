"""Qwen3-4B student compressor inference helper (stages 09 + 10).

Loads a PEFT-LoRA adapter on top of the base Qwen3-4B-Instruct-2507
model and generates compressions one (case, decoding-seed) tuple at
a time. Direct HF + PEFT (no vLLM), so we never need to coordinate
GPU memory with a serving daemon.

Memory: bf16 base ~8 GB + LoRA adapter ~50 MB + KV cache for ~12K
prompt + 2K output ≈ ~14-16 GB. Plenty for one 80 GB card.

Concurrency: HF + sdpa + a single A100/H100 is not multi-process
safe; we run sequentially. Throughput is ~5-10 sec per generate.

Typical usage:
    cm = StudentCompressor(adapter_dir="outputs/models/qwen_sft_ck",
                           base_model_id="Qwen/Qwen3-4B-Instruct-2507")
    txt = cm.generate(system_text, user_text, temperature=0.0, seed=42)

Greedy variants pass `temperature=0.0` (and any seed; sampling is
disabled). Stochastic variants pass `temperature=0.7, seed=1000..1007`.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Optional


_THINK_PATTERN = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)


def _strip_think(text: str) -> str:
    return _THINK_PATTERN.sub("", text).strip()


@dataclass
class _LoadedModel:
    tokenizer: object
    model: object


class StudentCompressor:
    """Singleton-style compressor wrapper.

    Loads the (base + adapter) once on first generate(); subsequent
    generate() calls reuse the loaded module. Pass adapter_dir=None
    to use the raw base model (the "Raw-Qwen" variant in stage 09).
    """

    def __init__(
        self,
        *,
        adapter_dir: Optional[str],
        base_model_id: str = "Qwen/Qwen3-4B-Instruct-2507",
        device: str = "cuda",
        bf16: bool = True,
    ):
        self.adapter_dir = adapter_dir
        self.base_model_id = base_model_id
        self.device = device
        self.bf16 = bf16
        self._loaded: Optional[_LoadedModel] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lazy loader
    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> _LoadedModel:
        with self._lock:
            if self._loaded is not None:
                return self._loaded
            import torch
            import transformers
            tok = transformers.AutoTokenizer.from_pretrained(
                self.adapter_dir if self.adapter_dir else self.base_model_id,
                trust_remote_code=True,
            )
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            dtype = torch.bfloat16 if self.bf16 else torch.float32
            base = transformers.AutoModelForCausalLM.from_pretrained(
                self.base_model_id,
                torch_dtype=dtype, trust_remote_code=True,
                attn_implementation="sdpa", device_map=self.device,
            )
            if self.adapter_dir:
                import peft
                model = peft.PeftModel.from_pretrained(base, self.adapter_dir)
            else:
                model = base
            model.eval()
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = True
            self._loaded = _LoadedModel(tokenizer=tok, model=model)
            return self._loaded

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(
        self,
        *,
        system_text: str,
        user_text: str,
        temperature: float = 0.0,
        seed: int = 42,
        max_new_tokens: int = 2048,
        strip_think: bool = True,
    ) -> dict:
        import torch
        m = self._ensure_loaded()
        tok, model = m.tokenizer, m.model

        messages = [
            {"role": "system", "content": system_text},
            {"role": "user",   "content": user_text},
        ]
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tok(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs["input_ids"].shape[1]

        do_sample = temperature > 0.0
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tok.pad_token_id,
        )
        if do_sample:
            gen_kwargs.update(temperature=temperature, top_p=1.0)

        # Seed RNG for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        new_tokens = out[0, prompt_len:]
        raw = tok.decode(new_tokens, skip_special_tokens=True)
        text = _strip_think(raw) if strip_think else raw

        return {
            "text":              text,
            "raw":               raw,
            "prompt_tokens":     int(prompt_len),
            "completion_tokens": int(new_tokens.shape[0]),
        }

    def unload(self) -> None:
        with self._lock:
            if self._loaded is None:
                return
            self._loaded = None
        import torch, gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = ["StudentCompressor"]
