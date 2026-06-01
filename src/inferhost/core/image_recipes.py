"""Built-in recipes for multi-file image models.

Modern diffusion models (Flux.1/.2, Z-Image, Qwen-Image) aren't a single file —
they need a diffusion model plus a VAE and one or more text encoders, usually
spread across different Hugging Face repos, and the *right* file matters (e.g. the
full Qwen encoder, not the fp4 one). Expecting every user to know that is a
non-starter.

A recipe maps a recognizable model family to the exact companion files (from
known-good, non-gated repos) and sane sampling defaults, so adding such a model
becomes "pick the diffusion file → companions fetched automatically". The manual
component picker (Configure screen) remains the fallback for anything unmatched.

All companion files here are non-gated and were verified to load with the bundled
sd-server. Recipes can drift if upstream repos move files — that's why the manual
picker stays as the always-works escape hatch.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ImageRecipe:
    key: str
    label: str
    # Substrings matched (case-insensitive) against the repo id.
    repo_patterns: tuple[str, ...]
    # HF tags matched (case-insensitive) as a secondary signal.
    tag_patterns: tuple[str, ...]
    # Model field -> (repo_id, filename) for each companion to download.
    companions: dict[str, tuple[str, str]] = field(default_factory=dict)
    # Default sd-server sampling flags, applied to the model's extra_args when
    # the user hasn't set their own.
    default_args: str = ""


# Order matters: more specific families first (flux2 before flux1, since
# "flux2" repos also contain the substring "flux").
RECIPES: tuple[ImageRecipe, ...] = (
    ImageRecipe(
        key="flux2-klein",
        label="Flux.2 Klein",
        repo_patterns=("flux2", "flux.2", "flux-2", "klein", "bonsai-image", "bonsai_image"),
        tag_patterns=("flux2",),
        companions={
            "vae_path": ("Comfy-Org/flux2-klein-4B", "split_files/vae/flux2-vae.safetensors"),
            # Quantized GGUF text encoder (~2.5 GB) instead of the fp16
            # safetensors (~8 GB) — cuts total VRAM roughly in half (verified:
            # 9.3 GB -> 4.6 GB on a bonsai q1_0 model) with no visible prompt or
            # quality loss. Non-gated repo (same one Z-Image uses). NOT the fp4
            # variant, which fails to load in sd-server.
            "text_encoder_path": (
                "unsloth/Qwen3-4B-Instruct-2507-GGUF",
                "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
            ),
        },
        default_args="--steps 4 --cfg-scale 1.0",
    ),
    ImageRecipe(
        key="z-image",
        label="Z-Image",
        repo_patterns=("z-image", "z_image", "zimage"),
        tag_patterns=("z-image", "z_image"),
        companions={
            "vae_path": ("second-state/FLUX.1-schnell-GGUF", "ae.safetensors"),
            "text_encoder_path": (
                "unsloth/Qwen3-4B-Instruct-2507-GGUF",
                "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
            ),
        },
        default_args="--steps 8 --cfg-scale 1.0",
    ),
    ImageRecipe(
        key="qwen-image",
        label="Qwen-Image / Qwen-Image-Edit",
        repo_patterns=("qwen-image", "qwen_image", "qwenimage"),
        tag_patterns=("qwen-image", "qwen_image"),
        companions={
            "vae_path": ("QuantStack/Qwen-Image-Edit-GGUF", "VAE/Qwen_Image-VAE.safetensors"),
            "text_encoder_path": (
                "unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
                "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
            ),
            "vision_encoder_path": (
                "QuantStack/Qwen-Image-Edit-GGUF",
                "mmproj/Qwen2.5-VL-7B-Instruct-mmproj-BF16.gguf",
            ),
        },
        default_args="--steps 8 --cfg-scale 2.5",
    ),
    ImageRecipe(
        key="flux1",
        label="Flux.1",
        repo_patterns=("flux1", "flux.1", "flux-1"),
        tag_patterns=(),  # bare "flux" tag is too broad; rely on repo id
        companions={
            "vae_path": ("second-state/FLUX.1-schnell-GGUF", "ae.safetensors"),
            "clip_l_path": ("comfyanonymous/flux_text_encoders", "clip_l.safetensors"),
            "t5xxl_path": ("comfyanonymous/flux_text_encoders", "t5xxl_fp8_e4m3fn.safetensors"),
        },
        default_args="--steps 4 --cfg-scale 1.0",
    ),
)


def match_recipe(repo_id: str, tags: list[str] | None = None) -> ImageRecipe | None:
    """Return the recipe whose family matches ``repo_id`` / ``tags``, else None.

    Checked in RECIPES order (specific first), by repo-id substring then tag.
    """
    rid = (repo_id or "").lower()
    tagset = {t.lower() for t in (tags or [])}
    for recipe in RECIPES:
        if any(p in rid for p in recipe.repo_patterns):
            return recipe
        if tagset and any(t in tagset for t in recipe.tag_patterns):
            return recipe
    return None
