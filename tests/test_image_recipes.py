"""Tests for the built-in image-model recipe matcher."""
from inferhost.core.image_recipes import RECIPES, match_recipe


def test_flux2_klein_matches_and_uses_full_encoder():
    r = match_recipe("Comfy-Org/flux2-klein-4B")
    assert r is not None and r.key == "flux2-klein"
    # The full encoder (not fp4 — fp4 fails to load in sd-server).
    enc = r.companions["text_encoder_path"]
    assert enc == ("Comfy-Org/flux2-klein-4B", "split_files/text_encoders/qwen_3_4b.safetensors")
    assert "vae_path" in r.companions
    assert "--steps" in r.default_args


def test_bonsai_image_maps_to_flux2_klein():
    # The Bonsai image GGUF is a Flux.2-Klein model — it must use that recipe.
    r = match_recipe("Green-Sky/bonsai-image-binary-4B-GGUF")
    assert r is not None and r.key == "flux2-klein"


def test_flux2_matches_before_flux1():
    # A flux2 repo contains the substring 'flux' but must NOT match flux1.
    assert match_recipe("black-forest-labs/FLUX.2-dev").key == "flux2-klein"
    assert match_recipe("city96/FLUX.1-schnell-gguf").key == "flux1"


def test_z_image_and_qwen_image():
    assert match_recipe("leejet/Z-Image-Turbo-GGUF").key == "z-image"
    z = match_recipe("leejet/Z-Image-Turbo-GGUF")
    assert z.companions["text_encoder_path"][0].startswith("unsloth/Qwen3-4B")
    q = match_recipe("QuantStack/Qwen-Image-Edit-GGUF")
    assert q.key == "qwen-image"
    # Qwen-Image-Edit needs the vision encoder (--llm_vision) too.
    assert "vision_encoder_path" in q.companions


def test_match_by_tag_when_repo_name_is_opaque():
    r = match_recipe("someuser/my-cool-model-gguf", tags=["flux2", "text-to-image"])
    assert r is not None and r.key == "flux2-klein"


def test_unknown_model_returns_none():
    assert match_recipe("stabilityai/sdxl-turbo") is None  # single-file, no recipe
    assert match_recipe("Qwen/Qwen2.5-7B-Instruct-GGUF") is None


def test_all_recipes_have_companions_and_defaults():
    for r in RECIPES:
        assert r.companions, f"{r.key} has no companions"
        assert r.default_args, f"{r.key} has no default args"
        for fld, (repo, fname) in r.companions.items():
            assert fld.endswith("_path") and "/" in repo and fname
