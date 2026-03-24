"""Resolve portable paths for checkpoints and Hugging Face caches (repo root + env overrides)."""
import os

_MLD_PACKAGE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_MLD_PACKAGE, "..", ".."))


def resolve_stuxlm_pooler_ckpt(kd_version: str) -> str:
    """PoolerStuXLM weight: STUXLM_CKPT_ROOT, then deps/StuXLM/, then legacy /data/wwj path."""
    env_root = os.environ.get("STUXLM_CKPT_ROOT")
    if env_root:
        return os.path.join(env_root, kd_version, "PoolerStuXLM.pth")
    preferred = os.path.join(REPO_ROOT, "deps", "StuXLM", kd_version, "PoolerStuXLM.pth")
    legacy = os.path.join("/data/wwj/ckpt/StuXLM", kd_version, "PoolerStuXLM.pth")
    if os.path.isfile(preferred):
        return preferred
    if os.path.isfile(legacy):
        return legacy
    return preferred


def resolve_transformers_cache_dir() -> str:
    for key in ("TRANSFORMERS_CACHE", "HF_HOME"):
        v = os.environ.get(key)
        if v:
            return v
    return os.path.join(REPO_ROOT, "deps", "hf")
