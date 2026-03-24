
<h1 align="center"><strong>Bilingual Text-Driven Human Motion Generation</strong></h1>
<p align="center">
 <a href='https://github.com/wengwanjiang' target='_blank'>Wanjiang Weng<sup>*</sup></a>&emsp;
 <a href='https://xiaofeng-tan.github.io/' target='_blank'>Xiaofeng Tan<sup>*</sup></a>&emsp;
 Junbo Wang&emsp;
 Guo-Sen Xie&emsp;
 Pan Zhou&emsp;
 Hongsong Wang<sup>†</sup>&emsp;
  <br>
  *Equal Contribution&emsp;
  †Corresponding Author
</p>

<p align="center">
  <!-- Replace with official links after publication -->
  <a href="./arxiv.pdf"><img src="https://img.shields.io/badge/Paper-PDF-red?style=flat&logo=adobeacrobatreader" alt="Paper PDF"></a>
  <a href=""><img src="https://img.shields.io/badge/Project-Page-gray?style=flat&logo=googlechrome&logoColor=gray" alt="Project Page (TBD)"></a>
  <a href=""><img src="https://img.shields.io/badge/arXiv-TBD-B31B1B?style=flat&logo=arXiv&logoColor=white" alt="arXiv"></a>
  <a href=""><img src="https://img.shields.io/badge/HuggingFace-Models-FFD21E?style=flat&logo=huggingface&logoColor=black" alt="HuggingFace (TBD)"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-See_LICENSE-blue.svg" alt="License"></a>
</p>


This repository is the reference implementation of our paper. For questions about the code or collaboration, please reach out by email wjweng@seu.edu.cn

> We aim to keep the training and evaluation pipeline readable and reproducible. **Public links to pre-trained weights** and the **project page** will be updated in this README and in the badges above once they are ready (currently placeholders).

---

## Table of Contents

- [News](#news)
- [Plan and TODO](#plan-and-todo)
- [Introduction](#introduction)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Data Preparation](#data-preparation)
- [Pre-trained Dependencies and Weights](#pre-trained-dependencies-and-weights)
- [Training](#training)
  - [Stage 0: Motion VAE](#stage-0-motion-vae)
  - [Stage 1: MLD Text-to-Motion (Multilingual)](#stage-1-mld-text-to-motion-multilingual)
- [Evaluation](#evaluation)
- [Demo and Inference](#demo-and-inference)
- [Visualization and SMPL](#visualization-and-smpl)
- [Environment Variables](#environment-variables)
- [FAQ](#faq)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)
- [License](#license)

---

## News

- **[2026/03]** Released training and evaluation code; **English / Chinese / bilingual** HumanML3D text modes are selectable via config.
- **[2026/03]** Camera-ready PDF added at the repository root (see above).

---

## Plan and TODO

> Maintenance checklist aligned with common academic open-source practice; mark items `[x]` when done.

- [x] Release core training code (VAE + MLD + DDIM)
- [x] Configurable `LANG_LIST` (`configs/*.yaml` → `DATASET.HUMANML3D.LANG_LIST`)
- [ ] **Upload pre-trained weights** to Hugging Face / mirrors and update direct links and `huggingface-cli` examples in this README
- [ ] Launch **project page** with demo videos / GIFs
- [ ] Publish **arXiv** or the official proceedings page (and update BibTeX)
- [ ] Optional **Chinese README** (`README_zh.md`) alongside this English document
- [ ] One-shot `prepare_deps.sh` for GloVe, T5, t2m, SMPL, StuXLM, etc.
- [ ] Docker / conda-lock for reproducible environments
- [ ] Unit tests and CI (lint + small smoke training)

---

## Introduction

This project studies **human motion generation under English and Chinese text conditions**. Building on the HumanML3D motion representation and evaluation protocol, we use a **motion VAE** to encode actions into a latent space and train a **latent diffusion model (MLD)** for text-driven generation. The text branch uses an **XLM-R–style student encoder (StuXLM)** (see `configs/modules/text_encoder.yaml`).

### Key Features

- **Multilingual training switch**: set `DATASET.HUMANML3D.LANG_LIST` to `['en']`, `['zh']`, or `['en', 'zh']` without code changes.
- **HumanML3D-aligned data pipeline**: `texts/` provides English captions and tokens; `texts_zh/` provides aligned Chinese captions (see [Data Preparation](#data-preparation)).
- **Standard t2m evaluation**: uses official matching-model weights under `deps/t2m/`, consistent with MLD / HumanML3D-style metrics.

---

## Repository Structure

```text
.
├── configs/                  # Main configs and modules/*.yaml
├── mld/                      # Data, models, metrics, utilities
├── datasets/humanml3d/       # HumanML3D (prepare locally, see below)
├── deps/                     # GloVe, Sentence-T5, t2m, SMPL, HF cache, StuXLM, etc.
├── experiments_recons/       # VAE training outputs (default)
├── experiments_t2m/          # MLD training outputs (default)
├── experiments_*_test/       # Evaluation / demo outputs (default)
├── private/tsne.py           # Feature extraction and t-SNE visualization
├── train_vae.py / train_mld.py / test.py / demo.py
├── fit.py / render.py        # SMPL fitting and Blender rendering
├── requirements.txt
└── README.md

```

---

## Setup

### Conda and Python

```bash
conda create -n bilingual-motion python=3.10.12 -y
conda activate bilingual-motion
pip install -r requirements.txt
```

We primarily use **Python 3.10** and **PyTorch 1.13.x** (CUDA build as noted in `requirements.txt`).

### System Dependencies

- **ffmpeg**: skeleton video export when `demo.py` calls `plot_3d_motion`
- **git-lfs** (optional): large file checkpoints

### Experiment Logging (SwanLab)

Similar to common practice in related projects, we support [SwanLab](https://swanlab.cn):

```bash
swanlab login   # optional; training still runs if not logged in
```

The default SwanLab project name is `bilingual-motion` (see `train_mld.py` / `train_vae.py`).

---

## Data Preparation

### 1. HumanML3D Motions and English Text

Follow the official [HumanML3D](https://github.com/EricGuo5513/HumanML3D) pipeline to obtain `new_joint_vecs`, `texts`, etc., and copy the dataset tree to:

```text
datasets/humanml3d/
├── new_joint_vecs/
├── texts/              # Official English lines (caption#tokens#...)
├── train.txt / val.txt / test.txt
├── Mean.npy / Std.npy
└── ...
```

Keep `DATASET.HUMANML3D.ROOT` in `configs/*.yaml` consistent with this path (default `./datasets/humanml3d`).

### 2. Parallel Chinese Captions `texts_zh/`

When `LANG_LIST` contains **`zh`**, Chinese captions are read from:

```text
datasets/humanml3d/texts_zh/<same basename as texts/*.txt>
```

**Convention**: line *i* in `texts_zh/<id>.txt` aligns with line *i* in `texts/<id>.txt` (same sub-segment / caption slot). English **tokens** for the word vectorizer still come from the corresponding line in `texts/`, preserving compatibility with the HumanML3D tokenization protocol and GloVe.

### 3. Spatial Normalization (Optional)

If you use `humanml_spatial_norm` or similar, keep paths aligned with `MEAN_STD_PATH` in the config (default `./datasets/humanml_spatial_norm`).

---

## Pre-trained Dependencies and Weights

Download or place the following under the indicated relative paths. **Replace placeholder download URLs** with your Hugging Face collection or mirror once weights are published (e.g. a dedicated repo `YOUR_ORG/bilingual-motion-assets`).

### Table 1: Shared Dependencies (Training & Evaluation)

| Asset | Local path | Notes |
|--------|------------|--------|
| GloVe | `deps/glove/` | Word vectors; see `WORD_VERTILIZER_PATH` |
| Sentence-T5-Large | `deps/sentence-t5-large/` | `model.t5_path` in config (if used by your setup) |
| T2M evaluators | `deps/t2m/t2m/...` | Official HumanML3D text–motion matching weights |
| SMPL (optional) | `deps/smpl/` | Used by `fit.py` / rendering |
| Hugging Face cache | `deps/hf/` | Default if `TRANSFORMERS_CACHE` is unset |

**Example download (placeholder — fill in repo and paths after upload)**:

```bash
# huggingface-cli download <YOUR_ORG>/<YOUR_REPO> --local-dir deps --include "glove/*" "t2m/*" ...
```

You can also mirror the dependency layout from the [MotionLCM](https://github.com/Dai-Wenxun/MotionLCM) preparation notes (e.g. Google Drive) into `deps/`.

### Table 2: StuXLM Distillation Weights (Current Text Encoder)

| File | Path | Notes |
|------|------|--------|
| PoolerStuXLM | `deps/StuXLM/<kd_version>/PoolerStuXLM.pth` | Default `kd_version` in `configs/mld_t2m.yaml` → `model.kd_version` |

Alternatively set `STUXLM_CKPT_ROOT` to the root that contains `<kd_version>/PoolerStuXLM.pth`.

### Table 3: Generation Checkpoints from This Repo (Placeholders)

| Stage | Example filename | Location | Link |
|--------|------------------|----------|------|
| Motion VAE | `vae_humanml.ckpt` | `experiments_recons/vae_humanml/` | **TBD** |
| MLD (English) | `*.ckpt` | `experiments_t2m/<run_name>/checkpoints/` | **TBD** |
| MLD (Chinese / bilingual) | `*.ckpt` | same as above | **TBD** |

Point `TRAIN.PRETRAINED` in `configs/mld_t2m.yaml` to your VAE; set `TEST.CHECKPOINTS` to the MLD you want to evaluate.

---

## Training

### Stage 0: Motion VAE

```bash
python -m train_vae --cfg configs/vae.yaml
```

Outputs default to `experiments_recons/<NAME>/<timestamp>/` with checkpoints under `checkpoints/`.

### Stage 1: MLD Text-to-Motion (Multilingual)

English-only, Chinese-only, or bilingual training is controlled by `LANG_LIST` (see below).

Main knob: `configs/mld_t2m.yaml` → `DATASET.HUMANML3D.LANG_LIST`.

| Goal | `LANG_LIST` | Data requirements |
|------|-------------|-------------------|
| **English only** | `['en']` | Standard `texts/` |
| **Chinese only** | `['zh']` | Full `texts_zh/` aligned line-by-line with `texts/` |
| **Bilingual** | `['en', 'zh']` | Same as above; samples are prefixed `en_*` / `zh_*` |

**Chinese training example**

```yaml
# snippet in configs/mld_t2m.yaml
DATASET:
  HUMANML3D:
    LANG_LIST: ['zh']
```

**Bilingual training example**

```yaml
DATASET:
  HUMANML3D:
    LANG_LIST: ['en', 'zh']
```

Then run:

```bash
python -m train_mld --cfg configs/mld_t2m.yaml
```

**Tip**: use different `NAME` values or separate config files (e.g. `configs/mld_t2m_zh.yaml`) per language setting to avoid overwriting runs.

**Notes**

1. `TRAIN.PRETRAINED` must point to a trained VAE from Stage 0.
2. Text encoder, learning rate, batch size, etc. are controlled in `configs/mld_t2m.yaml` and `configs/modules/*.yaml`.
3. At inference, use prompts in the **same language regime** as training (e.g. Chinese prompts for Chinese-only training).

---

## Evaluation

```bash
python -m test --cfg configs/mld_t2m.yaml
```

Ensure `TEST.CHECKPOINTS` points to the MLD checkpoint. Replication and related settings are controlled by `TEST.REPLICATION_TIMES` and neighboring fields.

---

## Demo and Inference

```bash
# Use assets/example.txt (first column: length, remainder: text)
python demo.py --cfg configs/mld_t2m.yaml --example assets/example.txt

# Use HumanML3D test-split captions
python demo.py --cfg configs/mld_t2m.yaml
```

Outputs are written under `TEST_FOLDER/NAME/demo_<timestamp>/`.

---

## Visualization and SMPL

- **Skeleton videos**: `demo.py` writes `.mp4` by default (requires ffmpeg).
- **SMPL meshes**: `python fit.py --pkl <sample.pkl>`, then Blender + `render.py` (see script arguments).
- **Helpers**: `mesh.sh`, `seq.sh` (set env vars as noted in the scripts).

**t-SNE (optional)**: install `open_clip_torch` and `scikit-learn`, then:

```bash
python private/tsne.py
```

---

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `STUXLM_CKPT_ROOT` | Root directory containing `<kd_version>/PoolerStuXLM.pth` |
| `TRANSFORMERS_CACHE` / `HF_HOME` | Hugging Face cache; defaults to `deps/hf/` if unset |
| `DATASET_PROMPT_PKL` | Optional `prompt.pkl` for commented prompt-replacement experiments in the dataset code |

---

## FAQ

1. **File-not-found during Chinese training**  
   Verify `datasets/humanml3d/texts_zh/<id>.txt` exists and line counts match `texts/<id>.txt`.

2. **Missing `PoolerStuXLM.pth`**  
   Place it under `deps/StuXLM/...` or set `STUXLM_CKPT_ROOT` as above.

3. **Unexpected evaluation metrics**  
   Check that `deps/t2m/` and HumanML3D `mean.npy` / `std.npy` paths match the logic in `get_mean_std`.

---

## Citation


```bibtex
@inproceedings{weng2026bilingualmotion,
  title     = {Bilingual Text-to-Motion Generation: A New Benchmark and Baselines},
  author    = {Weng, Wanjiang and Others},
  booktitle = {arXiv},
  year      = {2026},
}
```

---

## Acknowledgement

This codebase is adapted from the open-source [MotionLCM](https://github.com/Dai-Wenxun/MotionLCM) framework. Data and evaluation follow [HumanML3D](https://github.com/EricGuo5513/HumanML3D). We thank the authors for their work.

---

## License

See [LICENSE](LICENSE). Use of this software is subject to its terms (including non-commercial restrictions where applicable) and to the licenses of third-party datasets and libraries.
