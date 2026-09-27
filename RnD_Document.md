# Change-Agent: R&D Technical Document

> **Paper:** [Change-Agent: Toward Interactive Comprehensive Remote Sensing Change Interpretation and Analysis](https://ieeexplore.ieee.org/document/10591792)  
> **Published:** IEEE TGRS 2024  
> **Authors:** Chenyang Liu, Keyan Chen, Haotian Zhang, Zipeng Qi, Zhengxia Zou, Zhenwei Shi

---

## Table of Contents
1. [What It Does](#1-what-it-does)
2. [System Architecture](#2-system-architecture)
3. [MCI Model — Deep Dive](#3-mci-model--deep-dive)
4. [Input & Output Specification](#4-input--output-specification)
5. [Datasets](#5-datasets)
6. [Models & Download Links](#6-models--download-links)
7. [Supported Backbone Encoders](#7-supported-backbone-encoders)
8. [Key Hyperparameters You Can Tweak](#8-key-hyperparameters-you-can-tweak)
9. [Factors to Improve Results](#9-factors-to-improve-results)
10. [Alternatives to Components Used](#10-alternatives-to-components-used)
11. [Known Limitations & Gotchas](#11-known-limitations--gotchas)
12. [Resource Links](#12-resource-links)

---

## 1. What It Does

Change-Agent is a **multi-task, multi-agent AI system** for analyzing changes between two satellite/aerial images (bi-temporal remote sensing imagery). It does **three things simultaneously**:

| Task | Output | Description |
|---|---|---|
| **Change Detection (CD)** | Segmentation mask (PNG) | Pixel-level map of *what* changed (road vs building) |
| **Change Captioning (CC)** | Natural language text | Sentence describing *how* it changed |
| **Interactive Agent Q&A** | Chat response | LLM + tools answering questions about the change |

**Example use case:** Give it a satellite image from 2020 and 2023 of the same city block → it tells you "new buildings were constructed in the eastern section" AND generates a color mask showing exactly where.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────┐
│                  CHANGE-AGENT SYSTEM                │
│                                                     │
│  ┌──────────────────────────────────────────────┐   │
│  │         MCI MODEL (Vision Core)              │   │
│  │                                              │   │
│  │  Image A ──┐                                 │   │
│  │            ├──► Encoder ──► AttentiveEncoder │   │
│  │  Image B ──┘    (SegFormer)  (Transformer    │   │
│  │                              Neck)           │   │
│  │                      │                       │   │
│  │          ┌───────────┴──────────────┐        │   │
│  │          ▼                          ▼        │   │
│  │   CD Branch                   CC Branch      │   │
│  │  (Segmentation)            (Captioning)      │   │
│  │  Change Mask                  Caption Text   │   │
│  └──────────────────────────────────────────────┘   │
│                           │                         │
│  ┌────────────────────────▼────────────────────┐    │
│  │           LAGENT FRAMEWORK (Agent Layer)    │    │
│  │                                             │    │
│  │   LLM (GPT-3.5 / InternLM)                 │    │
│  │     + ReAct Agent Loop                      │    │
│  │     + Tool: Visual_Change_Process           │    │
│  │     + Tool: GoogleSearch                    │    │
│  │     + Tool: PythonInterpreter               │    │
│  └─────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

### Two-Layer Design

| Layer | Component | Purpose |
|---|---|---|
| **Vision Core** | MCI Model | Low-level change perception (detection + captioning) |
| **Agent Layer** | lagent + LLM | High-level reasoning, tool use, user interaction |

The agent uses the MCI model as a **tool** — it calls it like a function and reasons about the output using an LLM.

---

## 3. MCI Model — Deep Dive

The **Multi-Level Change Interpretation (MCI)** model has 3 main components:

### 3.1 Encoder (`model/model_encoder_att.py` → `Encoder`)

- **Default backbone:** `segformer-mit_b1` (Mix Transformer B1)
- Takes two images (A and B) independently through the same backbone
- SegFormer extracts **4 hierarchical feature maps** at different scales via `stage_123` + `stage_4`
- The 4 feature map dims for mit_b1: `[64, 128, 320, 512]`

### 3.2 Attentive Encoder / Neck (`AttentiveEncoder`)

This is the **most important and novel part** of the architecture. It has **two parallel branches**:

#### Change Detection Branch
```
Features A, B (4 scales)
    │
    ▼
CD_neck_s0: Add learnable positional embeddings
    │
    ▼
Transformer_aug_CD (3 layers of cross-attention)
    │  → feat_A attends to feat_B (cross-temporal)
    │  → feat_B attends to feat_A (cross-temporal)
    ▼
Multi-scale feature fusion:
    dif = conv(feat_B - feat_A) + CosineSimilarity(feat_A, feat_B)
    fused = conv([feat_A, dif, feat_B])
    │
    ▼
FPN-style upsampling decoder → to_seg (3-class segmentation head)
Output: (H, W, 3) — background / road / building
```

#### Change Captioning Branch
```
Last-scale features A, B
    │
    ▼
CC_neck_s0: Add learnable positional embeddings
    │
    ▼
Dynamic_DIF_aware_TR (3 layers of Transformer with Dynamic Conv)
    │  → feat_A cross-attends to feat_B
    │  → feat_B cross-attends to feat_A
    ▼
DecoderTransformer (language decoder)
Output: caption text tokens
```

#### Key Design: Dynamic Difference-Aware Transformer
The custom `Transformer` block uses a `Dynamic_conv` module with **parallel 3×3, 1×5, and 5×1 depthwise convolutions** to capture multi-scale spatial differences before attention. This captures both square and elongated change patterns (roads vs buildings).

### 3.3 Decoder (`model/model_decoder.py` → `DecoderTransformer`)

- Standard Transformer decoder with causal self-attention + cross-attention to image features
- `Conv1`: Projects concatenated [feat_A, feat_B] → `feature_dim (512)`
- `ResBlock` (LN): Residual refinement
- Sinusoidal + learnable positional encoding
- **Greedy decoding** by default (`sample()`), **Beam search** also available (`sample_beam()`)
- Vocabulary size: ~400 words (domain-specific RS vocabulary)

---

## 4. Input & Output Specification

### Inputs

| Parameter | Requirement |
|---|---|
| **Image A** (pre-change) | RGB image, any size (auto-resized to 256×256) |
| **Image B** (post-change) | RGB image, same location as A, different time |
| **Format** | PNG, JPG, JPEG |
| **Channels** | 3-channel RGB (NOT multi-spectral/SAR out of the box) |
| **Normalization** | Applied internally: mean=[99.8, 98.5, 84.1], std=[39.1, 37.3, 34.8] |

### Outputs

| Output | Description |
|---|---|
| **Change mask** | PNG image 256×256, saved to `--mask_save_path` |
| **Caption text** | English sentence describing changes, printed to console |

### Mask Color Coding

| Color | Class | When |
|---|---|---|
| **Black** | No change (background) | `pred == 0` |
| **Cyan** `(0, 255, 255)` | Road change | `pred == 1` |
| **Blue** `(0, 0, 255)` | Building change | `pred == 2` |

---

## 5. Datasets

### Primary: LEVIR-MCI Dataset

| Property | Value |
|---|---|
| **Images** | 256×256 bi-temporal RGB satellite images |
| **Train / Val / Test splits** | 7,025 / 1,063 / 1,063 pairs |
| **Classes** | Background, Road change, Building change |
| **Captions** | 5 captions per image pair |
| **Domain** | Urban areas (buildings, roads) |
| **Source** | Google Earth imagery |

📥 **Download:** https://huggingface.co/datasets/lcybuaa/LEVIR-MCI/tree/main

### Secondary: LEVIR-CC Dataset (Change Captioning only)

- ~10k image pairs, captioning-only (no detection masks)
- **Download:** https://github.com/Chen-Yang-Liu/RSICC

---

## 6. Models & Download Links

| Model | Description | Link |
|---|---|---|
| **MCI_model.pth** | Full pretrained MCI model (detection + captioning) | [Hugging Face](https://huggingface.co/lcybuaa/Change-Agent/tree/main) |
| **SegFormer-mit_b1** | Backbone weights (auto-downloaded by mmseg) | Auto via mmcv |
| **SegFormer-mit_b0~b5** | Alternative backbone sizes | [OpenMMLab](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/segformer) |

**Place pretrained model at:**
```
Multi_change/
  models_ckpt/
    MCI_model.pth
```

---

## 7. Supported Backbone Encoders

| Backbone | Output Dim | Speed | Accuracy | Notes |
|---|---|---|---|---|
| `segformer-mit_b0` | 256 | ⚡⚡⚡ | ★★☆ | Fastest, lowest accuracy |
| **`segformer-mit_b1`** | 512 | ⚡⚡ | ★★★ | **Default — best balance** |
| `segformer-mit_b2` | 512 | ⚡ | ★★★★ | Better accuracy, slower |
| `segformer-mit_b3` | 512 | 🐢 | ★★★★ | High accuracy |
| `segformer-mit_b4` | 512 | 🐢 | ★★★★★ | Very high accuracy |
| `segformer-mit_b5` | 512 | 🐢🐢 | ★★★★★ | Highest accuracy, heaviest |
| `resnet50` | 2048 | ⚡⚡ | ★★★ | Classic CNN alternative |
| `resnet101` | 2048 | ⚡ | ★★★★ | Better CNN baseline |
| `vgg16` | 512 | ⚡ | ★★ | Legacy, not recommended |
| `densenet121` | 1024 | ⚡ | ★★★ | Dense connections |

**To change backbone:**
```bash
python predict.py --network segformer-mit_b2 --encoder_dim 512 ...
```
> Note: if you change backbone, you need a matching checkpoint trained with that backbone.

---

## 8. Key Hyperparameters You Can Tweak

### Inference (`predict.py`)

| Param | Default | Effect |
|---|---|---|
| `--network` | `segformer-mit_b1` | Backbone — bigger = better but slower |
| `--encoder_dim` | `512` | Feature dimension (must match backbone) |
| `--feat_size` | `16` | Spatial size of feature maps (256÷16=16) |
| `--n_heads` | `8` | Attention heads in Transformer |
| `--n_layers` | `3` | Layers in AttentiveEncoder neck |
| `--decoder_n_layers` | `1` | Layers in caption decoder |
| `--max_length` | `41` | Max caption length in words |
| `--dropout` | `0.1` | Dropout (only affects training) |

### Training (`train.py`) — Most Impactful

| Param | Default | Effect |
|---|---|---|
| `--encoder_lr` | `1e-4` | Backbone learning rate |
| `--decoder_lr` | `1e-4` | Decoder learning rate |
| `--train_batchsize` | `16` | Batch size — increase if GPU allows |
| `--num_epochs` | `50` | Training epochs |
| `--train_goal` | `2` | `0`=CD only, `1`=CC only, `2`=both jointly |
| `--train_stage` | `s1/s2` | s1=from scratch, s2=fine-tune from checkpoint |
| `--fine_tune_encoder` | `True` | Whether to fine-tune backbone weights |
| `--n_layers` | `3` | More layers = more expressive but slower |

---

## 9. Factors to Improve Results

### Quick Wins (No Retraining)

**1. Use Beam Search instead of Greedy decoding**
In `predict.py`, change the `sample()` call to `sample_beam()`:
```python
# Change this:
seq = self.decoder.sample(feat1, feat2, k=1)
# To this (beam size k=3~5):
seq = self.decoder.sample_beam(feat1, feat2, k=5)
```
Beam search gives significantly more fluent and accurate captions.

**2. Input image quality**
- Images must be co-registered (aligned to same GPS coordinates/projection)
- Seasonal variations (snow/foliage) confuse the model — use same-season pairs when possible
- Cloud-free images only — clouds are falsely detected as changes
- Pre-process noisy/low-quality satellite imagery with contrast enhancement

**3. Patch-based inference for large images**
- Model auto-resizes to 256×256, losing fine detail on large images
- Crop your image into overlapping 256×256 patches, run inference per patch, then stitch masks together

**4. Post-process the segmentation mask**
```python
import cv2
import numpy as np
# Remove tiny false-positive blobs
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
# Fill small holes in detected regions
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
```
The `compute_object_num()` already filters `area < 5` — increase this threshold to reduce noise.

**5. Run multiple times and ensemble**
The model is slightly non-deterministic. Running 3–5 times and majority-voting the mask pixels can reduce errors.

---

### Retraining for Better Results

**1. Larger backbone**
Switch from `mit_b1` → `mit_b2` or `mit_b3`:
```bash
python train.py --network segformer-mit_b2 --encoder_dim 512 ...
```

**2. More Transformer layers**
```bash
python train.py --n_layers 5 --decoder_n_layers 3 ...
```

**3. Two-stage training (the intended procedure)**
```bash
# Stage 1: Train detection branch first
python train.py --train_goal 0 --train_stage s1 --savepath ./models_ckpt/

# Stage 2: Fine-tune captioning branch
python train.py --train_goal 1 --train_stage s2 --checkpoint ./models_ckpt/best.pth

# Stage 3: Joint fine-tuning of both tasks
python train.py --train_goal 2 --train_stage s2 --checkpoint ./models_ckpt/best.pth
```

**4. Data augmentation (add to dataset loader)**
- Random horizontal/vertical flips — apply identically to A and B
- Random rotation (same angle for both images)
- Color jitter — carefully, not too aggressive or it looks like a real change
- Random cropping — ensures model is robust to sub-image patches

**5. Loss weighting**
In `train.py`, detection loss and captioning loss are summed equally. If you care more about detection accuracy:
```python
total_loss = 2.0 * loss_det + 1.0 * loss_cap  # weight detection more
```

**6. Custom vocabulary**
If deploying in a non-urban domain (forest, agriculture), the vocabulary is too urban-centric. Retrain the decoder with a new `vocab.json` built from your domain captions.

---

## 10. Alternatives to Components Used

### Alternative to SegFormer Backbone

| Alternative | Advantage | Notes |
|---|---|---|
| **Swin Transformer** | Hierarchical, often better on RS | Available in `mmsegmentation` |
| **BEiT / BEiT-v2** | Better pretraining with masked image modeling | Larger memory |
| **ResNet + FPN** | Simpler, well-understood baseline | Lower accuracy on RS data |
| **ConvNeXt** | Modern CNN, efficient | Good accuracy/speed tradeoff |
| **ViT (plain)** | Strong features, but lacks hierarchical structure | Needs adapter for dense prediction |

### Alternative to Change Captioning Decoder

| Alternative | Advantage |
|---|---|
| **BLIP-2 / InstructBLIP** | Much stronger vision-language model, zero-shot capable |
| **LLaVA** | Open-source, can describe any image change conversationally |
| **GPT-4V (Vision)** | Best quality, not open-source |
| **mPLUG-Owl** | Lightweight VLM, fast inference |
| **Qwen-VL** | Strong multilingual VLM |

### Alternative to Change Detection Head

| Alternative | Advantage |
|---|---|
| **ChangeFormer** | Transformer-based, strong urban CD baseline |
| **BIT (Bitemporal Image Transformer)** | Efficient, good for urban RS |
| **SNUNet** | Dense connection network, good for small changes |
| **DSIFN** | Deep supervision + feature interaction |
| **Tiny-CD** | Lightweight, runs well on CPU |

### Alternative to lagent (Agent Framework)

| Alternative | Notes |
|---|---|
| **LangChain** | More popular, larger ecosystem, easier tool integration |
| **AutoGen** (Microsoft) | Multi-agent orchestration, role-based |
| **CrewAI** | Role-based multi-agent, simpler API |
| **LlamaIndex** | Better for document/knowledge retrieval tasks |
| **Semantic Kernel** | Microsoft, good .NET integration |

---

## 11. Known Limitations & Gotchas

| Limitation | Detail |
|---|---|
| **Only 2 semantic classes** | Detects only road/building — no vegetation, water, farmland, etc. |
| **Fixed input size** | Always resized to 256×256 — fine details in large images are lost |
| **Small vocabulary** | ~400 RS-specific words — limited caption variety |
| **GPU required** | CPU inference is very slow (~60s per pair on RTX 2050 equivalent) |
| **Co-registration required** | Images must be from same location/projection — misaligned pairs give random output |
| **Seasonal bias** | Trained on Google Earth imagery — may fail on SAR or multispectral |
| **Greedy decoding default** | Captions are suboptimal without beam search |
| **No uncertainty/confidence output** | Detection gives hard predictions only, no probability map |
| **Urban domain bias** | LEVIR-MCI is all urban — performance degrades on rural/forest/agricultural scenes |
| **Windows path issues** | Backslashes in paths cause escape sequence bugs (e.g., `\v`, `\n`) — use forward slashes |

---

## 12. Resource Links

### Official Resources

| Resource | Link |
|---|---|
| Paper (IEEE) | https://ieeexplore.ieee.org/document/10591792 |
| Pretrained MCI Model | https://huggingface.co/lcybuaa/Change-Agent/tree/main |
| LEVIR-MCI Dataset | https://huggingface.co/datasets/lcybuaa/LEVIR-MCI/tree/main |
| GitHub Repository | https://github.com/Chen-Yang-Liu/Change-Agent |
| LEVIR-CC Dataset | https://github.com/Chen-Yang-Liu/RSICC |
| Survey Paper | https://arxiv.org/abs/2412.02573 |

### Dependencies & Frameworks

| Resource | Link |
|---|---|
| lagent Framework | https://github.com/InternLM/lagent |
| SegFormer (mmseg configs) | https://github.com/open-mmlab/mmsegmentation/tree/main/configs/segformer |
| mmcv pre-built wheels | https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html |
| PyTorch CUDA wheels | https://download.pytorch.org/whl/cu118 |

### Alternative Datasets for Fine-tuning

| Dataset | Description | Link |
|---|---|---|
| **DSIFN-CD** | Large-scale building/road CD | https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection |
| **WHU-CD** | Building change detection | http://gpcv.whu.edu.cn/data/ |
| **SYSU-CD** | 20,000 aerial image pairs, 6 change types | https://github.com/liumency/SYSU-CD |
| **S2Looking** | Rural/agricultural change | https://github.com/AnonymousForACMMM/S2Looking |
| **CDD Dataset** | Seasonal change detection | https://drive.google.com/file/d/1GX656JqqoWabZSd7-N2DL2YEIlBNRXtK |

---

*Document generated for:* `Change-Agent-main`  
*Last updated: 2026-09-27*
