# Qwen2SAM Rich Description — Complete Architecture & Training Documentation

*Final version of the Qwen2SAM project for texture boundary segmentation.*

---

## 1. What the Model Does

Qwen2SAM Rich Description is a **two-stage texture boundary segmentation** system. Given a single image containing two adjacent textures (e.g., brick wall next to plaster), it produces pixel-level binary masks for each texture region.

**The problem:** SAM3 (Segment Anything Model 3) can segment objects from text prompts, but struggles with textures — it's very sensitive to exact wording, ~40% of descriptions return no masks, and it often scores correct masks lower than incorrect ones.

**The solution:** Use Qwen2.5-VL (a vision-language model) to understand the image, generate rich multi-word texture descriptions, and inject these as multi-token prompts into SAM3's fusion encoder. Then refine the coarse masks using learned point prompts through SAM3's tracker heads.

**Results (227 test samples, 26 training):**

| Stage | mIoU | Dice | ARI | Good samples (IoU>=0.7) |
|-------|------|------|-----|-------------------------|
| Zero-shot SAM3 | 0.448 | 0.483 | 0.734 | — |
| Stage 1: DETR (100 epochs) | 0.827 | 0.889 | 0.694 | 80.6% |
| **Stage 2: Tracker (200 epochs)** | **0.865** | **0.909** | **0.763** | **81.5%** |

Tracker adds **+4.5% mIoU** and **+10.1% ARI** over DETR alone.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT                                     │
│   Image (1008x1008) + Text prompt ("Identify two textures...")   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │  Qwen2.5-VL │  (2B params, LoRA-adapted)
                    │  with LoRA  │  Trainable: q_proj, v_proj (r=16)
                    └──────┬──────┘
                           │
              Hidden states (B, seq_len, 2048)
                           │
           ┌───────────────┼───────────────┐
           │               │               │
    ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
    │ Description │ │ Description │ │   POINT     │
    │  Tokens A   │ │  Tokens B   │ │  Tokens     │ (Stage 2 only)
    │ (up to 16)  │ │ (up to 16)  │ │  (8 total)  │
    └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
           │               │               │
    ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
    │ Description │ │ Description │ │  CoordHead  │
    │ Projector   │ │ Projector   │ │ 2048→256→2  │
    │2048→1024→256│ │2048→1024→256│ └──────┬──────┘
    └──────┬──────┘ └──────┬──────┘        │
           │               │          Point coords
           │               │          (B, 8, 2)
    ┌──────▼───────────────▼──────┐        │
    │      SAM3 Backbone (frozen) │        │
    │  ViT → FPN (72², 144², 288²)│        │
    └──────┬───────────────┬──────┘        │
           │               │               │
    ┌──────▼──────┐ ┌──────▼──────┐        │
    │ SAM3 Fusion │ │ SAM3 Fusion │        │
    │ Encoder + A │ │ Encoder + B │        │
    │ DETR Decoder│ │ DETR Decoder│        │
    └──────┬──────┘ └──────┬──────┘        │
           │               │               │
    Coarse masks      Coarse masks         │
    (B, 200, 288²)   (B, 200, 288²)       │
           │               │               │
           └───────┬───────┘               │
                   │ (select best,         │
                   │  DETACH)              │
                   │                       │
            ┌──────▼───────────────────────▼──────┐
            │      SAM3 Tracker Heads              │ (Stage 2 only)
            │  PromptEncoder + MaskDecoder         │
            │  Input: coarse mask + 4pos + 4neg    │
            └──────────────┬───────────────────────┘
                           │
                    Refined masks
                    (B, 1, 288x288) per texture
```

---

## 3. Model Components in Detail

### 3.1 Qwen2.5-VL (Vision-Language Model)

- **Model:** `Qwen/Qwen2.5-VL-2B-Instruct` (2B parameters)
- **Adaptation:** LoRA on `q_proj` and `v_proj` (rank=16, alpha=32) — ~3.7M trainable params
- **Gradient checkpointing:** Enabled for memory efficiency
- **Precision:** BFloat16

**Special tokens added:**
- Stage 1: `<START_SEG_A>`, `<END_SEG_A>`, `<START_SEG_B>`, `<END_SEG_B>` (bracket tokens)
- Stage 2: `<POINT_A_1>` through `<POINT_A_4>`, `<POINT_B_1>` through `<POINT_B_4>` (8 point tokens)

**Role:** Qwen sees the image, understands both textures, and produces hidden states that encode:
1. Rich texture descriptions (between bracket tokens)
2. Optimal point locations for each texture (at POINT token positions)

### 3.2 Description Extraction (`extract_description_tokens`)

Scans the token sequence for `<START_SEG_A>` and `<END_SEG_A>`, extracts all hidden states between them (exclusive of markers). Returns:
- `desc_embeds`: `(B, max_len, 2048)` — zero-padded to max_len=16
- `desc_mask`: `(B, max_len)` — True for padding positions (used by SAM3's `prompt_key_padding_mask`)
- `desc_lengths`: `(B,)` — actual token count (typically 4-13 tokens)

**Why multi-token?** Each word of the description gets its own attention slot in SAM3's fusion encoder. "rough diagonal brick" → 3 separate semantic signals vs the old single-token approach which compressed everything into 1 vector.

### 3.3 DescriptionProjector

Per-token MLP that bridges Qwen's embedding space to SAM3's prompt space:
```
Linear(2048 → 1024) → GELU → Linear(1024 → 256)
```
Applied independently to each description token. Padding positions are masked out by SAM3's `prompt_key_padding_mask`.

### 3.4 SAM3 Backbone (Frozen)

- **Input:** 1008x1008 image, normalized with mean/std=[0.5, 0.5, 0.5]
- **ViT encoder:** Produces trunk features
- **FPN neck:** 3 feature levels at 72x72, 144x144, 288x288 (all 256 channels)
- **Completely frozen** — no gradients, eval mode

### 3.5 SAM3 Fusion Encoder + DETR Decoder (Trainable in Stage 1)

- **Fusion encoder:** Cross-attention between image features (5184 tokens from 72x72) and description prompt (N tokens, N<=16). ~9.5M trainable params.
- **DETR decoder:** 200 learned object queries cross-attend to fused image+text features.
- **Outputs per texture:**
  - `pred_masks`: (B, 200, 288x288) — mask logits for each query
  - `pred_logits`: (B, 200, 1) — confidence scores via dot-product scoring against prompt
  - `pred_boxes`: (B, 200, 4) — bounding boxes (cxcywh format)

### 3.6 CoordHead (Stage 2 only)

MLP that regresses 2D point coordinates from POINT token hidden states:
```
Linear(2048 → 256) → ReLU → Linear(256 → 2) → Sigmoid
```
Output in [0,1], scaled to absolute pixels (x1008). Produces 4 positive + 4 negative points per texture.

**Key design:** Texture A's positive points = Texture B's negative points (cross-texture regularization).

### 3.7 SAM3 Tracker Heads (Stage 2 only)

Loaded from SAM3's pretrained tracker checkpoint:
- **PromptEncoder:** Encodes points (sparse) + coarse mask (dense) into embeddings
  - Config: embed_dim=256, image_embedding_size=(72,72), mask_in_chans=16
- **MaskDecoder:** TwoWayTransformer (depth=2, 8 heads, 2048 MLP dim) that refines masks using high-res features
- **Input:** Coarse DETR mask (DETACHED — no gradient to DETR) + 8 point prompts (4 pos + 4 neg)
- **Output:** Refined mask (B, 1, 288x288)

### 3.8 AlignProjector

Linear projection for contrastive alignment:
```
Linear(2048 → 768)
```
Maps mean-pooled description embeddings to SentenceTransformer (all-mpnet-base-v2) space for contrastive loss.

---

## 4. Training: Stage 1 (DETR)

### 4.1 Data Requirements

**Dataset:** RWTD (Real-World Texture Dataset)
- 253 samples total, split: 26 train / 10 val / 227 test
- Each sample: image (JPG), mask_a (PNG), mask_b (PNG), texture_a label, texture_b label
- Labels enriched to 5-10 word descriptions (e.g., "smooth beige woven fabric")

**Metadata format (metadata_phase1.json):**
```json
{
  "image_path": "path/to/image.jpg",
  "mask_a_path": "path/to/mask_a.png",
  "mask_b_path": "path/to/mask_b.png",
  "texture_a": "rough diagonal brick pattern",
  "texture_b": "smooth white plaster surface",
  "coords": [x, y, w, h]
}
```

**Mask format:** PNG grayscale 8-bit (0=background, 255=texture). Binarized at load: `(mask > 127) → float32`.

### 4.2 Batch Construction (V3Collator)

Builds chat-format prompts for Qwen with bracket tokens around GT descriptions:

```
System: "You are a texture boundary segmentation assistant..."
User:   [image] "Identify the two textures in this image..."
Assistant: "The transition is from <START_SEG_A> rough diagonal brick pattern <END_SEG_A>
            to <START_SEG_B> smooth white plaster surface <END_SEG_B>."
```

**At inference:** Uses generic template: "first texture region" / "second texture region" instead of GT labels.

**Collator outputs:**

| Key | Shape | Description |
|-----|-------|-------------|
| `input_ids` | (B, L) | Tokenized chat |
| `attention_mask` | (B, L) | Padding mask |
| `pixel_values` | (B, 3, H, W) | Qwen-normalized image patches |
| `labels` | (B, L) | LM targets (assistant tokens only, rest=-100) |
| `sam_images` | (B, 3, 1008, 1008) | SAM3-normalized image (mean/std=0.5) |
| `masks_a`, `masks_b` | (B, 1008, 1008) | Binary GT masks |
| `gt_boxes_cxcywh` | (B, 4) | Normalized GT boxes |
| `align_target_a/b` | (B, 768) | Pre-cached SentenceTransformer embeddings |

### 4.3 What's Trainable vs Frozen

| Component | Params | Status |
|-----------|--------|--------|
| Qwen LoRA (q_proj, v_proj) | 3.7M | **Trainable** |
| DescriptionProjector | ~2.1M | **Trainable** |
| SAM3 Fusion Encoder | 9.5M | **Trainable** |
| SAM3 Object Queries | ~51K | **Trainable** |
| SAM3 Seg Head | 2.3M | **Trainable** |
| SAM3 Scoring/Box Heads | ~1.3M | **Trainable** |
| AlignProjector | 1.6M | **Trainable** |
| SAM3 ViT Backbone | ~300M | Frozen |
| Qwen base weights | ~2B | Frozen |

### 4.4 Losses

**Total loss = seg_weight x seg + lm_weight x lm + align_weight x align**

Default weights: `seg=1.0, lm=0.5, align=1.0`

#### 4.4.1 Segmentation Loss (DETR-style, per texture)

1. **Hungarian matching:** Match best of 200 queries to 1 GT target. Tries both A/B assignment and swapped — picks whichever gives higher IoU. Prevents the model from learning a consistent label swap.

2. **Mask losses on matched query:**
   - **Focal loss** (weight=5.0): `FL(p_t) = -alpha * (1-p_t)^gamma * log(p_t)` with alpha=0.25, gamma=2.0
   - **Dice loss** (weight=1.0): `1 - (2 * pred∩target + 1) / (sum(pred) + sum(target) + 1)`

3. **Classification focal loss** (weight=2.0): 1 positive query vs 199 negative queries

4. **Box losses on matched query:**
   - **L1** (weight=5.0): `|pred_box - gt_box|`
   - **GIoU** (weight=2.0): Generalized IoU

5. **Exclusivity** (weight=0.5): `mean(sigmoid(mask_a) * sigmoid(mask_b))` — penalizes overlap between texture masks

```
seg = (loss_a + loss_b) / 2 + exclusivity_weight * exclusivity
```

#### 4.4.2 LM Loss (weight=0.5)

Standard cross-entropy on assistant tokens only. Teaches Qwen to generate accurate texture descriptions between bracket tokens. System/user/image tokens are masked (-100).

#### 4.4.3 Alignment Loss (weight=1.0)

Contrastive loss on a 2x2 cosine similarity matrix per sample:
```
sim[i,j] = cos(predicted_embed_i, gt_embed_j) / temperature
```
- `predicted_embed`: mean-pooled description tokens → AlignProjector → (B, 768)
- `gt_embed`: pre-cached SentenceTransformer embeddings of GT texture labels
- Temperature: 0.07
- Memory bank (size=32): stores past embeddings for harder negatives
- Target: diagonal (A→A, B→B)

### 4.5 Training Schedule

| Parameter | Value |
|-----------|-------|
| Epochs | 100 |
| Optimizer | AdamW (weight_decay=0.01) |
| Base LR | 1e-4 |
| Scheduler | Warmup cosine (3 epoch warmup, min_lr=1e-6) |
| Batch size | 1 (gradient accumulation=8, effective batch=8) |
| Max grad norm | 1.0 |
| Precision | Mixed (AMP with BFloat16) |
| Augmentation | Enabled (resize, flip, rotation, color jitter) |

**Seg-grad warmup:** First 10 epochs, segmentation gradients do NOT flow back to Qwen LoRA (description embeddings are detached before projection). After epoch 10, full gradient flow is enabled. This lets LoRA stabilize on LM + alignment loss before adding segmentation gradients.

### 4.6 Checkpoint

`checkpoints/v3/best.pt` — selected by best validation IoU (epoch 19). Contains:
- `projector_state_dict` (DescriptionProjector)
- `sam3_trainable_state_dict` (fusion encoder, queries, seg head, scoring, box)
- `qwen_lora_state_dict` (LoRA weights + special token embeddings)
- `align_projector_state_dict`
- Optimizer, scheduler, scaler states

---

## 5. Training: Stage 2 (Tracker)

### 5.1 Data Requirements

Same dataset as Stage 1. The collator appends POINT tokens to the assistant text:

```
Assistant: "The transition is from <START_SEG_A> rough brick <END_SEG_A> to
            <START_SEG_B> smooth plaster <END_SEG_B>.
            Points: <POINT_A_1> <POINT_A_2> <POINT_A_3> <POINT_A_4>
                    <POINT_B_1> <POINT_B_2> <POINT_B_3> <POINT_B_4>"
```

Additional batch output: `point_positions` (B, 8) — token indices for all POINT tokens.

GT for tracker: same binary masks, downsampled to 288x288 for loss computation.

### 5.2 What's Trainable vs Frozen

| Component | Params | Status |
|-----------|--------|--------|
| Qwen LoRA | 3.7M | **Trainable** (gradients from points → CoordHead → LoRA) |
| DescriptionProjector | ~2.1M | Frozen (loaded from Stage 1) |
| SAM3 DETR (encoder+decoder+heads) | ~13.1M | Frozen (loaded from Stage 1) |
| CoordHead | 525K | **Trainable** (NEW) |
| SAM3 PromptEncoder | 6K | **Trainable** (NEW) |
| SAM3 MaskDecoder | 4.2M | **Trainable** (NEW) |
| AlignProjector | 1.6M | **Trainable** |
| SAM3 ViT Backbone | ~300M | Frozen |
| SAM2 Convs (neck for tracker) | ~200K | Frozen |
| **Total trainable** | **~10M** | |

### 5.3 Forward Pass Flow

1. **Register trunk hook** → capture ViT intermediate features
2. **Qwen forward** (teacher-forced) → hidden states (B, L, 2048)
3. **Extract description tokens** (v3 style) → project to SAM3 space (B, N, 256)
4. **Alignment path:** mean-pool descriptions → AlignProjector → keeps gradient
5. **Extract POINT token hidden states** (NOT detached — gradients flow to LoRA)
6. **CoordHead** → point coordinates (B, 8, 2) in [0,1], scaled to x1008
7. **SAM3 backbone** (frozen) → FPN features
8. **Two DETR passes** (one per texture) → coarse masks (B, 200, 288x288)
9. **Select best coarse mask** per texture by confidence (DETACHED — no gradient to DETR)
10. **SAM2 convs** on cached trunk output → high-res features for tracker
11. **Tracker refinement A:** coarse_mask_a + pos_points_a + neg_points_b → refined_mask_a
12. **Tracker refinement B:** coarse_mask_b + pos_points_b + neg_points_a → refined_mask_b

### 5.4 Losses

**Total = tracker_weight x tracker + align_weight x align**

DETR loss weight = 0.0 (frozen), LM loss weight = 0.0, reconstruction weight = 0.0.

#### Tracker Mask Loss (weight=1.0)

Applied to refined masks (B, 1, 288x288) vs GT masks:
1. **Hungarian matching** (A↔B swap check — same as Stage 1)
2. **Focal loss** (weight=5.0) + **Dice loss** (weight=1.0) per texture
3. **Exclusivity** (weight=0.5): penalize overlap of refined masks

```
tracker = (focal_a + dice_a + focal_b + dice_b) / 2 + exclusivity
```

#### Alignment Loss (weight=1.0)

Same contrastive loss as Stage 1, maintaining text-vision alignment.

### 5.5 Training Schedule

| Parameter | Value |
|-----------|-------|
| Epochs | 200 |
| Optimizer | AdamW (weight_decay=0.01) |
| Base LR | 5e-5 (lower than Stage 1) |
| Scheduler | Warmup cosine (3 epoch warmup) |
| Batch size | 1 (gradient accumulation=8) |
| DETR LR scale | 0.1 (effectively frozen via detr_weight=0.0) |

### 5.6 Checkpoint

`checkpoints/v3_tracker/best.pt` — selected by best validation tracker IoU (epoch 48). Contains everything from Stage 1 plus:
- `coord_head_state_dict`
- `sam_prompt_encoder_state_dict`
- `sam_mask_decoder_state_dict`

---

## 6. Key Design Decisions

1. **Multi-token > single-token:** Each word of the description gets its own attention slot in SAM3's fusion encoder. "rough diagonal brick" → 3 separate semantic signals vs 1 compressed vector.

2. **Coarse masks DETACHED in Stage 2:** Prevents tracker loss from destabilizing the already-trained DETR. The tracker only improves boundaries, doesn't change coarse predictions.

3. **Point coords NOT detached:** Gradients flow from tracker mask loss → CoordHead → POINT tokens → Qwen LoRA. This teaches Qwen to predict points that maximize tracker refinement quality.

4. **Cross-texture points:** Texture A's negative points = Texture B's positive points. This naturally regularizes point placement near the boundary.

5. **No PointProjector:** Ablation on v2_tracker showed that injecting extra sparse tokens from point embeddings into SAM3 actually hurts ARI (-3.9%). Simple coordinate regression works better.

6. **Seg-grad warmup (Stage 1):** First 10 epochs, seg loss doesn't flow to LoRA. Lets Qwen learn stable descriptions via LM+alignment before adding noisy segmentation gradients.

7. **Bracket tokens over numbered tokens:** `<START>...<END>` lets Qwen decide description length based on texture complexity. Natural language tokens carry richer semantics than placeholder special tokens.

---

## 7. How to Evaluate

```bash
cd /home/aviad/real_world_texture_boundary

# Evaluate Stage 1 (DETR only)
python -m qwen2sam.scripts.evaluate_v3 \
  --config qwen2sam/configs/v3.yaml \
  --checkpoint checkpoints/v3/best.pt \
  --split test --output_dir eval_results/v3_test --no_vis

# Evaluate Stage 2 (Tracker) — includes 3-way comparison
python -m qwen2sam.scripts.evaluate_v3_tracker \
  --config qwen2sam/configs/v3_tracker.yaml \
  --checkpoint checkpoints/v3_tracker/best.pt \
  --split test --output_dir eval_results/v3_tracker_test --no_vis
```

## 8. How to Retrain

```bash
# Stage 1: DETR (100 epochs, ~30 min on RTX 5090)
python -u -m qwen2sam.training.train_v3 --config qwen2sam/configs/v3.yaml

# Stage 2: Tracker (200 epochs, ~60 min, requires Stage 1 checkpoint)
python -u -m qwen2sam.training.train_v3_tracker --config qwen2sam/configs/v3_tracker.yaml
```

---

## 9. File Structure

```
qwen2sam/
  models/
    qwen2sam_v3.py            # Stage 1 model (multi-token description + SAM3 DETR)
    qwen2sam_v3_tracker.py    # Stage 2 model (wraps v3 + tracker refinement)
    losses_v3.py              # v3 combined loss (seg + LM + alignment)
    losses_v2_tracker.py      # Tracker mask loss (focal + dice + exclusivity)
    qwen2sam_v2.py            # Base model dependency (SAM3 integration, LoRA)
    qwen2sam_v2_tracker.py    # CoordHead, POINT_TOKENS definitions
    losses.py                 # Shared: alignment loss, focal loss, dice loss
    losses_v2.py              # DETR seg loss with Hungarian matching
    alignment.py              # SentenceTextEmbedder, QwenTextEmbedder
  data/
    dataset_v3.py             # V3Dataset + V3Collator (bracket tokens)
    dataset_v2.py             # Base dataset (image loading, augmentation)
    dataset_phase3.py         # create_labels() for LM loss masking
  training/
    train_v3.py               # Stage 1 training loop
    train_v3_tracker.py       # Stage 2 training loop + V3TrackerCollator
    train_phase1.py           # Shared: load_config, set_seed, plot_training_curves
  scripts/
    evaluate_v3.py            # Stage 1 evaluation
    evaluate_v3_tracker.py    # Stage 2 evaluation (3-way comparison)
    evaluate_v2.py            # Shared: compute_iou, compute_dice, compute_ari
  configs/
    v3.yaml                   # Stage 1 configuration
    v3_tracker.yaml           # Stage 2 configuration
checkpoints/
  v3/best.pt                  # Stage 1 checkpoint (epoch 19, 234MB)
  v3_tracker/best.pt          # Stage 2 checkpoint (epoch 48, 765MB)
eval_results/
  v3_tracker_test/            # Full test set metrics and summary
```

## 10. Dependencies

- **Conda env:** `texture_boundary` (Python 3.11)
- **SAM3:** `/home/aviad/sam3/` (installed via `pip install -e "[dev]"`)
- **Dataset:** `/home/aviad/RWTD/` (253 samples, with `splits.json`)
- **Qwen:** `Qwen/Qwen2.5-VL-2B-Instruct` (HuggingFace, auto-downloaded)
- **GPU:** RTX 5090 (Blackwell, sm_120)
