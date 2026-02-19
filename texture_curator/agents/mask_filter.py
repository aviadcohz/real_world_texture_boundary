"""
Mask Filter Agent for Texture Curator.

Uses a Vision-Language Model (VLM) to assess each crop+mask pair and determine
whether the mask is a good quality texture boundary mask.

The VLM sees a side-by-side composite of the crop image and its mask, then
classifies the mask quality with a reason category.

BAD MASK CATEGORIES:
- NO_BOUNDARY: mask has no meaningful boundary lines
- NOT_GROUND_TRUTH: mask doesn't match real texture transitions in image
- INCOMPLETE: mask is fragmented, dotted, broken lines
- TOO_COMPLEX: too many branches, noisy, overly complicated
- NO_TEXTURE_TRANSITION: original image has no clear texture transitions

GOOD:
- PASS: mask is a good quality texture boundary mask
"""

import logging
import tempfile
import time
from pathlib import Path
from typing import List, Optional

from PIL import Image, ImageDraw, ImageFont

try:
    from agents.base import BaseAgent, ToolResult
    from config.settings import MaskStatus, Phase
    from llm.ollama_client import OllamaClient
    from state.graph_state import GraphState
    from state.models import MaskFilterVerdict
except ImportError:
    from .base import BaseAgent, ToolResult
    from ..config.settings import MaskStatus, Phase
    from ..llm.ollama_client import OllamaClient
    from ..state.graph_state import GraphState
    from ..state.models import MaskFilterVerdict

logger = logging.getLogger(__name__)


# ============================================
# VLM Prompt
# ============================================

MASK_FILTER_SYSTEM_PROMPT = """You assess crop+mask pairs for a texture boundary dataset.

You see two panels: LEFT = crop image, RIGHT = boundary mask (white lines on black).

Boundary masks mark where textures change (brick/stone, grass/pavement, wood/fabric). White lines in the mask trace these transitions. The lines are thin - this is normal.

Respond with JSON only."""

MASK_FILTER_PROMPT = """LEFT = crop image, RIGHT = boundary mask.

Answer these two questions, then give your verdict:

Q1: Does the RIGHT panel (mask) have visible white boundary lines?
Q2: Does the LEFT panel (image) contain at least two visually different surfaces, textures, or materials?

Q2 guidance - answer YES if you see ANY of these, even partially:
- Different materials meeting: grass/stone, brick/sky, wall/railing, tree/building, fence/facade, pavement/vegetation
- A surface changing: old brick vs new brick, shadow creating visible texture change
- Natural vs man-made elements together: trees near walls, grass near ruins
- ANY second surface visible, even in a small corner of the image
Answer NO for Q2 ONLY if the image is completely filled by one single uniform surface with absolutely nothing else visible (e.g. pure blue sky, open water, a blank wall filling the entire frame).
When in doubt about Q2, answer YES.

Rules:
- If Q1=NO (mask is blank/empty): verdict is "fail", reason "NO_BOUNDARY"
- If Q2=NO (entire image is one single uniform surface): verdict is "fail", reason "NO_TEXTURE_TRANSITION"
- If BOTH Q1=YES and Q2=YES: verdict is "pass", reason "PASS"

{"verdict": "pass" or "fail", "reason": "NO_BOUNDARY" or "NO_TEXTURE_TRANSITION" or "PASS", "confidence": 0.0-1.0, "explanation": "Q1:[yes/no] Q2:[yes/no] - brief reason"}"""


# ============================================
# Mask Filter Agent
# ============================================

class MaskFilterAgent(BaseAgent):
    """
    VLM-based mask quality filter agent.

    Assesses each crop+mask pair using a Vision-Language Model to determine
    whether the mask is a good quality texture boundary mask.
    """

    def __init__(
        self,
        llm_client: OllamaClient,
        vlm_model: str = "qwen2.5vl:7b",
        device: str = "cuda",
    ):
        super().__init__(
            name="mask_filter",
            llm_client=llm_client,
            system_prompt=MASK_FILTER_SYSTEM_PROMPT,
        )
        self.vlm_model = vlm_model
        self.device = device
        self._vlm_client = None

    @property
    def vlm_client(self) -> OllamaClient:
        """Lazy-loaded VLM client (separate from the text LLM)."""
        if self._vlm_client is None:
            self._vlm_client = OllamaClient(
                model=self.vlm_model,
                temperature=0.1,  # Low temperature for consistent classification
                max_tokens=256,
                timeout=120,
            )
        return self._vlm_client

    def get_available_tools(self) -> List[str]:
        return [
            "filter_masks",
            "done",
        ]

    def format_state_for_prompt(self, state: GraphState) -> str:
        return (
            f"Candidates discovered: {state.num_candidates}\n"
            f"Mask filter: {state.num_mask_filtered} assessed, "
            f"{state.num_mask_passed} passed"
        )

    # ============================================
    # Side-by-side Composite
    # ============================================

    @staticmethod
    def create_composite(
        image_path: Path,
        mask_path: Path,
        target_height: int = 384,
    ) -> Path:
        """
        Create a side-by-side composite: crop LEFT, mask RIGHT.

        The mask boundary lines are dilated to make them clearly visible to the VLM.
        Returns path to a temporary file.
        """
        import cv2
        import numpy as np

        crop = Image.open(image_path).convert("RGB")
        mask_gray = Image.open(mask_path).convert("L")

        # Dilate mask lines to make them clearly visible
        mask_arr = np.array(mask_gray)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_arr = cv2.dilate(mask_arr, kernel, iterations=2)
        mask = Image.fromarray(mask_arr).convert("RGB")

        # Resize both to the same height, preserving aspect ratio
        crop_w = int(crop.width * target_height / crop.height)
        mask_w = int(mask.width * target_height / mask.height)
        crop = crop.resize((crop_w, target_height), Image.LANCZOS)
        mask = mask.resize((mask_w, target_height), Image.LANCZOS)

        # Create composite with separator
        sep = 4
        label_h = 22
        total_w = crop_w + sep + mask_w
        total_h = target_height + label_h
        composite = Image.new("RGB", (total_w, total_h), (30, 30, 30))

        # Add labels
        draw = ImageDraw.Draw(composite)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except (OSError, IOError):
            font = ImageFont.load_default()
        draw.text((crop_w // 2 - 20, 3), "CROP", fill=(200, 200, 200), font=font)
        draw.text((crop_w + sep + mask_w // 2 - 20, 3), "MASK", fill=(200, 200, 200), font=font)

        # Paste images
        composite.paste(crop, (0, label_h))
        composite.paste(mask, (crop_w + sep, label_h))

        # Save to temp file
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        composite.save(tmp.name, quality=90)
        return Path(tmp.name)

    # ============================================
    # VLM Assessment
    # ============================================

    def assess_single(self, image_path: Path, mask_path: Path) -> MaskFilterVerdict:
        """Assess a single crop+mask pair using VLM."""
        composite_path = None
        try:
            composite_path = self.create_composite(
                image_path, mask_path,
            )

            result = self.vlm_client.chat_vision_json(
                prompt=MASK_FILTER_PROMPT,
                image_paths=[composite_path],
                system_prompt=MASK_FILTER_SYSTEM_PROMPT,
            )

            verdict_str = result.get("verdict", "fail").lower().strip()
            passed = verdict_str == "pass"
            reason = result.get("reason", "UNKNOWN").upper().strip()
            confidence = float(result.get("confidence", 0.5))
            explanation = result.get("explanation", "")

            return MaskFilterVerdict(
                passed=passed,
                reason=reason,
                confidence=min(max(confidence, 0.0), 1.0),
                explanation=explanation,
            )

        except Exception as e:
            logger.warning(f"VLM assessment failed for {image_path.name}: {e}")
            # On failure, don't reject - return uncertain verdict
            return MaskFilterVerdict(
                passed=True,
                reason="VLM_ERROR",
                confidence=0.0,
                explanation=f"VLM assessment failed: {e}",
            )
        finally:
            # Clean up temp file
            if composite_path and composite_path.exists():
                composite_path.unlink(missing_ok=True)

    # ============================================
    # Math Pre-filter (optional fast first pass)
    # ============================================

    @staticmethod
    def math_prefilter(mask_path: Path) -> Optional[str]:
        """
        Fast mathematical pre-filter for obvious failures.

        Returns rejection reason string if mask clearly fails, None if it passes.
        Only catches the most obvious cases to save VLM calls.
        """
        import cv2
        import numpy as np

        img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return "unreadable"

        h, w = img.shape
        total = h * w
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        white = int(np.count_nonzero(binary))
        white_ratio = white / total

        # Almost no boundary pixels
        if white_ratio < 0.003:
            return "no_boundary_pixels"

        # Region analysis
        inverted = cv2.bitwise_not(binary)
        n_labels, labels = cv2.connectedComponents(inverted)
        total_regions = n_labels - 1  # exclude background

        # Too fragmented (many tiny regions = noisy mask)
        if total_regions > 30:
            return "too_fragmented"

        # Boundary component analysis
        n_bcc, _ = cv2.connectedComponents(binary)
        n_bcc -= 1
        if n_bcc > 25:
            return "too_many_boundary_components"

        return None  # Passes pre-filter

    # ============================================
    # Full Pipeline
    # ============================================

    def run_full_filtering(self, state: GraphState) -> ToolResult:
        """
        Run mask quality filtering on all discovered candidates.

        1. Optionally pre-filter with fast math checks
        2. Send remaining to VLM for visual assessment
        3. Update candidate mask_status and verdict
        4. Mark mask_filtering_done on state
        """
        logger.info("=" * 60)
        logger.info("MASK FILTER: Starting VLM-based mask quality assessment")
        logger.info("=" * 60)

        pending = [
            c for c in state.candidates.values()
            if c.mask_status == MaskStatus.PENDING
        ]
        logger.info(f"Candidates to assess: {len(pending)}")

        if not pending:
            state.mask_filtering_done = True
            return ToolResult(
                success=True,
                data={"assessed": 0, "passed": 0, "failed": 0, "message": "No pending candidates"},
            )

        enable_prefilter = state.config.mask_filter.enable_prefilter
        skip_vlm = state.config.mask_filter.skip_vlm

        start = time.time()
        stats = {
            "prefilter_rejected": 0,
            "vlm_passed": 0,
            "vlm_failed": 0,
            "vlm_errors": 0,
            "reasons": {},
        }

        for i, candidate in enumerate(pending):
            # --- Math pre-filter ---
            if enable_prefilter:
                reject_reason = self.math_prefilter(candidate.mask_path)
                if reject_reason:
                    candidate.mask_status = MaskStatus.REJECTED
                    candidate.mask_filter_verdict = MaskFilterVerdict(
                        passed=False,
                        reason=f"PREFILTER_{reject_reason.upper()}",
                        confidence=1.0,
                        explanation=f"Failed fast pre-filter: {reject_reason}",
                    )
                    stats["prefilter_rejected"] += 1
                    reason_key = f"prefilter_{reject_reason}"
                    stats["reasons"][reason_key] = stats["reasons"].get(reason_key, 0) + 1
                    continue

            # --- VLM assessment ---
            if not skip_vlm:
                verdict = self.assess_single(candidate.image_path, candidate.mask_path)
                candidate.mask_filter_verdict = verdict

                if verdict.passed:
                    candidate.mask_status = MaskStatus.VALID
                    stats["vlm_passed"] += 1
                else:
                    candidate.mask_status = MaskStatus.REJECTED
                    stats["vlm_failed"] += 1
                    reason_key = verdict.reason
                    stats["reasons"][reason_key] = stats["reasons"].get(reason_key, 0) + 1

                if verdict.reason == "VLM_ERROR":
                    stats["vlm_errors"] += 1
            else:
                # VLM skipped - mark as valid (only prefilter used)
                candidate.mask_status = MaskStatus.VALID
                stats["vlm_passed"] += 1

            # Progress logging
            total_done = stats["prefilter_rejected"] + stats["vlm_passed"] + stats["vlm_failed"]
            if total_done % 50 == 0:
                elapsed = time.time() - start
                rate = total_done / elapsed if elapsed > 0 else 0
                logger.info(
                    f"  Progress: {total_done}/{len(pending)} "
                    f"({rate:.1f}/s) - passed={stats['vlm_passed']}, "
                    f"failed={stats['prefilter_rejected'] + stats['vlm_failed']}"
                )

        elapsed = time.time() - start
        total_passed = stats["vlm_passed"]
        total_failed = stats["prefilter_rejected"] + stats["vlm_failed"]

        state.mask_filtering_done = True
        state.current_phase = Phase.MASK_FILTERING

        logger.info(f"\nMask Filter Complete ({elapsed:.1f}s)")
        logger.info(f"  Assessed: {len(pending)}")
        logger.info(f"  Passed: {total_passed}")
        logger.info(f"  Rejected: {total_failed} (prefilter={stats['prefilter_rejected']}, vlm={stats['vlm_failed']})")
        if stats["reasons"]:
            logger.info(f"  Rejection reasons:")
            for reason, count in sorted(stats["reasons"].items(), key=lambda x: -x[1]):
                logger.info(f"    {reason}: {count}")

        return ToolResult(
            success=True,
            data={
                "assessed": len(pending),
                "passed": total_passed,
                "failed": total_failed,
                "prefilter_rejected": stats["prefilter_rejected"],
                "vlm_failed": stats["vlm_failed"],
                "vlm_errors": stats["vlm_errors"],
                "elapsed_seconds": round(elapsed, 1),
                "reasons": stats["reasons"],
            },
        )
