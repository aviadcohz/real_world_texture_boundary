"""Texture Refiner Pipeline — Real-ESRGAN x2plus SR with complementary mask smoothing.

Upscales small image crops (max 2x, max 512px output) using Real-ESRGAN x2plus.
Masks are upscaled with nearest-neighbor and smoothed using signed distance fields
to produce complementary coverage with smooth polynomial-like boundaries.

Usage:
    refiner = TextureRefinerPipeline(device="cuda")
    result = refiner.process_crop(image, mask_a, mask_b)
"""

import cv2
import numpy as np
from PIL import Image


MAX_SCALE = 2
MAX_OUTPUT = 512


class TextureRefinerPipeline:
    """Upscale small texture-boundary crops with Real-ESRGAN x2plus + SDF mask smoothing.

    Args:
        max_scale: Maximum upscale factor from original (default 2).
        max_output: Maximum output dimension in pixels (default 512).
        smooth_sigma: Gaussian sigma for SDF boundary smoothing (higher = smoother).
        device: Torch device string.
        half: Use FP16 inference.
        tile: Tile size for large images (0 = no tiling).
    """

    MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
    NATIVE_SCALE = 2

    def __init__(
        self,
        max_scale: int = MAX_SCALE,
        max_output: int = MAX_OUTPUT,
        smooth_sigma: float = 3.0,
        device: str = "cuda",
        half: bool = True,
        tile: int = 0,
    ):
        self.max_scale = max_scale
        self.max_output = max_output
        self.smooth_sigma = smooth_sigma
        self.device = device
        self._model = None
        self._half = half
        self._tile = tile

    def _load_model(self):
        """Lazy-load Real-ESRGAN x2plus on first use."""
        if self._model is not None:
            return

        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=self.NATIVE_SCALE,
        )
        self._model = RealESRGANer(
            scale=self.NATIVE_SCALE,
            model_path=self.MODEL_URL,
            model=model,
            tile=self._tile,
            tile_pad=10,
            pre_pad=0,
            half=self._half,
            device=self.device,
        )
        print(f"[TextureRefinerPipeline] Loaded RealESRGAN_x2plus")

    def _compute_scale_factor(self, h: int, w: int) -> int:
        """Compute scale factor: 1 (no upscale) or 2 (single 2x pass)."""
        min_dim = min(h, w)
        needed = self.max_output / min_dim
        if min(needed, self.max_scale) <= 1:
            return 1
        return self.NATIVE_SCALE

    def _sr_upscale(self, image_bgr: np.ndarray) -> np.ndarray:
        """Run a single 2x SR pass."""
        self._load_model()
        output, _ = self._model.enhance(image_bgr, outscale=self.NATIVE_SCALE)
        return output

    @staticmethod
    def smooth_masks_complementary(
        mask_a: np.ndarray,
        mask_b: np.ndarray,
        sigma: float = 3.0,
    ) -> tuple:
        """Smooth two binary masks to produce complementary coverage with smooth boundaries.

        Uses signed distance fields: computes distance transform for each mask,
        derives SDF = dist_to_b - dist_to_a, smooths with Gaussian, thresholds at zero.

        Guarantees:
        - Every pixel is assigned to exactly one mask (no gaps, no overlap)
        - The boundary lies at the midpoint of any gap between masks
        - The boundary curve is smooth (Gaussian-smoothed SDF)

        Args:
            mask_a: Binary mask (H, W), uint8 with values in {0, 255}.
            mask_b: Binary mask (H, W), uint8 with values in {0, 255}.
            sigma: Gaussian blur sigma for SDF smoothing. Higher = smoother boundary.

        Returns:
            (smooth_a, smooth_b): Complementary uint8 masks with values in {0, 255}.
        """
        bin_a = (mask_a > 127).astype(np.uint8)
        bin_b = (mask_b > 127).astype(np.uint8)

        # Distance from each pixel to the nearest foreground pixel of each mask
        dist_a = cv2.distanceTransform(1 - bin_a, cv2.DIST_L2, 5).astype(np.float32)
        dist_b = cv2.distanceTransform(1 - bin_b, cv2.DIST_L2, 5).astype(np.float32)

        # SDF: positive = closer to A, negative = closer to B
        sdf = dist_b - dist_a

        # Gaussian smooth the SDF for smooth boundary curves
        if sigma > 0:
            ksize = int(6 * sigma) | 1  # Ensure odd
            ksize = max(3, ksize)
            sdf = cv2.GaussianBlur(sdf, (ksize, ksize), sigma)

        # Threshold: A gets positive side, B gets the rest (complementary)
        smooth_a = (sdf > 0).astype(np.uint8) * 255
        smooth_b = 255 - smooth_a

        return smooth_a, smooth_b

    @staticmethod
    def smooth_mask_contour(
        mask: np.ndarray,
        epsilon_frac: float = 0.005,
        gaussian_ksize: int = 5,
    ) -> np.ndarray:
        """Smooth a single binary mask with polygon approximation + Gaussian blur.

        Retained for backward compatibility with comparison scripts.
        For pipeline use, prefer smooth_masks_complementary().
        """
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            return mask

        smoothed = np.zeros_like(mask)
        for i, cnt in enumerate(contours):
            epsilon = epsilon_frac * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            is_hole = hierarchy[0][i][3] >= 0 if hierarchy is not None else False
            cv2.drawContours(smoothed, [approx], 0, 0 if is_hole else 255, -1)

        if gaussian_ksize > 0:
            blurred = cv2.GaussianBlur(
                smoothed.astype(np.float32),
                (gaussian_ksize, gaussian_ksize), 0,
            )
            smoothed = (blurred > 127).astype(np.uint8) * 255

        return smoothed

    @staticmethod
    def _normalize_mask(mask: np.ndarray) -> np.ndarray:
        """Ensure mask is 2D uint8 with values in {0, 255}."""
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        if mask.dtype == bool:
            return mask.astype(np.uint8) * 255
        mask = mask.astype(np.uint8)
        if mask.max() <= 1:
            mask = mask * 255
        return mask

    def process_crop(
        self,
        image: "np.ndarray | Image.Image",
        mask_a: np.ndarray,
        mask_b: np.ndarray,
    ) -> dict:
        """Upscale an image crop and its two binary masks.

        Returns dict with: image (PIL RGB), mask_a, mask_b (uint8 {0,255}),
        scale_factor, sr_size, input_size.
        """
        # Normalize inputs
        if isinstance(image, Image.Image):
            image_rgb = np.array(image.convert("RGB"))
        elif image.ndim == 3 and image.shape[2] == 3:
            image_rgb = image.copy()
        else:
            raise ValueError(f"Expected RGB image, got shape {image.shape}")

        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        mask_a = self._normalize_mask(mask_a)
        mask_b = self._normalize_mask(mask_b)

        h, w = image_bgr.shape[:2]
        assert mask_a.shape == (h, w), f"mask_a shape {mask_a.shape} != image {(h, w)}"
        assert mask_b.shape == (h, w), f"mask_b shape {mask_b.shape} != image {(h, w)}"

        # SR upscale (single 2x pass or skip)
        scale_factor = self._compute_scale_factor(h, w)
        sr_bgr = self._sr_upscale(image_bgr) if scale_factor > 1 else image_bgr
        sr_h, sr_w = sr_bgr.shape[:2]

        # Clip to max_output if SR overshot
        if sr_h > self.max_output or sr_w > self.max_output:
            clip_h = min(sr_h, self.max_output)
            clip_w = min(sr_w, self.max_output)
            y0 = (sr_h - clip_h) // 2
            x0 = (sr_w - clip_w) // 2
            sr_bgr = sr_bgr[y0:y0 + clip_h, x0:x0 + clip_w]
            sr_h, sr_w = sr_bgr.shape[:2]

        # Resize masks with nearest-neighbor
        mask_a_up = cv2.resize(mask_a, (sr_w, sr_h), interpolation=cv2.INTER_NEAREST)
        mask_b_up = cv2.resize(mask_b, (sr_w, sr_h), interpolation=cv2.INTER_NEAREST)

        # Smooth masks jointly (complementary, gap-free, smooth boundaries)
        mask_a_up, mask_b_up = self.smooth_masks_complementary(
            mask_a_up, mask_b_up, sigma=self.smooth_sigma
        )

        sr_rgb = cv2.cvtColor(sr_bgr, cv2.COLOR_BGR2RGB)
        return {
            "image": Image.fromarray(sr_rgb),
            "mask_a": mask_a_up,
            "mask_b": mask_b_up,
            "scale_factor": scale_factor,
            "sr_size": (sr_h, sr_w),
            "input_size": (h, w),
        }
