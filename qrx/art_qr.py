"""M4: AI-Blended QR Art — optimize QR codes to resemble a logo via bit-flipping and similarity scoring.

Implements:
    4.4  SSIM similarity scoring + perceptual hashing
    4.5  Brute-force bit-flipper (greedy module flipping)
    4.6  Best-of-N mask selector with bit-flipping
"""

import math

import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity

from qrx.generator import get_module_map
from qrx.logging import audit, get_logger, trace
from qrx.logo import (
    classify_module_zones,
    compute_module_coverage,
    create_apple_silhouette,
    fit_logo_to_qr,
    load_logo,
    logo_to_binary_mask,
)
from qrx.logo_qr import (
    _draw_module,
    compute_ecc_budget,
    generate_logo_qr,
    render_styled_qr,
)

log = get_logger("art_qr")


# ---------------------------------------------------------------------------
# 4.4  Perceptual hashing (fast comparison)
# ---------------------------------------------------------------------------

@trace
def perceptual_hash(image: Image.Image, hash_size: int = 16) -> np.ndarray:
    """Compute a perceptual hash (pHash) of an image.

    Uses a simplified DCT-based approach:
    1. Resize to hash_size x hash_size
    2. Convert to grayscale
    3. Compute DCT-like features via mean comparison
    4. Return binary hash as bool array

    Args:
        image: Input PIL Image.
        hash_size: Hash dimension (hash_size x hash_size bits). Default 16.

    Returns:
        Boolean numpy array of shape (hash_size * hash_size,).
    """
    gray = image.convert("L").resize((hash_size, hash_size), Image.LANCZOS)
    pixels = np.array(gray, dtype=np.float64)
    mean_val = pixels.mean()
    return pixels.flatten() >= mean_val


@trace
def phash_distance(hash_a: np.ndarray, hash_b: np.ndarray) -> float:
    """Compute normalised Hamming distance between two perceptual hashes.

    Returns:
        Float 0.0 (identical) to 1.0 (completely different).
    """
    if hash_a.shape != hash_b.shape:
        raise ValueError("Hash arrays must have the same shape")
    diff = np.count_nonzero(hash_a != hash_b)
    return float(diff) / hash_a.size


@trace
def phash_similarity(image_a: Image.Image, image_b: Image.Image, hash_size: int = 16) -> float:
    """Compute perceptual hash similarity between two images.

    Args:
        image_a: First image.
        image_b: Second image.
        hash_size: Hash dimension.

    Returns:
        Float 0.0 (completely different) to 1.0 (identical).
    """
    h_a = perceptual_hash(image_a, hash_size)
    h_b = perceptual_hash(image_b, hash_size)
    return 1.0 - phash_distance(h_a, h_b)


# ---------------------------------------------------------------------------
# 4.4  SSIM similarity scoring
# ---------------------------------------------------------------------------

@trace
def compute_similarity(qr_image: Image.Image, logo_reference: Image.Image, method: str = "ssim") -> float:
    """Compare two images using structural similarity or perceptual hashing.

    Args:
        qr_image: The QR code image.
        logo_reference: The logo reference image (same size target).
        method: Similarity method — "ssim" (default) or "phash".

    Returns:
        Float score 0.0-1.0 (higher = more similar).
    """
    if method == "phash":
        return phash_similarity(qr_image, logo_reference)

    qr_gray = np.array(qr_image.convert("L"))
    logo_gray = np.array(logo_reference.convert("L"))

    # Resize logo reference to match QR image if needed
    if qr_gray.shape != logo_gray.shape:
        logo_ref_resized = logo_reference.convert("L").resize(
            (qr_gray.shape[1], qr_gray.shape[0]), Image.LANCZOS,
        )
        logo_gray = np.array(logo_ref_resized)

    score = structural_similarity(qr_gray, logo_gray, data_range=255)
    # Clamp to [0, 1]
    return max(0.0, min(1.0, float(score)))


@trace
def identify_flippable_modules(
    module_map: dict,
    zone_map: dict[str, set],
    budget: object,
) -> list[tuple[int, int]]:
    """Find data modules safe to flip within remaining ECC capacity.

    Args:
        module_map: QR module map from get_module_map.
        zone_map: Module zone classification (covered/edge/clear).
        budget: ECCBudget with correctable_modules and logo_covered_modules.

    Returns:
        List of (row, col) positions sorted by distance from center.
    """
    data_positions = set(map(tuple, module_map["data_positions"]))
    clear_positions = zone_map["clear"]

    # Only flip data modules in clear zone (not covered or edge)
    flippable = list(data_positions & clear_positions)

    # Sort by distance from center (closest first — most logo impact)
    size = module_map["size"]
    center = size / 2.0

    flippable.sort(key=lambda pos: math.hypot(pos[0] - center, pos[1] - center))

    # Limit to remaining ECC capacity
    remaining = budget.correctable_modules - budget.logo_covered_modules
    remaining = max(0, remaining)

    result = flippable[:remaining]
    audit(
        "art.flippable_identified", logger=log,
        total_clear_data=len(flippable),
        ecc_remaining=remaining,
        selected=len(result),
    )
    return result


@trace
def flip_and_score(
    qr_image: Image.Image,
    module_map: dict,
    positions_to_flip: list[tuple[int, int]],
    box_size: int,
    border: int,
    logo_reference: Image.Image,
    shape: str,
    data_color: tuple[int, ...],
    bg_color: tuple[int, ...],
) -> tuple[Image.Image, float]:
    """Flip a batch of modules and compute the resulting similarity score.

    Args:
        qr_image: Current QR code image.
        module_map: QR module map.
        positions_to_flip: List of (row, col) module positions to flip.
        box_size: Pixel size of each module.
        border: Quiet zone width in modules.
        logo_reference: Logo reference image for comparison.
        shape: Module shape (circle/rounded/square).
        data_color: Foreground colour.
        bg_color: Background colour.

    Returns:
        Tuple of (modified_image, similarity_score).
    """
    img = qr_image.copy()
    draw = ImageDraw.Draw(img)
    modules = module_map["modules"]
    margin = max(1, box_size // 8)

    for r, c in positions_to_flip:
        px = (c + border) * box_size
        py = (r + border) * box_size
        is_black = modules[r][c]
        # Flip: black -> white, white -> black
        new_color = bg_color if is_black else data_color
        # Clear the module area first
        draw.rectangle([px, py, px + box_size - 1, py + box_size - 1], fill=bg_color)
        if new_color != bg_color:
            _draw_module(draw, px, py, box_size, margin, new_color, shape)

    score = compute_similarity(img, logo_reference)
    return img, score


def _create_logo_reference(
    logo_mask: np.ndarray,
    total_px: int,
    logo_coverage: float,
    data_color: tuple[int, ...],
    bg_color: tuple[int, ...],
) -> Image.Image:
    """Create an idealized logo reference image for similarity comparison."""
    logo_on_qr = fit_logo_to_qr(logo_mask, total_px, coverage=logo_coverage)
    ref = Image.new("RGB", (total_px, total_px), bg_color)
    ref_arr = np.array(ref)
    ref_arr[logo_on_qr] = list(data_color)
    return Image.fromarray(ref_arr)


# ---------------------------------------------------------------------------
# 4.5  Brute-force bit-flipper (internal — operates on a single mask variant)
# ---------------------------------------------------------------------------

def _bitflip_optimize(
    base_image: Image.Image,
    module_map: dict,
    flippable: list[tuple[int, int]],
    logo_reference: Image.Image,
    data: str,
    iterations: int,
    box_size: int,
    border: int,
    shape: str,
    data_color: tuple[int, ...],
    bg_color: tuple[int, ...],
    verify_scan: bool,
) -> tuple[Image.Image, float, int]:
    """Run greedy bit-flip optimization on a single QR image.

    Returns:
        (optimized_image, final_similarity_score, total_flips_applied)
    """
    current_image = base_image.copy()
    current_score = compute_similarity(base_image, logo_reference)
    total_flips = 0
    batch_size = 8

    for iteration in range(iterations):
        if not flippable:
            break

        improved_in_round = 0

        for batch_start in range(0, len(flippable), batch_size):
            batch = flippable[batch_start:batch_start + batch_size]

            candidate_image, candidate_score = flip_and_score(
                current_image, module_map, batch,
                box_size=box_size, border=border,
                logo_reference=logo_reference,
                shape=shape,
                data_color=data_color,
                bg_color=bg_color,
            )

            if candidate_score > current_score:
                if verify_scan:
                    from qrx.verify import verify
                    scan_results = verify(candidate_image, expected_data=data)
                    scan_ok = any(r.success for r in scan_results)
                    if not scan_ok:
                        continue

                current_image = candidate_image
                current_score = candidate_score
                total_flips += len(batch)
                improved_in_round += len(batch)

        audit(
            "art.iteration", logger=log,
            iteration=iteration + 1,
            score=round(current_score, 4),
            flips_in_round=improved_in_round,
            total_flips=total_flips,
        )

        if improved_in_round == 0:
            break

    return current_image, current_score, total_flips


# ---------------------------------------------------------------------------
# 4.6  Best-of-N mask selector
# ---------------------------------------------------------------------------

@trace
def best_of_n_masks(
    data: str,
    *,
    logo_mask: np.ndarray,
    logo_reference: Image.Image,
    actual_version: int,
    ecc: str = "H",
    iterations: int = 3,
    box_size: int = 20,
    border: int = 4,
    shape: str = "circle",
    data_color: tuple[int, ...] = (0, 0, 0),
    bg_color: tuple[int, ...] = (255, 255, 255),
    finder_color: tuple[int, ...] | None = None,
    finder_style: str = "rounded",
    logo_color: tuple[int, ...] = (0, 0, 0),
    logo_coverage: float = 0.35,
    contour_softness: float = 1.0,
    verify_scan: bool = True,
) -> dict:
    """Try all 8 mask patterns, apply bit-flipping to each, and return the best.

    For each mask:
      1. Generate QR with that mask via generate_logo_qr
      2. Run bit-flip optimization
      3. Track similarity score

    Selects the mask+flips combination with the highest SSIM that still scans.

    Returns dict with:
        best_mask, image, similarity, flips_applied, all_mask_similarities, ...
    """
    from qrx.logo_qr import (
        apply_contrast_boost,
        find_best_mask,
        rasterize_logo_on_qr,
    )

    total_px_probe = get_module_map(data=data, version=actual_version, ecc=ecc, mask=0)
    qr_size = total_px_probe["size"]
    total_px = (qr_size + border * 2) * box_size

    logo_on_qr = fit_logo_to_qr(logo_mask, total_px, coverage=logo_coverage)
    cov_map = compute_module_coverage(logo_on_qr, qr_size, box_size, border)
    zone_map = classify_module_zones(cov_map)

    # Use pHash for fast pre-screening, then SSIM for final selection
    best_mask = -1
    best_score = -1.0
    best_image = None
    best_flips = 0
    all_mask_similarities: dict[int, float] = {}

    # Pre-screen with pHash to find top 3 candidates, then do full SSIM on those
    phash_scores: dict[int, float] = {}
    mask_images: dict[int, Image.Image] = {}
    mask_module_maps: dict[int, dict] = {}

    for mask_idx in range(8):
        mmap = get_module_map(data=data, version=actual_version, ecc=ecc, mask=mask_idx)
        qr_img = render_styled_qr(
            mmap, box_size=box_size, border=border, shape=shape,
            data_color=data_color, bg_color=bg_color,
            finder_color=finder_color, finder_style=finder_style,
        )
        qr_img = apply_contrast_boost(qr_img, strength=0.12)
        qr_img = rasterize_logo_on_qr(
            qr_img, logo_on_qr, zone_map,
            box_size=box_size, border=border,
            logo_color=logo_color, shape=shape,
            cov_map=cov_map, contour_softness=contour_softness,
        )
        mask_images[mask_idx] = qr_img
        mask_module_maps[mask_idx] = mmap
        phash_scores[mask_idx] = phash_similarity(qr_img, logo_reference)

    # Select top 3 masks by pHash for full optimization
    top_masks = sorted(phash_scores, key=phash_scores.get, reverse=True)[:3]

    audit(
        "art.phash_prescreen", logger=log,
        phash_scores={str(k): round(v, 4) for k, v in sorted(phash_scores.items())},
        top_masks=top_masks,
    )

    for mask_idx in top_masks:
        mmap = mask_module_maps[mask_idx]
        qr_img = mask_images[mask_idx]

        data_positions = set(map(tuple, mmap["data_positions"]))
        covered_data = len(zone_map["covered"] & data_positions)
        budget = compute_ecc_budget(actual_version, ecc, covered_data)

        flippable = identify_flippable_modules(mmap, zone_map, budget)

        opt_image, opt_score, opt_flips = _bitflip_optimize(
            qr_img, mmap, flippable, logo_reference, data,
            iterations=iterations, box_size=box_size, border=border,
            shape=shape, data_color=data_color, bg_color=bg_color,
            verify_scan=verify_scan,
        )

        all_mask_similarities[mask_idx] = opt_score

        # Verify scannability of final result
        if verify_scan:
            from qrx.verify import verify
            scan_results = verify(opt_image, expected_data=data)
            if not any(r.success for r in scan_results):
                continue

        if opt_score > best_score:
            best_score = opt_score
            best_mask = mask_idx
            best_image = opt_image
            best_flips = opt_flips

    audit(
        "art.best_of_n", logger=log,
        best_mask=best_mask,
        best_score=round(best_score, 4),
        all_similarities={str(k): round(v, 4) for k, v in sorted(all_mask_similarities.items())},
    )

    return {
        "best_mask": best_mask,
        "image": best_image,
        "similarity": best_score,
        "flips_applied": best_flips,
        "all_mask_similarities": all_mask_similarities,
    }


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

@trace
def optimize_art_qr(
    data: str,
    *,
    logo: str | Image.Image | None = None,
    use_apple: bool = False,
    iterations: int = 3,
    try_all_masks: bool = False,
    version: int | None = None,
    ecc: str = "H",
    box_size: int = 20,
    border: int = 4,
    shape: str = "circle",
    data_color: tuple[int, ...] = (0, 0, 0),
    bg_color: tuple[int, ...] = (255, 255, 255),
    finder_color: tuple[int, ...] | None = None,
    finder_style: str = "rounded",
    logo_color: tuple[int, ...] = (0, 0, 0),
    logo_coverage: float = 0.35,
    contour_softness: float = 1.0,
    verify_scan: bool = True,
) -> dict:
    """Optimize a QR code to look more like the logo via greedy bit-flipping.

    Starts with generate_logo_qr output, then iteratively flips data modules
    in the clear zone to improve SSIM similarity with the logo reference,
    while maintaining scannability.

    When try_all_masks=True, uses best_of_n_masks to try all 8 mask patterns
    (with pHash pre-screening for speed) before applying bit-flipping.

    Returns a dict with:
        image              Final optimized PIL Image
        similarity_before  SSIM score before optimization
        similarity_after   SSIM score after optimization
        flips_applied      Number of module flips accepted
        scan_ok            bool
        phash_similarity   Perceptual hash similarity score
        + all fields from generate_logo_qr
    """
    from pathlib import Path

    # -- 1. Generate base logo QR --
    base_result = generate_logo_qr(
        data=data,
        logo=logo,
        use_apple=use_apple,
        version=version,
        ecc=ecc,
        box_size=box_size,
        border=border,
        shape=shape,
        data_color=data_color,
        bg_color=bg_color,
        finder_color=finder_color,
        finder_style=finder_style,
        logo_color=logo_color,
        logo_coverage=logo_coverage,
        contour_softness=contour_softness,
        verify_scan=verify_scan,
    )

    base_image = base_result["image"]
    budget = base_result["ecc_budget"]

    # -- 2. Obtain logo mask for reference image --
    if use_apple:
        logo_img = create_apple_silhouette(512)
    elif isinstance(logo, (str, Path)):
        logo_img = load_logo(str(logo))
    elif isinstance(logo, Image.Image):
        logo_img = logo.convert("L")
    else:
        raise ValueError("Provide a logo path, PIL Image, or set use_apple=True")

    logo_mask = logo_to_binary_mask(logo_img)

    # -- 3. Create logo reference image --
    total_px = base_image.size[0]
    logo_reference = _create_logo_reference(
        logo_mask, total_px, logo_coverage, data_color, bg_color,
    )

    # -- 4. Compute baseline similarity --
    similarity_before = compute_similarity(base_image, logo_reference)

    # -- 5. Optimization: best-of-N or single-mask --
    if try_all_masks:
        bon_result = best_of_n_masks(
            data,
            logo_mask=logo_mask,
            logo_reference=logo_reference,
            actual_version=budget.version,
            ecc=ecc,
            iterations=iterations,
            box_size=box_size,
            border=border,
            shape=shape,
            data_color=data_color,
            bg_color=bg_color,
            finder_color=finder_color,
            finder_style=finder_style,
            logo_color=logo_color,
            logo_coverage=logo_coverage,
            contour_softness=contour_softness,
            verify_scan=verify_scan,
        )
        current_image = bon_result["image"]
        total_flips = bon_result["flips_applied"]
        used_mask = bon_result["best_mask"]
        all_mask_similarities = bon_result["all_mask_similarities"]
    else:
        # Single-mask bit-flip optimization
        probe = get_module_map(data=data, version=budget.version, ecc=ecc, mask=base_result["mask"])
        qr_size = probe["size"]
        logo_on_qr = fit_logo_to_qr(logo_mask, total_px, coverage=logo_coverage)
        cov_map = compute_module_coverage(logo_on_qr, qr_size, box_size, border)
        zone_map = classify_module_zones(cov_map)
        flippable = identify_flippable_modules(probe, zone_map, budget)

        current_image, _, total_flips = _bitflip_optimize(
            base_image, probe, flippable, logo_reference, data,
            iterations=iterations, box_size=box_size, border=border,
            shape=shape, data_color=data_color, bg_color=bg_color,
            verify_scan=verify_scan,
        )
        used_mask = base_result["mask"]
        all_mask_similarities = None

    # -- 6. Final scan verification --
    scan_results = []
    scan_ok = None
    if verify_scan:
        from qrx.verify import verify
        scan_results = verify(current_image, expected_data=data)
        scan_ok = any(r.success for r in scan_results)
        for sr in scan_results:
            audit(
                "art_qr.verify", logger=log,
                decoder=sr.decoder,
                success=sr.success,
                data=(sr.decoded_data or "")[:80],
                error=sr.error,
            )

    similarity_after = compute_similarity(current_image, logo_reference)
    phash_score = phash_similarity(current_image, logo_reference)

    audit(
        "art_qr.optimized", logger=log,
        data=data[:80],
        iterations=iterations,
        try_all_masks=try_all_masks,
        similarity_before=round(similarity_before, 4),
        similarity_after=round(similarity_after, 4),
        phash_similarity=round(phash_score, 4),
        flips_applied=total_flips,
        mask_used=used_mask,
        scan_ok=scan_ok,
    )

    return {
        "image": current_image,
        "similarity_before": similarity_before,
        "similarity_after": similarity_after,
        "phash_similarity": phash_score,
        "flips_applied": total_flips,
        "scan_ok": scan_ok,
        "scan_results": scan_results,
        "mask": used_mask,
        "mask_scores": base_result["mask_scores"],
        "all_mask_similarities": all_mask_similarities,
        "ecc_budget": base_result["ecc_budget"],
        "leaf_anchor": base_result["leaf_anchor"],
    }
