"""M2: Logo-Aware QR Engine — generate QR codes optimised for logo integration.

Implements:
    2.1  Logo safe-zone calculator
    2.2  Geometry-aware mask optimiser
    2.3  ECC budget tracker
    2.4  Finder-pattern integration (rounded / dots)
    2.5  Module shape renderer (circle, rounded, square)
    2.6  Dynamic contrast booster
    2.7  Adaptive colour with WCAG contrast validation
    2.8  Apple logo overlay + scan verification
    2.9  Leaf-as-anchor metadata
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from qrx.generator import ECC_NAMES, get_module_map
from qrx.logging import audit, get_logger, trace
from qrx.logo import (
    classify_module_zones,
    compute_module_coverage,
    create_apple_silhouette,
    fit_logo_to_qr,
    load_logo,
    logo_to_binary_mask,
)

log = get_logger("logo_qr")

# ---------------------------------------------------------------------------
# QR spec tables (versions 1-7, extend as needed)
# ---------------------------------------------------------------------------

# Total codewords per version (data + ECC)
_TOTAL_CODEWORDS = {
    1: 26, 2: 44, 3: 70, 4: 100, 5: 134, 6: 172, 7: 196,
}

# Data codewords per (version, ecc_letter)
_DATA_CODEWORDS = {
    (1, "L"): 19, (1, "M"): 16, (1, "Q"): 13, (1, "H"): 9,
    (2, "L"): 34, (2, "M"): 28, (2, "Q"): 22, (2, "H"): 16,
    (3, "L"): 55, (3, "M"): 44, (3, "Q"): 34, (3, "H"): 26,
    (4, "L"): 80, (4, "M"): 64, (4, "Q"): 48, (4, "H"): 36,
    (5, "L"): 108, (5, "M"): 86, (5, "Q"): 62, (5, "H"): 46,
    (6, "L"): 136, (6, "M"): 108, (6, "Q"): 76, (6, "H"): 60,
    (7, "L"): 156, (7, "M"): 124, (7, "Q"): 88, (7, "H"): 66,
}


# ---------------------------------------------------------------------------
# 2.3  ECC budget tracker
# ---------------------------------------------------------------------------

@dataclass
class ECCBudget:
    """ECC error-correction budget analysis."""

    version: int
    ecc: str
    total_modules: int
    data_ecc_modules: int
    fixed_modules: int
    total_codewords: int
    data_codewords: int
    ecc_codewords: int
    correctable_codewords: int
    correctable_modules: int
    logo_covered_modules: int = 0
    budget_used_pct: float = 0.0
    safe: bool = True

    def summary(self) -> str:
        grid = int(self.total_modules ** 0.5)
        return (
            f"ECC Budget (V{self.version}-{self.ecc}):\n"
            f"  Grid: {grid}x{grid} = {self.total_modules} modules\n"
            f"  Fixed (finder/timing/etc): {self.fixed_modules}\n"
            f"  Data+ECC: {self.data_ecc_modules} modules "
            f"({self.total_codewords} codewords)\n"
            f"  ECC can correct: {self.correctable_codewords} codewords "
            f"= ~{self.correctable_modules} modules\n"
            f"  Logo covers: {self.logo_covered_modules} data modules "
            f"({self.budget_used_pct:.1f}% of budget)\n"
            f"  Status: {'SAFE' if self.safe else 'OVER BUDGET'}"
        )


@trace
def compute_ecc_budget(version: int, ecc: str, logo_covered_modules: int = 0) -> ECCBudget:
    """Compute the ECC error-correction budget for a given version/level."""
    size = version * 4 + 17
    total_modules = size * size

    total_cw = _TOTAL_CODEWORDS.get(version, 0)
    data_cw = _DATA_CODEWORDS.get((version, ecc.upper()), 0)
    ecc_cw = total_cw - data_cw

    # Reed-Solomon can correct up to floor(ecc_cw / 2) codewords
    correctable_cw = ecc_cw // 2
    correctable_modules = correctable_cw * 8  # 8 bits per codeword

    # Estimate fixed module count (finders + separators + timing + format + dark)
    fixed_estimate = 3 * 64 + (size - 16) * 2 + 31
    if version >= 2:
        fixed_estimate += 25  # one alignment pattern
    data_ecc_modules = total_modules - fixed_estimate

    budget_used = (
        logo_covered_modules / correctable_modules
        if correctable_modules > 0
        else 1.0
    )

    budget = ECCBudget(
        version=version,
        ecc=ecc.upper(),
        total_modules=total_modules,
        data_ecc_modules=data_ecc_modules,
        fixed_modules=fixed_estimate,
        total_codewords=total_cw,
        data_codewords=data_cw,
        ecc_codewords=ecc_cw,
        correctable_codewords=correctable_cw,
        correctable_modules=correctable_modules,
        logo_covered_modules=logo_covered_modules,
        budget_used_pct=budget_used * 100,
        safe=budget_used < 0.95,
    )

    audit(
        "ecc.budget", logger=log,
        version=version, ecc=ecc,
        correctable=correctable_modules,
        covered=logo_covered_modules,
        budget_pct=f"{budget_used:.1%}",
        safe=budget.safe,
    )
    return budget


# ---------------------------------------------------------------------------
# 2.2  Geometry-aware mask optimiser
# ---------------------------------------------------------------------------

@trace
def score_mask_for_logo(
    data: str,
    version: int,
    ecc: str,
    mask: int,
    zone_map: dict[str, set],
) -> int:
    """Score a mask pattern based on logo compatibility.

    Lower score = better mask for the logo.

    Penalties:
        +10  black data module fully under the logo  (ruins silhouette)
        +5   black data module on logo edge           (messy boundary)
        -1   white data module under the logo          (clean space)
    """
    mmap = get_module_map(data=data, version=version, ecc=ecc, mask=mask)
    modules = mmap["modules"]
    data_positions = set(map(tuple, mmap["data_positions"]))

    covered = zone_map["covered"]
    edge = zone_map["edge"]

    score = 0
    for r, c in data_positions:
        is_black = modules[r][c]
        if (r, c) in covered:
            score += 10 if is_black else -1
        elif (r, c) in edge:
            score += 5 if is_black else 0
    return score


@trace
def find_best_mask(
    data: str,
    version: int,
    ecc: str,
    zone_map: dict[str, set],
) -> tuple[int, dict[int, int]]:
    """Try all 8 masks and return the best one for the logo.

    Returns:
        (best_mask_index, {mask_idx: score, ...})
    """
    scores = {}
    for mask_idx in range(8):
        scores[mask_idx] = score_mask_for_logo(
            data, version, ecc, mask_idx, zone_map,
        )

    best = min(scores, key=scores.get)
    audit(
        "mask.optimised", logger=log,
        best_mask=best, best_score=scores[best],
        all_scores={str(k): v for k, v in sorted(scores.items())},
    )
    return best, scores


# ---------------------------------------------------------------------------
# 2.7  WCAG contrast ratio
# ---------------------------------------------------------------------------

def _linearize(channel: int) -> float:
    """Convert sRGB channel (0-255) to linear light value."""
    c = channel / 255.0
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def _luminance(rgb: tuple[int, int, int]) -> float:
    """Relative luminance per WCAG 2.0."""
    r, g, b = [_linearize(ch) for ch in rgb]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


@trace
def check_contrast(fg: tuple[int, ...], bg: tuple[int, ...]) -> float:
    """WCAG contrast ratio between two RGB colours (1.0 – 21.0)."""
    l1 = _luminance(fg[:3])
    l2 = _luminance(bg[:3])
    if l1 < l2:
        l1, l2 = l2, l1
    return (l1 + 0.05) / (l2 + 0.05)


# ---------------------------------------------------------------------------
# 2.5  Module shape renderer  +  2.4  Finder pattern integration
# ---------------------------------------------------------------------------

def _draw_module(
    draw: ImageDraw.Draw,
    px: int,
    py: int,
    box: int,
    margin: int,
    color: tuple[int, ...],
    shape: str,
) -> None:
    """Draw a single QR module with the given shape."""
    if shape == "circle":
        cx = px + box // 2
        cy = py + box // 2
        r = (box - margin * 2) // 2
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif shape == "rounded":
        rad = max(1, box // 4)
        draw.rounded_rectangle(
            [px + margin, py + margin, px + box - margin, py + box - margin],
            radius=rad, fill=color,
        )
    else:  # square
        draw.rectangle(
            [px + margin, py + margin, px + box - margin, py + box - margin],
            fill=color,
        )


def _contour_scale(coverage: float, softness: float = 1.0) -> float:
    """Map module coverage [0.0, 1.0] to dot scale [0.0, 1.0] via power curve.

    ``softness=0`` → hard edge (step function),
    ``softness=1`` → linear,
    ``softness=2`` → gradual (wider fade).
    """
    t = max(0.0, min(1.0, coverage))
    if softness <= 0:
        return 1.0 if t > 0 else 0.0
    return t ** (1.0 / softness)


def _draw_styled_finders(
    draw: ImageDraw.Draw,
    module_map: dict,
    box_size: int,
    border: int,
    fg: tuple[int, ...],
    bg: tuple[int, ...],
    style: str,
):
    """Draw the three 7x7 finder patterns with a cohesive style."""
    size = module_map["size"]
    origins = [(0, 0), (0, size - 7), (size - 7, 0)]  # TL, TR, BL

    for orig_r, orig_c in origins:
        ox = (orig_c + border) * box_size
        oy = (orig_r + border) * box_size
        fpx = 7 * box_size
        radius = box_size if style == "rounded" else box_size * 2

        # Outer dark ring
        draw.rounded_rectangle(
            [ox, oy, ox + fpx - 1, oy + fpx - 1],
            radius=radius, fill=fg,
        )
        # Inner white ring
        m1 = box_size
        draw.rounded_rectangle(
            [ox + m1, oy + m1, ox + fpx - 1 - m1, oy + fpx - 1 - m1],
            radius=max(1, radius // 2), fill=bg,
        )
        # Centre dark block
        m2 = 2 * box_size
        if style == "dots":
            cx = ox + fpx // 2
            cy = oy + fpx // 2
            cr = int(box_size * 1.4)
            draw.ellipse([cx - cr, cy - cr, cx + cr, cy + cr], fill=fg)
        else:
            draw.rounded_rectangle(
                [ox + m2, oy + m2, ox + fpx - 1 - m2, oy + fpx - 1 - m2],
                radius=max(1, radius // 3), fill=fg,
            )


@trace
def render_styled_qr(
    module_map: dict,
    box_size: int = 20,
    border: int = 4,
    shape: str = "circle",
    data_color: tuple[int, ...] = (0, 0, 0),
    bg_color: tuple[int, ...] = (255, 255, 255),
    finder_color: tuple[int, ...] | None = None,
    finder_style: str = "rounded",
) -> Image.Image:
    """Render QR modules with custom shapes, colours, and styled finders."""
    size = module_map["size"]
    modules = module_map["modules"]

    finder_set = set(map(tuple, module_map["finder_positions"]))
    alignment_set = set(map(tuple, module_map["alignment_positions"]))
    timing_set = set(map(tuple, module_map["timing_positions"]))
    format_set = set(map(tuple, module_map["format_positions"]))

    fc = finder_color or data_color
    total_px = (size + border * 2) * box_size
    img = Image.new("RGB", (total_px, total_px), bg_color)
    draw = ImageDraw.Draw(img)
    margin = max(1, box_size // 8)

    # 1) Styled finder patterns (drawn first as cohesive blocks)
    if finder_style in ("rounded", "dots"):
        _draw_styled_finders(draw, module_map, box_size, border, fc, bg_color, finder_style)
    else:
        # Standard square finders
        for r, c in finder_set:
            if modules[r][c]:
                px = (c + border) * box_size
                py = (r + border) * box_size
                draw.rectangle([px, py, px + box_size - 1, py + box_size - 1], fill=fc)

    # 2) Timing pattern (dots)
    for r, c in timing_set:
        if modules[r][c]:
            px = (c + border) * box_size
            py = (r + border) * box_size
            _draw_module(draw, px, py, box_size, margin, data_color, "circle")

    # 3) Alignment patterns (dots)
    for r, c in alignment_set:
        if modules[r][c]:
            px = (c + border) * box_size
            py = (r + border) * box_size
            _draw_module(draw, px, py, box_size, margin, data_color, "circle")

    # 4) Format info (small squares — critical metadata)
    for r, c in format_set:
        if modules[r][c]:
            px = (c + border) * box_size
            py = (r + border) * box_size
            draw.rectangle([px, py, px + box_size - 1, py + box_size - 1], fill=data_color)

    # 5) Data & ECC modules (user-selected shape)
    non_data = finder_set | alignment_set | timing_set | format_set
    for r in range(size):
        for c in range(size):
            if (r, c) in non_data or not modules[r][c]:
                continue
            px = (c + border) * box_size
            py = (r + border) * box_size
            _draw_module(draw, px, py, box_size, margin, data_color, shape)

    return img


# ---------------------------------------------------------------------------
# 2.6  Dynamic contrast booster
# ---------------------------------------------------------------------------

@trace
def apply_contrast_boost(image: Image.Image, strength: float = 0.15) -> Image.Image:
    """Mildly boost contrast to ensure scannability."""
    return ImageEnhance.Contrast(image).enhance(1.0 + strength)


# ---------------------------------------------------------------------------
# Shared geometry helper
# ---------------------------------------------------------------------------

def _scale_preserving_aspect(
    original_size: tuple[int, int],
    target: int,
) -> tuple[int, int]:
    """Scale (w, h) so the larger dimension equals *target*, preserving aspect."""
    w, h = original_size
    aspect = w / h
    if aspect >= 1:
        return target, int(target / aspect)
    return int(target * aspect), target


# ---------------------------------------------------------------------------
# 2.8  Logo composite
# ---------------------------------------------------------------------------

@trace
def composite_logo_on_qr(
    qr_image: Image.Image,
    logo_image: Image.Image,
    logo_color: tuple[int, ...] = (0, 0, 0),
    bg_margin_px: int = 6,
    coverage: float = 0.30,
) -> Image.Image:
    """Overlay the logo on the QR code with a tight white background.

    Uses the logo's own silhouette (dilated by *bg_margin_px*) as the white
    zone instead of a big ellipse — this destroys far fewer QR modules.

    Args:
        qr_image:      Styled QR (RGB).
        logo_image:    Grayscale silhouette (mode 'L').
        logo_color:    RGB fill for the logo silhouette.
        bg_margin_px:  White padding in pixels around the silhouette.
        coverage:      Target logo coverage fraction of QR area.
    """
    qr_w, qr_h = qr_image.size

    # --- Scale logo to target coverage (fraction of QR width) ---
    logo_gray = logo_image.convert("L")
    new_w, new_h = _scale_preserving_aspect(logo_gray.size, int(qr_w * coverage))
    logo_resized = logo_gray.resize((new_w, new_h), Image.LANCZOS)

    x_off = (qr_w - new_w) // 2
    y_off = (qr_h - new_h) // 2

    # --- Tight white zone: dilate the silhouette by bg_margin_px ---
    logo_mask_full = Image.new("L", (qr_w, qr_h), 0)
    logo_mask_full.paste(logo_resized, (x_off, y_off))

    # Dilate via MaxFilter (repeat for wider margin)
    white_zone = logo_mask_full
    passes = max(1, bg_margin_px // 2)
    for _ in range(passes):
        white_zone = white_zone.filter(ImageFilter.MaxFilter(5))

    # Apply white zone to QR image
    result_arr = np.array(qr_image)
    wz_arr = np.array(white_zone)
    result_arr[wz_arr > 64] = [255, 255, 255]
    result = Image.fromarray(result_arr)

    # --- Logo silhouette in logo_color via alpha composite ---
    logo_arr = np.array(logo_resized)
    rgba = np.zeros((new_h, new_w, 4), dtype=np.uint8)
    rgba[..., 0] = logo_color[0]
    rgba[..., 1] = logo_color[1]
    rgba[..., 2] = logo_color[2]
    rgba[..., 3] = logo_arr
    logo_rgba = Image.fromarray(rgba)

    result_rgba = result.convert("RGBA")
    result_rgba.paste(logo_rgba, (x_off, y_off), logo_rgba)

    final = result_rgba.convert("RGB")
    audit(
        "logo.composited", logger=log,
        qr_size=f"{qr_w}x{qr_h}",
        logo_size=f"{new_w}x{new_h}",
        coverage=coverage,
        bg_margin_px=bg_margin_px,
    )
    return final


# ---------------------------------------------------------------------------
# 2.8b  Rasterized logo (dot-grid rendering)
# ---------------------------------------------------------------------------

@trace
def rasterize_logo_on_qr(
    qr_image: Image.Image,
    logo_mask_on_qr: np.ndarray,
    zone_map: dict[str, set],
    box_size: int = 20,
    border: int = 4,
    logo_color: tuple[int, ...] = (0, 0, 0),
    shape: str = "circle",
    sub_grid: int = 3,
    cov_map: dict | None = None,
    contour_softness: float = 1.0,
    min_dot_scale: float = 0.15,
) -> Image.Image:
    """Render the logo as a dot grid matching the QR module style.

    Instead of pasting a solid silhouette, this subdivides each covered
    module into a *sub_grid* x *sub_grid* cell grid and draws small dots
    where the logo mask is active — creating a cohesive halftone effect.

    When *cov_map* is provided, edge modules are included with graduated
    dot sizes (contour strategy) for a smooth logo-to-QR transition.

    Args:
        qr_image:        Styled QR (RGB).
        logo_mask_on_qr: Bool array (total_px x total_px) from fit_logo_to_qr.
        zone_map:        Module zone classification (covered / edge / clear).
        box_size:        Pixel size of one QR module.
        border:          QR quiet-zone width in modules.
        logo_color:      RGB fill for the logo dots.
        shape:           Dot shape — 'circle', 'rounded', or 'square'.
        sub_grid:        Sub-divisions per module (default 3 → 9 dots max).
        cov_map:         Per-module coverage fractions (enables contour).
        contour_softness: Falloff curve (0=hard, 1=linear, 2=gradual).
        min_dot_scale:   Skip dots below this scale (avoids 1px noise).

    Returns:
        New RGB image with the logo rendered as dots.
    """
    result = qr_image.copy()
    draw = ImageDraw.Draw(result)

    covered = zone_map["covered"]
    edge = zone_map["edge"]

    sub_size = box_size // sub_grid
    pad = (box_size - sub_size * sub_grid) // 2
    margin = max(1, sub_size // 8)

    # Clear covered + edge modules to white
    for r, c in covered | edge:
        px = (c + border) * box_size
        py = (r + border) * box_size
        draw.rectangle([px, py, px + box_size - 1, py + box_size - 1], fill=(255, 255, 255))

    # Determine which modules to render dots for
    render_modules = covered | edge if cov_map is not None else covered

    # Draw sub-grid dots where logo mask is active
    mask_h, mask_w = logo_mask_on_qr.shape
    dots_drawn = 0

    for r, c in render_modules:
        mod_px = (c + border) * box_size
        mod_py = (r + border) * box_size

        # Compute contour scale for this module
        if cov_map is not None:
            scale = _contour_scale(cov_map.get((r, c), 0.0), contour_softness)
            if scale < min_dot_scale:
                continue
            scaled_sub = max(2, int(sub_size * scale))
            offset = (sub_size - scaled_sub) // 2
        else:
            scaled_sub = sub_size
            offset = 0

        scaled_margin = max(1, scaled_sub // 8)

        for si in range(sub_grid):
            for sj in range(sub_grid):
                center_x = mod_px + pad + sj * sub_size + sub_size // 2
                center_y = mod_py + pad + si * sub_size + sub_size // 2

                # Bounds check against the mask array
                if 0 <= center_y < mask_h and 0 <= center_x < mask_w:
                    if logo_mask_on_qr[center_y, center_x]:
                        dot_x = mod_px + pad + sj * sub_size + offset
                        dot_y = mod_py + pad + si * sub_size + offset
                        _draw_module(draw, dot_x, dot_y, scaled_sub, scaled_margin, logo_color, shape)
                        dots_drawn += 1

    audit(
        "logo.rasterized", logger=log,
        sub_grid=sub_grid,
        sub_size=sub_size,
        covered_modules=len(covered),
        edge_modules=len(edge),
        dots_drawn=dots_drawn,
        contour_enabled=cov_map is not None,
    )
    return result


# ---------------------------------------------------------------------------
# 2.9  Leaf-as-anchor metadata
# ---------------------------------------------------------------------------

def _detect_leaf_position(logo_mask: np.ndarray) -> dict:
    """Detect approximate leaf position from the logo mask.

    Returns a dict with normalised (x, y) centre of the leaf region
    (top 20 % of the logo bounding box, right of centre).
    """
    rows = np.any(logo_mask, axis=1)
    cols = np.any(logo_mask, axis=0)
    if not rows.any():
        return {"leaf_x": 0.5, "leaf_y": 0.0}
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    h = rmax - rmin
    top_region = logo_mask[rmin : rmin + int(h * 0.2), :]
    if top_region.any():
        tr_cols = np.where(np.any(top_region, axis=0))[0]
        leaf_x = float((tr_cols[0] + tr_cols[-1]) / 2 / logo_mask.shape[1])
        leaf_y = float(rmin / logo_mask.shape[0])
    else:
        leaf_x, leaf_y = 0.5, 0.0

    return {"leaf_x": round(leaf_x, 3), "leaf_y": round(leaf_y, 3)}


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

@trace
def generate_logo_qr(
    data: str,
    *,
    logo: str | Image.Image | None = None,
    use_apple: bool = False,
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
    """Generate a logo-aware QR code — the M2 master function.

    Returns a dict with:
        image          Final composite PIL Image
        mask           Best mask index used
        mask_scores    All 8 mask scores
        ecc_budget     ECCBudget analysis
        scan_results   list[ScanResult] (if verify_scan)
        scan_ok        bool
        leaf_anchor    Leaf position metadata
    """
    # -- 1. Obtain logo mask ------------------------------------------------
    if use_apple:
        logo_img = create_apple_silhouette(512)
    elif isinstance(logo, (str, Path)):
        logo_img = load_logo(str(logo))
    elif isinstance(logo, Image.Image):
        logo_img = logo.convert("L")
    else:
        raise ValueError("Provide a logo path, PIL Image, or set use_apple=True")

    logo_mask = logo_to_binary_mask(logo_img)

    # -- 2. Probe QR version -----------------------------------------------
    probe = get_module_map(data=data, version=version, ecc=ecc, mask=0)
    actual_version = probe["version"]
    qr_size = probe["size"]
    total_px = (qr_size + border * 2) * box_size

    # -- 3. Fit logo to QR pixel space & compute zones ---------------------
    logo_on_qr = fit_logo_to_qr(logo_mask, total_px, coverage=logo_coverage)
    cov_map = compute_module_coverage(logo_on_qr, qr_size, box_size, border)
    zone_map = classify_module_zones(cov_map)

    # -- 4. ECC budget check -----------------------------------------------
    data_positions = set(map(tuple, probe["data_positions"]))
    covered_data = len(zone_map["covered"] & data_positions)
    budget = compute_ecc_budget(actual_version, ecc, covered_data)

    if not budget.safe:
        log.warning("Logo covers too many data modules — shrinking by 30 %%")
        logo_on_qr = fit_logo_to_qr(logo_mask, total_px, coverage=logo_coverage * 0.7)
        cov_map = compute_module_coverage(logo_on_qr, qr_size, box_size, border)
        zone_map = classify_module_zones(cov_map)
        covered_data = len(zone_map["covered"] & data_positions)
        budget = compute_ecc_budget(actual_version, ecc, covered_data)

    # -- 5. Geometry-aware mask selection -----------------------------------
    best_mask, mask_scores = find_best_mask(data, actual_version, ecc, zone_map)

    # -- 6. Final module map with best mask ---------------------------------
    final_map = get_module_map(data=data, version=actual_version, ecc=ecc, mask=best_mask)

    # -- 7. Render styled QR -----------------------------------------------
    # Validate contrast
    ratio = check_contrast(data_color, bg_color)
    if ratio < 4.5:
        log.warning("Contrast ratio %.1f:1 is below 4.5:1 — scannability at risk", ratio)

    qr_img = render_styled_qr(
        final_map,
        box_size=box_size,
        border=border,
        shape=shape,
        data_color=data_color,
        bg_color=bg_color,
        finder_color=finder_color,
        finder_style=finder_style,
    )

    # -- 8. Contrast boost --------------------------------------------------
    qr_img = apply_contrast_boost(qr_img, strength=0.12)

    # -- 9. Rasterize logo as dot grid --------------------------------------
    final = rasterize_logo_on_qr(
        qr_img, logo_on_qr, zone_map,
        box_size=box_size,
        border=border,
        logo_color=logo_color,
        shape=shape,
        cov_map=cov_map,
        contour_softness=contour_softness,
    )

    # -- 10. Leaf-as-anchor metadata ----------------------------------------
    leaf_anchor = _detect_leaf_position(logo_mask)

    # -- 11. Verify scan ----------------------------------------------------
    scan_results = []
    scan_ok = None
    if verify_scan:
        from qrx.verify import verify

        scan_results = verify(final, expected_data=data)
        scan_ok = any(r.success for r in scan_results)
        for sr in scan_results:
            audit(
                "logo_qr.verify", logger=log,
                decoder=sr.decoder,
                success=sr.success,
                data=(sr.decoded_data or "")[:80],
                error=sr.error,
            )

    audit(
        "logo_qr.generated", logger=log,
        data=data[:80],
        version=actual_version,
        mask=best_mask,
        shape=shape,
        finder_style=finder_style,
        contrast_ratio=f"{ratio:.1f}:1",
        ecc_budget_pct=f"{budget.budget_used_pct:.1f}%",
        scan_ok=scan_ok,
    )

    return {
        "image": final,
        "mask": best_mask,
        "mask_scores": mask_scores,
        "ecc_budget": budget,
        "scan_results": scan_results,
        "scan_ok": scan_ok,
        "leaf_anchor": leaf_anchor,
    }
