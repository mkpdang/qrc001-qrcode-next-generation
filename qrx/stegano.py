"""M6: Chroma & Grayscale Steganography — hide QR data in color/grayscale channels.

Implements:
    6.1  B&W halftone QR with 3x3 sub-module dithering
    6.2  Chroma steganographic QR with hue-shift encoding
    6.3  Calibration reference embedding
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from qrx.generator import get_module_map
from qrx.logging import audit, get_logger, trace
from qrx.logo import create_apple_silhouette, load_logo
from qrx.logo_qr import (
    _draw_styled_finders,
    find_best_mask,
)

log = get_logger("stegano")


# ---------------------------------------------------------------------------
# 6.1  Bayer dithering threshold matrix
# ---------------------------------------------------------------------------

def _bayer_threshold_2x2() -> list[list[int]]:
    """Return a 2x2 Bayer matrix normalized to 0-7 range for 8-pixel dithering.

    The canonical 2x2 Bayer matrix is::

        [[0, 2],
         [3, 1]]

    We scale each entry so it maps neatly onto the 8 surrounding sub-pixels.
    """
    return [[0, 4], [6, 2]]


# ---------------------------------------------------------------------------
# 6.1  Halftone QR generation
# ---------------------------------------------------------------------------

@trace
def generate_halftone_qr(
    data: str,
    *,
    logo: str | Image.Image | None = None,
    use_apple: bool = False,
    version: int | None = None,
    ecc: str = "H",
    box_size: int = 20,
    border: int = 4,
    data_color: tuple[int, ...] = (0, 0, 0),
    bg_color: tuple[int, ...] = (255, 255, 255),
    finder_color: tuple[int, ...] | None = None,
    finder_style: str = "rounded",
    verify_scan: bool = True,
) -> dict:
    """Generate a B&W halftone QR code with 3x3 sub-module dithering.

    For each QR module, the module is subdivided into a 3x3 sub-pixel grid:
      - Center sub-pixel = TRUE data bit (pure black or white)
      - Surrounding 8 sub-pixels = logo grayscale dithering

    The number of black surrounding sub-pixels is determined by the logo
    grayscale value at that module position using ordered Bayer dithering.

    Returns:
        dict with: image, scan_ok, scan_results, version, mask
    """
    # -- 1. Load logo for grayscale values ----------------------------------
    if use_apple:
        logo_img = create_apple_silhouette(512)
    elif isinstance(logo, (str, Path)):
        logo_img = load_logo(str(logo))
    elif isinstance(logo, Image.Image):
        logo_img = logo.convert("L")
    else:
        logo_img = None

    # -- 2. Probe QR version ------------------------------------------------
    probe = get_module_map(data=data, version=version, ecc=ecc, mask=0)
    actual_version = probe["version"]
    qr_size = probe["size"]

    # -- 3. Find best mask --------------------------------------------------
    # Build a minimal zone_map (no logo coverage for mask selection in halftone)
    dummy_zone = {"covered": set(), "edge": set(), "clear": set()}

    if logo_img is not None:
        # Build zone map from logo for better mask selection
        from qrx.logo import (
            classify_module_zones,
            compute_module_coverage,
            fit_logo_to_qr,
            logo_to_binary_mask,
        )

        logo_mask = logo_to_binary_mask(logo_img)
        total_px = (qr_size + border * 2) * box_size
        logo_on_qr = fit_logo_to_qr(logo_mask, total_px, coverage=0.35)
        cov_map = compute_module_coverage(logo_on_qr, qr_size, box_size, border)
        zone_map = classify_module_zones(cov_map)
        best_mask, mask_scores = find_best_mask(data, actual_version, ecc, zone_map)
    else:
        best_mask, mask_scores = find_best_mask(data, actual_version, ecc, dummy_zone)

    # -- 4. Final module map with best mask ---------------------------------
    final_map = get_module_map(data=data, version=actual_version, ecc=ecc, mask=best_mask)
    modules = final_map["modules"]

    # -- 5. Prepare logo grayscale at module resolution ---------------------
    logo_gray_grid = None
    if logo_img is not None:
        # Resize logo to qr_size x qr_size so each pixel = one module
        logo_resized = logo_img.convert("L").resize((qr_size, qr_size), Image.LANCZOS)
        logo_gray_grid = np.array(logo_resized)

    # -- 6. Build image with 3x3 sub-module grid ----------------------------
    # Each QR module gets box_size pixels. We subdivide into 3x3 sub-cells.
    sub_size = box_size // 3
    total_px = (qr_size + border * 2) * box_size
    img = Image.new("RGB", (total_px, total_px), bg_color)
    draw = ImageDraw.Draw(img)

    # Identify fixed pattern positions
    finder_set = set(map(tuple, final_map["finder_positions"]))
    alignment_set = set(map(tuple, final_map["alignment_positions"]))
    timing_set = set(map(tuple, final_map["timing_positions"]))
    format_set = set(map(tuple, final_map["format_positions"]))
    fixed_set = finder_set | alignment_set | timing_set | format_set

    fc = finder_color or data_color

    # -- 7. Draw styled finder patterns (critical for scanning) -------------
    if finder_style in ("rounded", "dots"):
        _draw_styled_finders(draw, final_map, box_size, border, fc, bg_color, finder_style)
    else:
        for r, c in finder_set:
            if modules[r][c]:
                px = (c + border) * box_size
                py = (r + border) * box_size
                draw.rectangle([px, py, px + box_size - 1, py + box_size - 1], fill=fc)

    # -- 8. Draw timing, alignment, format modules normally -----------------
    for r, c in timing_set:
        if modules[r][c]:
            px = (c + border) * box_size
            py = (r + border) * box_size
            draw.rectangle([px, py, px + box_size - 1, py + box_size - 1], fill=data_color)

    for r, c in alignment_set:
        if modules[r][c]:
            px = (c + border) * box_size
            py = (r + border) * box_size
            draw.rectangle([px, py, px + box_size - 1, py + box_size - 1], fill=data_color)

    for r, c in format_set:
        if modules[r][c]:
            px = (c + border) * box_size
            py = (r + border) * box_size
            draw.rectangle([px, py, px + box_size - 1, py + box_size - 1], fill=data_color)

    # -- 9. Draw data modules with 3x3 halftone dithering -------------------
    bayer = _bayer_threshold_2x2()

    # The 8 surrounding positions in a 3x3 grid (excluding center)
    surround_positions = [
        (0, 0), (0, 1), (0, 2),
        (1, 0),         (1, 2),
        (2, 0), (2, 1), (2, 2),
    ]

    for r in range(qr_size):
        for c in range(qr_size):
            if (r, c) in fixed_set:
                continue

            px = (c + border) * box_size
            py = (r + border) * box_size
            is_black = modules[r][c]

            # Center sub-pixel = true data bit
            center_x = px + sub_size
            center_y = py + sub_size
            center_color = data_color if is_black else bg_color
            draw.rectangle(
                [center_x, center_y, center_x + sub_size - 1, center_y + sub_size - 1],
                fill=center_color,
            )

            # Surrounding 8 sub-pixels = logo dithering
            if logo_gray_grid is not None:
                gray_val = int(logo_gray_grid[r, c])
                # gray_val: 0=background (no logo), 255=logo (full coverage)
                # Number of black surrounding pixels = round(gray_val / 255 * 8)
                n_black = round(gray_val / 255.0 * 8)
            else:
                # No logo: surrounding pixels match the data bit
                n_black = 8 if is_black else 0

            # Build threshold ordering for the 8 surrounding positions
            # Map 2x2 Bayer matrix across the 8 positions
            thresholds = []
            for idx, (si, sj) in enumerate(surround_positions):
                bayer_val = bayer[si % 2][sj % 2]
                thresholds.append((bayer_val, idx, si, sj))
            thresholds.sort(key=lambda t: (t[0], t[1]))

            for rank, (_, _, si, sj) in enumerate(thresholds):
                sub_x = px + sj * sub_size
                sub_y = py + si * sub_size
                sub_color = data_color if rank < n_black else bg_color
                draw.rectangle(
                    [sub_x, sub_y, sub_x + sub_size - 1, sub_y + sub_size - 1],
                    fill=sub_color,
                )

    # -- 10. Verify scan ----------------------------------------------------
    scan_results = []
    scan_ok = None
    if verify_scan:
        from qrx.verify import verify

        scan_results = verify(img, expected_data=data)
        scan_ok = any(r.success for r in scan_results)
        for sr in scan_results:
            audit(
                "halftone_qr.verify", logger=log,
                decoder=sr.decoder,
                success=sr.success,
                data=(sr.decoded_data or "")[:80],
                error=sr.error,
            )

    audit(
        "halftone_qr.generated", logger=log,
        data=data[:80],
        version=actual_version,
        mask=best_mask,
        has_logo=logo_img is not None,
        scan_ok=scan_ok,
    )

    return {
        "image": img,
        "scan_ok": scan_ok,
        "scan_results": scan_results,
        "version": actual_version,
        "mask": best_mask,
    }


# ---------------------------------------------------------------------------
# 6.3  Calibration reference embedding
# ---------------------------------------------------------------------------

@trace
def _embed_calibration(
    image: Image.Image,
    border: int,
    box_size: int,
) -> Image.Image:
    """Add B&W reference pixels in quiet zone corners for decoder calibration.

    Places small black and white squares in the four corners of the quiet zone
    so a decoder can establish absolute black/white reference points.
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    cal_size = max(2, box_size // 4)

    # Top-left: black reference
    draw.rectangle([0, 0, cal_size - 1, cal_size - 1], fill=(0, 0, 0))
    # Top-right: white reference
    draw.rectangle([w - cal_size, 0, w - 1, cal_size - 1], fill=(255, 255, 255))
    # Bottom-left: white reference
    draw.rectangle([0, h - cal_size, cal_size - 1, h - 1], fill=(255, 255, 255))
    # Bottom-right: black reference
    draw.rectangle([w - cal_size, h - cal_size, w - 1, h - 1], fill=(0, 0, 0))

    audit(
        "calibration.embedded", logger=log,
        cal_size=cal_size,
        image_size=f"{w}x{h}",
    )
    return img


# ---------------------------------------------------------------------------
# 6.2  Chroma steganographic QR
# ---------------------------------------------------------------------------

@trace
def generate_chroma_qr(
    data: str,
    *,
    logo_color_image: str | Image.Image | None = None,
    use_apple: bool = False,
    version: int | None = None,
    ecc: str = "H",
    box_size: int = 20,
    border: int = 4,
    finder_color: tuple[int, ...] | None = None,
    finder_style: str = "rounded",
    verify_scan: bool = True,
) -> dict:
    """Generate a color steganographic QR where data is encoded as hue shifts.

    Starts with a full-color logo image and overlays QR data as subtle hue
    shifts (+/-5 in HSV H channel). Module=1 shifts hue +5, module=0 shifts -5.

    Returns:
        dict with: image, scan_ok, delta_e_max
    """
    # -- 1. Load color image ------------------------------------------------
    if use_apple:
        # Create a colored apple logo on a gradient background
        gray = create_apple_silhouette(512)
        # Convert to RGB with a colorful background
        arr = np.array(gray)
        color_img = Image.new("RGB", (512, 512), (100, 150, 200))
        color_arr = np.array(color_img)
        # Where logo is present, make it dark
        mask = arr > 128
        color_arr[mask] = [40, 40, 40]
        color_img = Image.fromarray(color_arr)
    elif isinstance(logo_color_image, (str, Path)):
        color_img = Image.open(str(logo_color_image)).convert("RGB")
    elif isinstance(logo_color_image, Image.Image):
        color_img = logo_color_image.convert("RGB")
    else:
        # Default: create a gradient image
        color_img = Image.new("RGB", (512, 512), (128, 128, 200))

    # -- 2. Probe QR version ------------------------------------------------
    probe = get_module_map(data=data, version=version, ecc=ecc, mask=0)
    actual_version = probe["version"]
    qr_size = probe["size"]
    total_px = (qr_size + border * 2) * box_size

    # -- 3. Resize color image to QR canvas ---------------------------------
    color_canvas = color_img.resize((total_px, total_px), Image.LANCZOS)

    # -- 4. Get module map --------------------------------------------------
    final_map = get_module_map(data=data, version=actual_version, ecc=ecc, mask=0)
    modules = final_map["modules"]

    finder_set = set(map(tuple, final_map["finder_positions"]))
    alignment_set = set(map(tuple, final_map["alignment_positions"]))
    timing_set = set(map(tuple, final_map["timing_positions"]))
    format_set = set(map(tuple, final_map["format_positions"]))
    fixed_set = finder_set | alignment_set | timing_set | format_set

    # -- 5. Convert to HSV and apply hue shifts -----------------------------
    hsv_arr = np.array(color_canvas.convert("HSV")).astype(np.int16)
    original_arr = np.array(color_canvas).copy()
    hue_shift = 5

    for r in range(qr_size):
        for c in range(qr_size):
            if (r, c) in fixed_set:
                continue

            py_start = (r + border) * box_size
            px_start = (c + border) * box_size
            py_end = py_start + box_size
            px_end = px_start + box_size

            is_black = modules[r][c]
            shift = hue_shift if is_black else -hue_shift

            hsv_arr[py_start:py_end, px_start:px_end, 0] = np.clip(
                hsv_arr[py_start:py_end, px_start:px_end, 0] + shift, 0, 255,
            )

    # Convert back to RGB
    hsv_img = Image.fromarray(hsv_arr.astype(np.uint8), mode="HSV")
    result_img = hsv_img.convert("RGB")

    # -- 6. Draw finder patterns over the image (critical for scanning) -----
    draw = ImageDraw.Draw(result_img)
    fc = finder_color or (0, 0, 0)
    bg = (255, 255, 255)

    if finder_style in ("rounded", "dots"):
        _draw_styled_finders(draw, final_map, box_size, border, fc, bg, finder_style)
    else:
        for r, c in finder_set:
            if modules[r][c]:
                px = (c + border) * box_size
                py = (r + border) * box_size
                draw.rectangle([px, py, px + box_size - 1, py + box_size - 1], fill=fc)
            else:
                px = (c + border) * box_size
                py = (r + border) * box_size
                draw.rectangle([px, py, px + box_size - 1, py + box_size - 1], fill=bg)

    # Draw timing/alignment/format normally
    for r, c in timing_set | alignment_set | format_set:
        px = (c + border) * box_size
        py = (r + border) * box_size
        mod_color = (0, 0, 0) if modules[r][c] else (255, 255, 255)
        draw.rectangle([px, py, px + box_size - 1, py + box_size - 1], fill=mod_color)

    # -- 7. Compute Delta-E between original and modified -------------------
    modified_arr = np.array(result_img).astype(np.float64)
    orig_f = original_arr.astype(np.float64)
    # Approximate Delta-E as Euclidean distance in RGB (simplified)
    diff = modified_arr - orig_f
    per_pixel_de = np.sqrt(np.sum(diff ** 2, axis=2))
    delta_e_max = float(per_pixel_de.max())

    # -- 8. Embed calibration reference -------------------------------------
    result_img = _embed_calibration(result_img, border, box_size)

    # -- 9. Verify scan (convert to grayscale for standard QR reader) -------
    scan_results = []
    scan_ok = None
    if verify_scan:
        from qrx.verify import verify

        gray_for_scan = result_img.convert("L").convert("RGB")
        scan_results = verify(gray_for_scan, expected_data=data)
        scan_ok = any(r.success for r in scan_results)
        for sr in scan_results:
            audit(
                "chroma_qr.verify", logger=log,
                decoder=sr.decoder,
                success=sr.success,
                data=(sr.decoded_data or "")[:80],
                error=sr.error,
            )

    audit(
        "chroma_qr.generated", logger=log,
        data=data[:80],
        version=actual_version,
        delta_e_max=f"{delta_e_max:.2f}",
        scan_ok=scan_ok,
    )

    return {
        "image": result_img,
        "scan_ok": scan_ok,
        "delta_e_max": delta_e_max,
    }
