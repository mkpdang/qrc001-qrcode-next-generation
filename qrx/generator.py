"""M1: QR Generation Engine — generate QR codes with full control over version, ECC, mask, and pad codewords."""

from enum import Enum

import qrcode
import qrcode.constants
from PIL import Image, ImageDraw

from qrx.logging import audit, get_logger, trace

log = get_logger("generator")


class ECCLevel(Enum):
    L = qrcode.constants.ERROR_CORRECT_L  # 7%
    M = qrcode.constants.ERROR_CORRECT_M  # 15%
    Q = qrcode.constants.ERROR_CORRECT_Q  # 25%
    H = qrcode.constants.ERROR_CORRECT_H  # 30%


ECC_NAMES = {"L": ECCLevel.L, "M": ECCLevel.M, "Q": ECCLevel.Q, "H": ECCLevel.H}


@trace
def generate_qr(
    data: str,
    version: int | None = None,
    ecc: str = "H",
    mask: int | None = None,
    box_size: int = 20,
    border: int = 4,
    custom_pad_bytes: bytes | None = None,
) -> Image.Image:
    """Generate a QR code image with full parameter control.

    Args:
        data: The string to encode (URL, text, etc.)
        version: QR version 1-40 (None = auto-detect minimum)
        ecc: Error correction level: L/M/Q/H
        mask: Mask pattern 0-7 (None = auto-select best)
        box_size: Pixel size of each module
        border: Quiet zone width in modules
        custom_pad_bytes: Custom pad codewords to replace default 0xEC/0x11 pattern.
                          Used to create module patterns aligned with a logo.

    Returns:
        PIL Image of the QR code.
    """
    ecc_level = ECC_NAMES[ecc.upper()]

    qr = qrcode.QRCode(
        version=version,
        error_correction=ecc_level.value,
        box_size=box_size,
        border=border,
        mask_pattern=mask,
    )
    qr.add_data(data)
    qr.make(fit=(version is None))

    if custom_pad_bytes is not None:
        _inject_custom_pad(qr, custom_pad_bytes)

    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")

    actual_version = qr.version
    size = actual_version * 4 + 17
    audit("qr.generated", logger=log,
          data=data[:80], version=actual_version, size=f"{size}x{size}",
          ecc=ecc.upper(), mask=mask if mask is not None else "auto",
          image_px=f"{img.size[0]}x{img.size[1]}")
    return img


def _inject_custom_pad(qr: qrcode.QRCode, custom_pad_bytes: bytes) -> None:
    """Inject custom pad codewords into the QR data before ECC calculation.

    Standard QR fills unused data capacity with alternating 0xEC/0x11.
    By replacing these with custom bytes, we can influence which modules
    are black/white in ways that better match a logo — at zero ECC cost
    since the ECC is calculated AFTER padding.
    """
    # Access the internal data buffer
    # The qrcode library stores data in qr.data_list entries
    # We need to hook into the make process at the right point
    # For now, this is a placeholder that logs what would be injected
    # Full implementation requires forking the qrcode library's _best_fit method
    log.debug("custom_pad_inject: placeholder (pad_bytes=%d)", len(custom_pad_bytes))


@trace
def generate_all_masks(
    data: str,
    version: int | None = None,
    ecc: str = "H",
    box_size: int = 20,
    border: int = 4,
) -> list[tuple[int, Image.Image]]:
    """Generate QR codes with all 8 mask patterns.

    Returns list of (mask_index, image) tuples.
    """
    results = []
    for mask_idx in range(8):
        img = generate_qr(
            data=data,
            version=version,
            ecc=ecc,
            mask=mask_idx,
            box_size=box_size,
            border=border,
        )
        results.append((mask_idx, img))
    audit("qr.all_masks_generated", logger=log, data=data[:80], count=len(results), ecc=ecc.upper())
    return results


@trace
def get_module_matrix(
    data: str,
    version: int | None = None,
    ecc: str = "H",
    mask: int | None = None,
) -> list[list[bool]]:
    """Get the raw module matrix (True=black, False=white) without rendering.

    Useful for analyzing which modules are data vs ECC vs fixed patterns.
    """
    ecc_level = ECC_NAMES[ecc.upper()]
    qr = qrcode.QRCode(
        version=version,
        error_correction=ecc_level.value,
        box_size=1,
        border=0,
        mask_pattern=mask,
    )
    qr.add_data(data)
    qr.make(fit=(version is None))
    return qr.modules


@trace
def get_module_map(
    data: str,
    version: int | None = None,
    ecc: str = "H",
    mask: int | None = None,
) -> dict:
    """Analyze the QR code structure: which modules are what type.

    Returns a dict with:
        - 'version': actual version used
        - 'size': grid dimension (e.g., 25 for V2)
        - 'modules': the bool matrix
        - 'finder_positions': list of (row, col) for finder pattern modules
        - 'alignment_positions': list of (row, col) for alignment pattern modules
        - 'timing_positions': list of (row, col) for timing pattern modules
        - 'format_positions': list of (row, col) for format info modules
        - 'data_positions': list of (row, col) for data+ECC modules
    """
    ecc_level = ECC_NAMES[ecc.upper()]
    qr = qrcode.QRCode(
        version=version,
        error_correction=ecc_level.value,
        box_size=1,
        border=0,
        mask_pattern=mask,
    )
    qr.add_data(data)
    qr.make(fit=(version is None))

    actual_version = qr.version
    size = actual_version * 4 + 17

    finder_pos = _map_finder_positions(size)
    timing_pos = _map_timing_positions(size)
    alignment_pos = _map_alignment_positions(actual_version, set(finder_pos))
    format_pos = _map_format_positions(size)

    fixed = set(finder_pos + alignment_pos + timing_pos + format_pos)
    data_pos = [
        (r, c)
        for r in range(size)
        for c in range(size)
        if (r, c) not in fixed
    ]

    audit("qr.module_map", logger=log,
          version=actual_version, size=f"{size}x{size}",
          finder=len(finder_pos), alignment=len(alignment_pos),
          timing=len(timing_pos), format=len(format_pos),
          data_ecc=len(data_pos))

    return {
        "version": actual_version,
        "size": size,
        "modules": qr.modules,
        "finder_positions": finder_pos,
        "alignment_positions": alignment_pos,
        "timing_positions": timing_pos,
        "format_positions": format_pos,
        "data_positions": data_pos,
    }


def _map_finder_positions(size: int) -> list[tuple[int, int]]:
    """Map all finder pattern modules including separators."""
    positions = []
    # 7x7 finder grids: top-left, top-right, bottom-left
    for r in range(7):
        for c in range(7):
            positions.append((r, c))
            positions.append((r, size - 7 + c))
            positions.append((size - 7 + r, c))

    # 1-module white separators around finders
    finder_set = set(positions)
    for i in range(8):
        candidates = [
            (7, i), (i, 7),
            (7, size - 8 + i), (i, size - 8),
            (size - 8, i),
        ]
        if i < 7:
            candidates.append((size - 8 + i, 7))
        for r, c in candidates:
            if 0 <= r < size and 0 <= c < size and (r, c) not in finder_set:
                positions.append((r, c))
                finder_set.add((r, c))
    return positions


def _map_timing_positions(size: int) -> list[tuple[int, int]]:
    """Map timing pattern modules (row 6 and column 6)."""
    positions = []
    for i in range(8, size - 8):
        positions.append((6, i))
        positions.append((i, 6))
    return positions


def _map_alignment_positions(
    version: int,
    finder_set: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Map alignment pattern modules, skipping finder overlaps."""
    if version < 2:
        return []
    positions = []
    for ar, ac in _get_alignment_centers(version):
        block = [
            (r, c)
            for r in range(ar - 2, ar + 3)
            for c in range(ac - 2, ac + 3)
        ]
        if not any(pos in finder_set for pos in block):
            positions.extend(block)
    return positions


def _map_format_positions(size: int) -> list[tuple[int, int]]:
    """Map format information modules and the dark module."""
    seen: set[tuple[int, int]] = set()
    positions: list[tuple[int, int]] = []

    def _add(r: int, c: int) -> None:
        if (r, c) not in seen:
            seen.add((r, c))
            positions.append((r, c))

    # Top-left format info (row 8 and column 8, skipping timing at index 6)
    for i in range(9):
        if i != 6:
            _add(8, i)
            _add(i, 8)
    # Top-right format info (row 8, rightmost 8 columns)
    for i in range(8):
        _add(8, size - 8 + i)
    # Bottom-left format info (column 8, bottom 7 rows)
    for i in range(7):
        _add(size - 7 + i, 8)
    # Dark module (always black, required by spec)
    _add(size - 8, 8)
    return positions


def _get_alignment_centers(version: int) -> list[tuple[int, int]]:
    """Get alignment pattern center coordinates for a given QR version."""
    # Alignment pattern position table (from QR spec)
    _ALIGNMENT_TABLE = {
        2: [6, 18], 3: [6, 22], 4: [6, 26], 5: [6, 30], 6: [6, 34],
        7: [6, 22, 38], 8: [6, 24, 42], 9: [6, 26, 46], 10: [6, 28, 50],
        11: [6, 30, 54], 12: [6, 32, 58], 13: [6, 34, 62], 14: [6, 26, 46, 66],
        15: [6, 26, 48, 70], 16: [6, 26, 50, 74], 17: [6, 30, 54, 78],
        18: [6, 30, 56, 82], 19: [6, 30, 58, 86], 20: [6, 34, 62, 90],
    }
    coords = _ALIGNMENT_TABLE.get(version, [])
    return [(r, c) for r in coords for c in coords]


@trace
def render_bitmap_dump(module_map: dict, output_path: str | None = None) -> Image.Image:
    """Render a color-coded bitmap showing module types.

    Colors:
        - Red: Finder patterns
        - Blue: Alignment patterns
        - Green: Timing patterns
        - Yellow: Format information
        - Black/White: Data modules (actual value)
        - Gray: Data modules marked "safe to destroy" (center zone)
    """
    size = module_map["size"]
    scale = 20
    img = Image.new("RGB", (size * scale, size * scale), (255, 255, 255))

    finder_set = set(map(tuple, module_map["finder_positions"]))
    alignment_set = set(map(tuple, module_map["alignment_positions"]))
    timing_set = set(map(tuple, module_map["timing_positions"]))
    format_set = set(map(tuple, module_map["format_positions"]))
    modules = module_map["modules"]

    # Center safe zone (for logo) — center 30% of grid
    center = size // 2
    safe_radius = int(size * 0.15)  # ~30% diameter = ~15% radius
    draw = ImageDraw.Draw(img)

    for r in range(size):
        for c in range(size):
            x0, y0 = c * scale, r * scale
            x1, y1 = x0 + scale - 1, y0 + scale - 1
            pos = (r, c)

            if pos in finder_set:
                color = (220, 50, 50) if modules[r][c] else (255, 180, 180)
            elif pos in alignment_set:
                color = (50, 50, 220) if modules[r][c] else (180, 180, 255)
            elif pos in timing_set:
                color = (50, 180, 50) if modules[r][c] else (180, 255, 180)
            elif pos in format_set:
                color = (220, 200, 50) if modules[r][c] else (255, 240, 180)
            else:
                # Data module
                in_safe_zone = abs(r - center) <= safe_radius and abs(c - center) <= safe_radius
                if in_safe_zone:
                    color = (160, 160, 160) if modules[r][c] else (220, 220, 220)
                else:
                    color = (0, 0, 0) if modules[r][c] else (255, 255, 255)

            draw.rectangle([x0, y0, x1, y1], fill=color)
            # Grid lines
            draw.rectangle([x0, y0, x1, y1], outline=(230, 230, 230))

    if output_path:
        img.save(output_path)
        audit("bitmap.saved", logger=log, path=output_path, size=f"{size}x{size}")
    return img
