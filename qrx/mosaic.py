"""M5: Non-Square & Mosaic Codes — arrange multiple QR tiles into composite layouts.

Implements:
    5.1  Structured Append — linked multi-QR mosaics via segno
    5.2  Parallel redundancy — identical micro QR tiles for fault tolerance
    5.3  Per-tile contour — rasterized logo overlay on each tile
    Layout engine — horizontal, vertical, grid, cross, L-shape, T-shape, custom
"""

import io

import numpy as np
import segno
from PIL import Image

from qrx.logging import audit, get_logger, trace

log = get_logger("mosaic")

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

LAYOUTS = {
    "horizontal": "tiles in a row",
    "vertical": "tiles in a column",
    "grid": "NxM grid",
    "cross": "plus shape",
    "l-shape": "L-shaped arrangement",
    "t-shape": "T-shaped arrangement",
}


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

@trace
def compute_layout_positions(
    layout: str,
    n_tiles: int,
    tile_size: int,
    gap: int = 10,
    custom_positions: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    """Compute (x, y) pixel offsets for each tile in the given layout.

    Args:
        layout: One of 'horizontal', 'vertical', 'grid', 'cross',
                'l-shape', 't-shape', or 'custom'.
        n_tiles: Number of tiles to place.
        tile_size: Pixel dimension of each square tile.
        gap: Pixel gap between tiles.
        custom_positions: Grid-cell (col, row) positions for 'custom' layout.
                          Each entry is a (grid_col, grid_row) pair.

    Returns:
        List of (x, y) pixel offsets, one per tile.
    """
    stride = tile_size + gap

    if layout == "horizontal":
        return [(i * stride, 0) for i in range(n_tiles)]

    if layout == "vertical":
        return [(0, i * stride) for i in range(n_tiles)]

    if layout == "grid":
        cols = int(np.ceil(np.sqrt(n_tiles)))
        positions = []
        for idx in range(n_tiles):
            r = idx // cols
            c = idx % cols
            positions.append((c * stride, r * stride))
        return positions

    if layout == "cross":
        # Centre tile + arms extending in 4 directions
        positions = [(stride, stride)]  # centre
        arms = [
            (stride, 0),          # top
            (0, stride),          # left
            (2 * stride, stride), # right
            (stride, 2 * stride), # bottom
        ]
        for pos in arms:
            if len(positions) >= n_tiles:
                break
            positions.append(pos)
        return positions[:n_tiles]

    if layout == "l-shape":
        # L-shape: vertical column + horizontal extension at bottom
        positions = []
        # Vertical arm (left column)
        vert_count = max(2, (n_tiles + 1) // 2)
        for i in range(min(vert_count, n_tiles)):
            positions.append((0, i * stride))
        # Horizontal arm at bottom row
        bottom_y = (vert_count - 1) * stride
        for j in range(1, n_tiles - len(positions) + 1):
            if len(positions) >= n_tiles:
                break
            positions.append((j * stride, bottom_y))
        return positions[:n_tiles]

    if layout == "t-shape":
        # T-shape: top horizontal row + vertical column from centre
        positions = []
        # Determine top row width
        top_count = max(2, (n_tiles + 1) // 2)
        for i in range(min(top_count, n_tiles)):
            positions.append((i * stride, 0))
        # Vertical stem from the centre of the top row
        centre_x = (top_count // 2) * stride
        for j in range(1, n_tiles - len(positions) + 1):
            if len(positions) >= n_tiles:
                break
            positions.append((centre_x, j * stride))
        return positions[:n_tiles]

    if layout == "custom":
        if custom_positions is None:
            raise ValueError("custom_positions required for layout='custom'")
        return [(gc * stride, gr * stride) for gc, gr in custom_positions[:n_tiles]]

    raise ValueError(f"Unknown layout: {layout!r}. Choose from {list(LAYOUTS)}")


def _canvas_size(positions: list[tuple[int, int]], tile_size: int) -> tuple[int, int]:
    """Compute the minimum canvas size to hold all tiles."""
    if not positions:
        return (0, 0)
    max_x = max(x for x, _ in positions) + tile_size
    max_y = max(y for _, y in positions) + tile_size
    return (max_x, max_y)


# ---------------------------------------------------------------------------
# 5.1  Structured Append via segno
# ---------------------------------------------------------------------------

@trace
def generate_structured_append(
    data: str,
    *,
    symbol_count: int = 3,
    ecc: str = "H",
    scale: int = 10,
    border: int = 2,
) -> list[Image.Image]:
    """Generate a structured-append QR sequence using segno.

    Structured Append splits data across multiple linked QR symbols.
    Each symbol includes a header (mode 0011, symbol index, total count,
    parity) so scanners can reassemble the full message.

    Args:
        data: The data to split across symbols.
        symbol_count: Target number of symbols.
        ecc: Error correction level (L/M/Q/H).
        scale: Pixel scale per module.
        border: Quiet zone in modules.

    Returns:
        List of PIL Images, one per structured-append symbol.
    """
    ecc_map = {"L": "L", "M": "M", "Q": "Q", "H": "H"}
    qrs = segno.make_sequence(data, symbol_count=symbol_count,
                              error=ecc_map.get(ecc.upper(), "H"))

    images = []
    for i, qr in enumerate(qrs):
        buf = io.BytesIO()
        qr.save(buf, kind="png", scale=scale, border=border)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        images.append(img)
        audit(
            "mosaic.structured_segment", logger=log,
            segment=i,
            total=len(qrs),
            version=qr.version,
            size=f"{img.size[0]}x{img.size[1]}",
        )

    audit(
        "mosaic.structured_append", logger=log,
        data=data[:80],
        symbols=len(images),
        ecc=ecc.upper(),
    )
    return images


# ---------------------------------------------------------------------------
# Single tile generator
# ---------------------------------------------------------------------------

@trace
def generate_single_tile(
    data: str,
    *,
    version: int | None = None,
    ecc: str = "H",
    box_size: int = 20,
    border: int = 2,
    shape: str = "circle",
    data_color: tuple[int, ...] = (0, 0, 0),
    bg_color: tuple[int, ...] = (255, 255, 255),
    finder_color: tuple[int, ...] | None = None,
    finder_style: str = "rounded",
    logo: str | Image.Image | None = None,
    use_apple: bool = False,
    logo_color: tuple[int, ...] = (0, 0, 0),
    contour_softness: float = 1.0,
) -> dict:
    """Generate one small QR tile, optionally with per-tile logo contour.

    Wraps generate_logo_qr with a smaller default border suitable for tiling.
    When a logo is provided, each tile gets its own contour treatment (5.3).

    Returns:
        dict matching generate_logo_qr output (image, scan_ok, etc.).
    """
    from qrx.logo_qr import generate_logo_qr

    result = generate_logo_qr(
        data,
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
        contour_softness=contour_softness,
        verify_scan=False,  # we verify tiles after compositing
    )

    audit(
        "mosaic.tile_generated", logger=log,
        data=data[:80],
        tile_px=f"{result['image'].size[0]}x{result['image'].size[1]}",
    )
    return result


# ---------------------------------------------------------------------------
# Main mosaic generator
# ---------------------------------------------------------------------------

@trace
def generate_mosaic_qr(
    data: str,
    *,
    layout: str = "horizontal",
    tiles: int = 3,
    logo: str | Image.Image | None = None,
    use_apple: bool = False,
    version: int | None = None,
    ecc: str = "H",
    box_size: int = 20,
    border: int = 2,
    gap: int = 10,
    shape: str = "circle",
    data_color: tuple[int, ...] = (0, 0, 0),
    bg_color: tuple[int, ...] = (255, 255, 255),
    finder_color: tuple[int, ...] | None = None,
    finder_style: str = "rounded",
    logo_color: tuple[int, ...] = (0, 0, 0),
    contour_softness: float = 1.0,
    custom_positions: list[tuple[int, int]] | None = None,
    mode: str = "redundant",
) -> dict:
    """Generate a mosaic of QR tiles arranged in a layout.

    Supports two modes:
    - "redundant" (default): Each tile encodes the same data (parallel
      redundancy). Succeeds if any single tile scans correctly.
    - "structured": Uses segno structured append to split data across
      tiles. All tiles together reconstruct the full message.

    Args:
        data: URL or text to encode.
        layout: Layout name or 'custom' with custom_positions.
        tiles: Number of tiles.
        logo: Path or PIL Image for per-tile logo overlay (5.3).
        use_apple: Use built-in Apple silhouette.
        version: QR version (None = auto).
        ecc: Error correction level.
        box_size: Module pixel size.
        border: Quiet zone in modules (smaller for tiling).
        gap: Pixel gap between tiles.
        shape: Data module shape.
        data_color: Module colour.
        bg_color: Background colour.
        finder_color: Finder pattern colour.
        finder_style: Finder pattern style.
        logo_color: Logo fill colour.
        contour_softness: Contour falloff.
        custom_positions: Grid-cell positions for 'custom' layout.
        mode: 'redundant' or 'structured'.

    Returns:
        dict with: image, tiles_count, layout, mode,
                   per_tile_scan_results, any_scan_ok, tile_results.
    """
    if mode == "structured":
        return _generate_structured_mosaic(
            data, layout=layout, tiles=tiles, ecc=ecc,
            box_size=box_size, border=border, gap=gap,
            bg_color=bg_color, custom_positions=custom_positions,
        )

    return _generate_redundant_mosaic(
        data, layout=layout, tiles=tiles,
        logo=logo, use_apple=use_apple,
        version=version, ecc=ecc,
        box_size=box_size, border=border, gap=gap,
        shape=shape, data_color=data_color, bg_color=bg_color,
        finder_color=finder_color, finder_style=finder_style,
        logo_color=logo_color, contour_softness=contour_softness,
        custom_positions=custom_positions,
    )


def _generate_structured_mosaic(
    data: str,
    *,
    layout: str,
    tiles: int,
    ecc: str,
    box_size: int,
    border: int,
    gap: int,
    bg_color: tuple[int, ...],
    custom_positions: list[tuple[int, int]] | None,
) -> dict:
    """Internal: structured-append mosaic."""
    tile_images = generate_structured_append(
        data, symbol_count=tiles, ecc=ecc, scale=box_size, border=border,
    )
    actual_tiles = len(tile_images)
    tile_size = tile_images[0].size[0]

    positions = compute_layout_positions(
        layout, actual_tiles, tile_size, gap,
        custom_positions=custom_positions,
    )
    canvas_w, canvas_h = _canvas_size(positions, tile_size)

    canvas = Image.new("RGB", (canvas_w, canvas_h), bg_color)
    for pos, img in zip(positions, tile_images):
        canvas.paste(img, pos)

    # Verify each segment independently
    from qrx.verify import verify

    per_tile_scans = []
    any_scan_ok = False
    for i, pos in enumerate(positions):
        x, y = pos
        tile_crop = canvas.crop((x, y, x + tile_size, y + tile_size))
        scan_results = verify(tile_crop)
        tile_ok = any(r.success for r in scan_results)
        if tile_ok:
            any_scan_ok = True
        per_tile_scans.append({
            "tile_index": i,
            "scan_results": scan_results,
            "scan_ok": tile_ok,
        })
        for sr in scan_results:
            audit(
                "mosaic.tile_verify", logger=log,
                tile=i, decoder=sr.decoder,
                success=sr.success,
                data=(sr.decoded_data or "")[:80],
                error=sr.error,
            )

    audit(
        "mosaic.structured_mosaic_generated", logger=log,
        data=data[:80], layout=layout, tiles=actual_tiles,
        canvas_size=f"{canvas_w}x{canvas_h}",
        any_scan_ok=any_scan_ok,
    )

    return {
        "image": canvas,
        "tiles_count": actual_tiles,
        "layout": layout,
        "mode": "structured",
        "per_tile_scan_results": per_tile_scans,
        "any_scan_ok": any_scan_ok,
        "tile_results": [{"image": img} for img in tile_images],
    }


def _generate_redundant_mosaic(
    data: str,
    *,
    layout: str,
    tiles: int,
    logo: str | Image.Image | None,
    use_apple: bool,
    version: int | None,
    ecc: str,
    box_size: int,
    border: int,
    gap: int,
    shape: str,
    data_color: tuple[int, ...],
    bg_color: tuple[int, ...],
    finder_color: tuple[int, ...] | None,
    finder_style: str,
    logo_color: tuple[int, ...],
    contour_softness: float,
    custom_positions: list[tuple[int, int]] | None,
) -> dict:
    """Internal: parallel-redundancy mosaic."""
    has_logo = use_apple or logo is not None

    # Generate tiles (all identical data for redundancy)
    tile_results = []
    for i in range(tiles):
        if has_logo:
            tr = generate_single_tile(
                data,
                version=version,
                ecc=ecc,
                box_size=box_size,
                border=border,
                shape=shape,
                data_color=data_color,
                bg_color=bg_color,
                finder_color=finder_color,
                finder_style=finder_style,
                logo=logo,
                use_apple=use_apple,
                logo_color=logo_color,
                contour_softness=contour_softness,
            )
        else:
            from qrx.generator import get_module_map
            from qrx.logo_qr import render_styled_qr

            mmap = get_module_map(data=data, version=version, ecc=ecc)
            img = render_styled_qr(
                mmap,
                box_size=box_size,
                border=border,
                shape=shape,
                data_color=data_color,
                bg_color=bg_color,
                finder_color=finder_color,
                finder_style=finder_style,
            )
            tr = {"image": img, "scan_ok": None}

        tile_results.append(tr)

    tile_size = tile_results[0]["image"].size[0]

    # Compute layout positions
    positions = compute_layout_positions(
        layout, tiles, tile_size, gap,
        custom_positions=custom_positions,
    )
    canvas_w, canvas_h = _canvas_size(positions, tile_size)

    # Composite tiles onto canvas
    canvas = Image.new("RGB", (canvas_w, canvas_h), bg_color)
    for pos, tr in zip(positions, tile_results):
        canvas.paste(tr["image"], pos)

    # Verify each tile independently by cropping from the composite
    from qrx.verify import verify

    per_tile_scans = []
    any_scan_ok = False
    for i, pos in enumerate(positions):
        x, y = pos
        tile_crop = canvas.crop((x, y, x + tile_size, y + tile_size))
        scan_results = verify(tile_crop, expected_data=data)
        tile_ok = any(r.success for r in scan_results)
        if tile_ok:
            any_scan_ok = True
        per_tile_scans.append({
            "tile_index": i,
            "scan_results": scan_results,
            "scan_ok": tile_ok,
        })
        for sr in scan_results:
            audit(
                "mosaic.tile_verify", logger=log,
                tile=i,
                decoder=sr.decoder,
                success=sr.success,
                data=(sr.decoded_data or "")[:80],
                error=sr.error,
            )

    audit(
        "mosaic.generated", logger=log,
        data=data[:80],
        layout=layout,
        tiles=tiles,
        gap=gap,
        canvas_size=f"{canvas_w}x{canvas_h}",
        any_scan_ok=any_scan_ok,
    )

    return {
        "image": canvas,
        "tiles_count": tiles,
        "layout": layout,
        "mode": "redundant",
        "per_tile_scan_results": per_tile_scans,
        "any_scan_ok": any_scan_ok,
        "tile_results": tile_results,
    }


# ---------------------------------------------------------------------------
# 5.2  Explicit parallel redundancy entry point
# ---------------------------------------------------------------------------

@trace
def generate_redundant_mosaic(
    data: str,
    *,
    logo: str | Image.Image | None = None,
    use_apple: bool = False,
    tiles: int = 3,
    layout: str = "horizontal",
    version: int | None = None,
    ecc: str = "H",
    box_size: int = 20,
    border: int = 2,
    gap: int = 10,
    shape: str = "circle",
    data_color: tuple[int, ...] = (0, 0, 0),
    bg_color: tuple[int, ...] = (255, 255, 255),
    finder_color: tuple[int, ...] | None = None,
    finder_style: str = "rounded",
    logo_color: tuple[int, ...] = (0, 0, 0),
    contour_softness: float = 1.0,
    custom_positions: list[tuple[int, int]] | None = None,
) -> dict:
    """Generate a parallel-redundancy mosaic of identical QR tiles.

    Convenience wrapper that always uses redundant mode. Each tile
    encodes the same URL — any single tile scanning equals success.

    Returns:
        dict with: image, tiles_count, layout, mode='redundant',
                   per_tile_scan_results, any_scan_ok.
    """
    return generate_mosaic_qr(
        data,
        layout=layout,
        tiles=tiles,
        logo=logo,
        use_apple=use_apple,
        version=version,
        ecc=ecc,
        box_size=box_size,
        border=border,
        gap=gap,
        shape=shape,
        data_color=data_color,
        bg_color=bg_color,
        finder_color=finder_color,
        finder_style=finder_style,
        logo_color=logo_color,
        contour_softness=contour_softness,
        custom_positions=custom_positions,
        mode="redundant",
    )


# ---------------------------------------------------------------------------
# Shaped mosaic (advanced layout engine)
# ---------------------------------------------------------------------------

@trace
def generate_shaped_mosaic(
    data: str,
    *,
    logo_mask: np.ndarray,
    tiles: int = 3,
    version: int | None = None,
    ecc: str = "H",
    box_size: int = 20,
    border: int = 2,
    gap: int = 10,
    shape: str = "circle",
    data_color: tuple[int, ...] = (0, 0, 0),
    bg_color: tuple[int, ...] = (255, 255, 255),
    finder_color: tuple[int, ...] | None = None,
    finder_style: str = "rounded",
) -> dict:
    """Generate a mosaic where tiles fill a logo shape.

    Given a binary logo mask, compute optimal tile placement so tiles
    cover regions where the logo has the most density. Areas outside
    the shape are left transparent, giving a non-square overall appearance.

    Args:
        data: URL or text to encode.
        logo_mask: Bool ndarray defining the target shape.
        tiles: Number of tiles to place.
        version: QR version.
        ecc: Error correction level.
        box_size: Module pixel size.
        border: Quiet zone modules.
        gap: Pixel gap between tiles.
        shape: Data module shape.
        data_color: Module colour.
        bg_color: Background colour.
        finder_color: Finder colour.
        finder_style: Finder style.

    Returns:
        dict with: image (RGBA with transparency outside shape),
                   tiles_count, positions, per_tile_scan_results, any_scan_ok.
    """
    from qrx.generator import get_module_map
    from qrx.logo_qr import render_styled_qr

    # Generate a reference tile to get its size
    mmap = get_module_map(data=data, version=version, ecc=ecc)
    ref_img = render_styled_qr(
        mmap,
        box_size=box_size,
        border=border,
        shape=shape,
        data_color=data_color,
        bg_color=bg_color,
        finder_color=finder_color,
        finder_style=finder_style,
    )
    tile_size = ref_img.size[0]

    # Scale the logo mask to a grid of potential tile positions
    mask_h, mask_w = logo_mask.shape
    stride = tile_size + gap
    grid_cols = max(1, mask_w // stride)
    grid_rows = max(1, mask_h // stride)

    # Score each grid cell by logo mask density
    cell_scores = []
    for gr in range(grid_rows):
        for gc in range(grid_cols):
            y0 = int(gr * mask_h / grid_rows)
            y1 = int((gr + 1) * mask_h / grid_rows)
            x0 = int(gc * mask_w / grid_cols)
            x1 = int((gc + 1) * mask_w / grid_cols)
            region = logo_mask[y0:y1, x0:x1]
            density = float(region.sum()) / max(1, region.size)
            cell_scores.append((density, gc, gr))

    # Pick top-N densest cells for tile placement
    cell_scores.sort(reverse=True)
    selected = cell_scores[:tiles]

    positions = [(gc * stride, gr * stride) for _, gc, gr in selected]
    canvas_w, canvas_h = _canvas_size(positions, tile_size)
    canvas_w = max(canvas_w, mask_w)
    canvas_h = max(canvas_h, mask_h)

    # Composite tiles
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 0))
    for pos in positions:
        tile_img = ref_img.convert("RGBA")
        canvas.paste(tile_img, pos, tile_img)

    # Apply shape mask — areas outside the logo become transparent
    shape_mask_resized = Image.fromarray(
        (logo_mask.astype(np.uint8) * 255)
    ).resize((canvas_w, canvas_h), Image.LANCZOS)
    shape_arr = np.array(shape_mask_resized)

    canvas_arr = np.array(canvas)
    canvas_arr[..., 3] = np.where(shape_arr > 64, canvas_arr[..., 3], 0)
    canvas = Image.fromarray(canvas_arr)

    # Verify tiles
    from qrx.verify import verify

    per_tile_scans = []
    any_scan_ok = False
    for i, pos in enumerate(positions):
        x, y = pos
        x1 = min(x + tile_size, canvas_w)
        y1 = min(y + tile_size, canvas_h)
        tile_crop = canvas.crop((x, y, x1, y1)).convert("RGB")
        scan_results = verify(tile_crop, expected_data=data)
        tile_ok = any(r.success for r in scan_results)
        if tile_ok:
            any_scan_ok = True
        per_tile_scans.append({
            "tile_index": i,
            "scan_results": scan_results,
            "scan_ok": tile_ok,
        })

    audit(
        "mosaic.shaped_generated", logger=log,
        data=data[:80],
        tiles=len(positions),
        canvas_size=f"{canvas_w}x{canvas_h}",
        any_scan_ok=any_scan_ok,
    )

    return {
        "image": canvas,
        "tiles_count": len(positions),
        "positions": positions,
        "per_tile_scan_results": per_tile_scans,
        "any_scan_ok": any_scan_ok,
    }
