"""M2: Logo Processing — load logos, extract silhouettes, compute QR module overlaps."""

import numpy as np
from PIL import Image, ImageDraw

from qrx.logging import audit, get_logger, trace

log = get_logger("logo")


# ---------------------------------------------------------------------------
# Bezier utilities
# ---------------------------------------------------------------------------

def _cubic_bezier(p0, p1, p2, p3, n=30):
    """Generate *n+1* points along a cubic Bezier curve."""
    pts = []
    for i in range(n + 1):
        t = i / n
        u = 1 - t
        x = u**3 * p0[0] + 3 * u**2 * t * p1[0] + 3 * u * t**2 * p2[0] + t**3 * p3[0]
        y = u**3 * p0[1] + 3 * u**2 * t * p1[1] + 3 * u * t**2 * p2[1] + t**3 * p3[1]
        pts.append((x, y))
    return pts


# ---------------------------------------------------------------------------
# Built-in Apple logo silhouette
# ---------------------------------------------------------------------------

@trace
def create_apple_silhouette(size: int = 512) -> Image.Image:
    """Create a stylized Apple logo silhouette.

    Returns a grayscale PIL Image (mode 'L') where 255 = logo, 0 = background.
    Rendered at 4x and downscaled for anti-aliased edges.
    """
    # Render at 4x for smooth anti-aliasing
    rs = size * 4
    s = rs / 1000.0  # work in a 1000x1000 coordinate space

    img = Image.new("L", (rs, rs), 0)
    draw = ImageDraw.Draw(img)

    # --- Apple body (8 cubic-Bezier segments, clockwise from bottom) ---
    body_segments = [
        # bottom → lower-right
        ((500, 920), (600, 920), (700, 840), (740, 720)),
        # lower-right → right-mid
        ((740, 720), (780, 600), (790, 500), (780, 400)),
        # right-mid → upper-right
        ((780, 400), (770, 300), (730, 220), (670, 165)),
        # upper-right → cleft
        ((670, 165), (620, 125), (560, 140), (500, 175)),
        # cleft → upper-left
        ((500, 175), (440, 140), (380, 125), (330, 165)),
        # upper-left → left-mid
        ((330, 165), (270, 220), (230, 300), (220, 400)),
        # left-mid → lower-left
        ((220, 400), (210, 500), (220, 600), (260, 720)),
        # lower-left → bottom
        ((260, 720), (300, 840), (400, 920), (500, 920)),
    ]

    outline = []
    for p0, p1, p2, p3 in body_segments:
        pts = _cubic_bezier(p0, p1, p2, p3, n=30)
        outline.extend(pts[:-1])  # drop last (= next segment's first)

    scaled = [(int(x * s), int(y * s)) for x, y in outline]
    draw.polygon(scaled, fill=255)

    # --- Bite (circle cut from right side) ---
    bite_cx, bite_cy, bite_r = 765, 475, 95
    draw.ellipse(
        [int((bite_cx - bite_r) * s), int((bite_cy - bite_r) * s),
         int((bite_cx + bite_r) * s), int((bite_cy + bite_r) * s)],
        fill=0,
    )

    # --- Leaf (two Bezier arcs forming a closed leaf shape) ---
    leaf_top = _cubic_bezier((510, 160), (530, 100), (590, 55), (640, 40), n=20)
    leaf_bot = _cubic_bezier((640, 40), (590, 75), (540, 130), (510, 160), n=20)
    leaf_outline = leaf_top[:-1] + leaf_bot[:-1]
    leaf_scaled = [(int(x * s), int(y * s)) for x, y in leaf_outline]
    if len(leaf_scaled) >= 3:
        draw.polygon(leaf_scaled, fill=255)

    # --- Stem (thin line from cleft upward) ---
    stem_pts = _cubic_bezier((500, 175), (498, 140), (502, 110), (508, 85), n=15)
    stem_scaled = [(int(x * s), int(y * s)) for x, y in stem_pts]
    draw.line(stem_scaled, fill=255, width=int(8 * s))

    # Downscale with Lanczos for smooth edges
    result = img.resize((size, size), Image.LANCZOS)
    audit("logo.apple_created", logger=log, size=size)
    return result


# ---------------------------------------------------------------------------
# Logo loading & mask conversion
# ---------------------------------------------------------------------------

@trace
def load_logo(path: str) -> Image.Image:
    """Load a logo image and return a grayscale mask (255 = logo, 0 = bg).

    Supports PNG with alpha (preferred), or any format (auto-inverts if
    background is brighter than foreground).
    """
    img = Image.open(path)
    if img.mode == "RGBA":
        return img.split()[3]  # alpha channel
    if img.mode == "LA":
        return img.split()[1]
    # Fallback: convert to grayscale, auto-invert if needed
    gray = img.convert("L")
    arr = np.array(gray)
    if arr.mean() > 128:
        arr = 255 - arr
    return Image.fromarray(arr)


@trace
def logo_to_binary_mask(logo_image: Image.Image, threshold: int = 128) -> np.ndarray:
    """Convert a grayscale logo to a strict binary mask.

    Returns:
        numpy bool array where True = logo pixel.
    """
    arr = np.array(logo_image.convert("L"))
    return arr >= threshold


# ---------------------------------------------------------------------------
# Logo ↔ QR geometry helpers
# ---------------------------------------------------------------------------

@trace
def fit_logo_to_qr(logo_mask: np.ndarray, qr_pixel_size: int, coverage: float = 0.30) -> np.ndarray:
    """Scale and centre the logo mask onto a canvas matching the QR image.

    Args:
        logo_mask: bool array of logo silhouette.
        qr_pixel_size: Total pixel width/height of the QR image.
        coverage: Target fraction of QR width the logo should span.

    Returns:
        bool array of shape (qr_pixel_size, qr_pixel_size).
    """
    target_size = int(qr_pixel_size * coverage)

    logo_img = Image.fromarray(logo_mask.astype(np.uint8) * 255)
    lh, lw = logo_mask.shape
    aspect = lw / lh
    if aspect >= 1:
        new_w = target_size
        new_h = int(target_size / aspect)
    else:
        new_h = target_size
        new_w = int(target_size * aspect)

    logo_resized = logo_img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("L", (qr_pixel_size, qr_pixel_size), 0)
    x_off = (qr_pixel_size - new_w) // 2
    y_off = (qr_pixel_size - new_h) // 2
    canvas.paste(logo_resized, (x_off, y_off))

    return np.array(canvas) >= 128


@trace
def compute_module_coverage(
    logo_mask_on_qr: np.ndarray,
    qr_size: int,
    box_size: int,
    border: int,
) -> dict[tuple[int, int], float]:
    """Compute what fraction of each QR module is covered by the logo.

    Returns:
        dict mapping (row, col) -> coverage fraction [0.0, 1.0].
    """
    coverage: dict[tuple[int, int], float] = {}
    for r in range(qr_size):
        for c in range(qr_size):
            py = (r + border) * box_size
            px = (c + border) * box_size
            region = logo_mask_on_qr[py : py + box_size, px : px + box_size]
            if region.size > 0:
                coverage[(r, c)] = float(region.sum()) / region.size
            else:
                coverage[(r, c)] = 0.0
    return coverage


@trace
def classify_module_zones(
    coverage_map: dict[tuple[int, int], float],
    covered_threshold: float = 0.70,
    edge_threshold: float = 0.20,
) -> dict[str, set[tuple[int, int]]]:
    """Classify modules into zones based on logo coverage.

    Returns:
        dict with keys 'covered', 'edge', 'clear' — each a set of (r, c).
    """
    covered: set[tuple[int, int]] = set()
    edge: set[tuple[int, int]] = set()
    clear: set[tuple[int, int]] = set()

    for pos, cov in coverage_map.items():
        if cov >= covered_threshold:
            covered.add(pos)
        elif cov >= edge_threshold:
            edge.add(pos)
        else:
            clear.add(pos)

    audit(
        "logo.zones_classified", logger=log,
        covered=len(covered), edge=len(edge), clear=len(clear),
    )
    return {"covered": covered, "edge": edge, "clear": clear}
