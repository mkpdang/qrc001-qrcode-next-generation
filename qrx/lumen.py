"""M7: Lumen-Code — The Logo IS the Code.

Encodes data into logo contour geometry with no visible QR grid.
Two encoding methods:
  - polar: radial displacement of contour points
  - stroke: stroke-width modulation along the contour
"""

import numpy as np
from PIL import Image
from scipy import ndimage

from qrx.logging import audit, get_logger, trace

log = get_logger("lumen")

# Magic byte identifying a Lumen-Code bit stream (0x1C chosen for "Lumen Code")
_MAGIC = 0x1C


# ---------------------------------------------------------------------------
# Bit-level helpers
# ---------------------------------------------------------------------------

def _string_to_bits(data: str) -> list[int]:
    """Convert a UTF-8 string to a flat list of bits (MSB first per byte)."""
    bits = []
    for byte in data.encode("utf-8"):
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def _bits_to_string(bits: list[int]) -> str:
    """Convert a flat list of bits back to a UTF-8 string."""
    if len(bits) % 8 != 0:
        bits = bits[: len(bits) - len(bits) % 8]
    chars = []
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bits[i + j]
        chars.append(byte)
    return bytes(chars).decode("utf-8", errors="replace")


def _compute_crc8(bits: list[int]) -> list[int]:
    """Compute an 8-bit CRC over a bit list. Returns 8 bits."""
    crc = 0x00
    for bit in bits:
        crc ^= bit << 7
        if crc & 0x80:
            crc = ((crc << 1) ^ 0x07) & 0xFF
        else:
            crc = (crc << 1) & 0xFF
    result = []
    for i in range(7, -1, -1):
        result.append((crc >> i) & 1)
    return result


def _int_to_bits(value: int, width: int) -> list[int]:
    """Convert an integer to a fixed-width bit list (MSB first)."""
    bits = []
    for i in range(width - 1, -1, -1):
        bits.append((value >> i) & 1)
    return bits


def _bits_to_int(bits: list[int]) -> int:
    """Convert a bit list (MSB first) to an integer."""
    val = 0
    for b in bits:
        val = (val << 1) | b
    return val


# ---------------------------------------------------------------------------
# Contour extraction
# ---------------------------------------------------------------------------

@trace
def _extract_contour(mask: np.ndarray) -> np.ndarray:
    """Extract ordered contour points from a binary mask.

    Uses binary erosion to find edge pixels, then sorts them by angle
    from the mask centroid (polar sort).

    Returns:
        Nx2 array of (y, x) contour points, ordered by angle.
    """
    bool_mask = mask.astype(bool)
    eroded = ndimage.binary_erosion(bool_mask, iterations=1)
    edge = bool_mask ^ eroded

    ys, xs = np.where(edge)
    if len(ys) == 0:
        return np.empty((0, 2), dtype=np.float64)

    # Compute centroid of the full mask
    cy, cx = ndimage.center_of_mass(bool_mask)

    # Sort by angle from centroid
    angles = np.arctan2(ys - cy, xs - cx)
    order = np.argsort(angles)

    return np.stack([ys[order], xs[order]], axis=1).astype(np.float64)


# ---------------------------------------------------------------------------
# Radial ray helpers (robust encode/decode)
# ---------------------------------------------------------------------------

def _measure_radial_extent(mask_bool: np.ndarray, cy: float, cx: float,
                           angle: float, max_radius: float) -> float:
    """Measure the farthest set pixel along a radial ray from (cy, cx).

    Returns the distance from centroid to the farthest mask pixel on this ray.
    """
    h, w = mask_bool.shape
    best_dist = 0.0
    steps = int(max_radius * 1.5)
    uy = np.sin(angle)
    ux = np.cos(angle)

    for step in range(steps):
        r = step * max_radius / steps
        y = int(round(cy + uy * r))
        x = int(round(cx + ux * r))
        if 0 <= y < h and 0 <= x < w:
            if mask_bool[y, x]:
                best_dist = r
        else:
            break
    return best_dist


# ---------------------------------------------------------------------------
# Build / parse bit stream
# ---------------------------------------------------------------------------

def _build_bitstream(data: str) -> list[int]:
    """Build the complete bit stream: magic(8) + length(16) + data + CRC8."""
    data_bits = _string_to_bits(data)
    length = len(data_bits)

    header = _int_to_bits(_MAGIC, 8) + _int_to_bits(length, 16)
    payload = header + data_bits
    crc = _compute_crc8(payload)
    return payload + crc


def _parse_bitstream(bits: list[int]) -> str | None:
    """Parse a bit stream. Returns decoded string or None on failure."""
    if len(bits) < 32:  # 8 magic + 16 length + 8 crc minimum
        return None

    magic = _bits_to_int(bits[:8])
    if magic != _MAGIC:
        return None

    length = _bits_to_int(bits[8:24])
    total_needed = 24 + length + 8  # header + data + crc
    if len(bits) < total_needed:
        return None

    payload = bits[:24 + length]
    crc_received = bits[24 + length: 24 + length + 8]
    crc_computed = _compute_crc8(payload)

    if crc_received != crc_computed:
        return None

    data_bits = bits[24:24 + length]
    return _bits_to_string(data_bits)


# ---------------------------------------------------------------------------
# Polar displacement encoding
# ---------------------------------------------------------------------------

def _assign_contour_to_sectors(contour: np.ndarray, cy: float, cx: float,
                                n_sectors: int) -> list[np.ndarray]:
    """Assign contour points to angular sectors.

    Divides [-pi, pi) into n_sectors equal slices and bins contour points.
    Returns list of n_sectors arrays, each containing points in that sector.
    """
    angles = np.arctan2(contour[:, 0] - cy, contour[:, 1] - cx)
    sector_width = 2 * np.pi / n_sectors
    # Map angle from [-pi, pi) to sector index [0, n_sectors)
    sector_indices = ((angles + np.pi) / sector_width).astype(int)
    sector_indices = np.clip(sector_indices, 0, n_sectors - 1)

    sectors = []
    for i in range(n_sectors):
        mask = sector_indices == i
        sectors.append(contour[mask])
    return sectors


@trace
def encode_polar(logo_mask: np.ndarray, data: str, strength: float = 2.0) -> np.ndarray:
    """Encode data via polar displacement of logo contour points.

    Divides the contour into angular sectors (one per bit) and displaces
    each sector radially: outward for bit=1, inward for bit=0.

    Returns a new binary mask (uint8, 0/255) with the displaced contour.
    """
    bool_mask = logo_mask.astype(bool)
    contour = _extract_contour(bool_mask)

    if len(contour) == 0:
        audit("lumen.encode_polar_fail", logger=log, reason="empty_contour")
        return bool_mask.astype(np.uint8) * 255

    bitstream = _build_bitstream(data)
    n_bits = len(bitstream)

    audit("lumen.encode_polar", logger=log, data_len=len(data),
          bits=n_bits, contour_pts=len(contour))

    cy, cx = ndimage.center_of_mass(bool_mask)
    sectors = _assign_contour_to_sectors(contour, cy, cx, n_bits)

    h, w = bool_mask.shape
    new_mask = bool_mask.copy().astype(np.uint8) * 255

    for bit_idx in range(n_bits):
        segment = sectors[bit_idx]
        if len(segment) == 0:
            continue

        bit_val = bitstream[bit_idx]
        displacement = strength if bit_val == 1 else -strength

        for pt in segment:
            y, x = pt
            dy = y - cy
            dx = x - cx
            dist = np.sqrt(dy * dy + dx * dx)
            if dist < 1e-6:
                continue

            uy = dy / dist
            ux = dx / dist

            if displacement > 0:
                for step_f in np.arange(0, displacement + 0.5, 0.5):
                    ny = int(round(y + uy * step_f))
                    nx_val = int(round(x + ux * step_f))
                    if 0 <= ny < h and 0 <= nx_val < w:
                        new_mask[ny, nx_val] = 255
            else:
                for step_f in np.arange(0, abs(displacement) + 0.5, 0.5):
                    ny = int(round(y - uy * step_f))
                    nx_val = int(round(x - ux * step_f))
                    if 0 <= ny < h and 0 <= nx_val < w:
                        new_mask[ny, nx_val] = 0

    audit("lumen.encode_polar_done", logger=log, bits_encoded=n_bits)
    return new_mask


@trace
def decode_polar(encoded_mask: np.ndarray, original_mask: np.ndarray,
                 strength: float = 2.0) -> str | None:
    """Decode polar-displacement encoding by comparing radial extents.

    For each angular sector, casts multiple rays from the centroid through
    both masks and compares the maximum radial extent. This is robust to
    contour topology changes (new edges from erosion near concavities).

    Returns decoded string or None on failure.
    """
    orig_bool = original_mask.astype(bool)
    enc_bool = encoded_mask > 127 if encoded_mask.max() > 1 else encoded_mask.astype(bool)

    if not np.any(orig_bool) or not np.any(enc_bool):
        audit("lumen.decode_polar_fail", logger=log, reason="empty_mask")
        return None

    cy, cx = ndimage.center_of_mass(orig_bool)
    max_radius = max(encoded_mask.shape) * 0.6

    # Precompute radial extents at many angles for fast lookup
    n_precompute = 2000
    precompute_angles = np.linspace(-np.pi, np.pi, n_precompute, endpoint=False)
    orig_extents = np.array([
        _measure_radial_extent(orig_bool, cy, cx, a, max_radius)
        for a in precompute_angles
    ])
    enc_extents = np.array([
        _measure_radial_extent(enc_bool, cy, cx, a, max_radius)
        for a in precompute_angles
    ])

    def _read_bits_fast(n_bits: int) -> list[int]:
        """Read n_bits using precomputed radial extents."""
        sector_width = 2 * np.pi / n_bits
        bits = []

        for i in range(n_bits):
            sector_center = -np.pi + (i + 0.5) * sector_width
            lo = sector_center - sector_width * 0.4
            hi = sector_center + sector_width * 0.4

            # Find precomputed angles in this sector
            in_sector = ((precompute_angles >= lo) & (precompute_angles < hi))
            if not np.any(in_sector):
                # Fallback: use closest angle
                closest = np.argmin(np.abs(precompute_angles - sector_center))
                in_sector = np.zeros(n_precompute, dtype=bool)
                in_sector[closest] = True

            seg_orig = orig_extents[in_sector]
            seg_enc = enc_extents[in_sector]

            # Only use rays that hit the original mask
            valid = seg_orig > 0
            if not np.any(valid):
                bits.append(0)
            else:
                disps = seg_enc[valid] - seg_orig[valid]
                bits.append(1 if np.median(disps) > 0 else 0)

        return bits

    # The encoder uses total_bits sectors (= 32 + data_bits_len).
    # We don't know total_bits a priori. Try plausible byte counts
    # (1..200 data bytes = 40..1632 total bits) and check CRC.
    for data_bytes in range(1, 201):
        data_bits_len = data_bytes * 8
        total_bits = 24 + data_bits_len + 8  # header + data + crc

        all_bits = _read_bits_fast(total_bits)
        result = _parse_bitstream(all_bits)
        if result is not None:
            audit("lumen.decode_polar_ok", logger=log,
                  decoded_len=len(result), total_bits=total_bits)
            return result

    audit("lumen.decode_polar_fail", logger=log, reason="no_valid_decode")
    return None


# ---------------------------------------------------------------------------
# Stroke-width modulation encoding
# ---------------------------------------------------------------------------

@trace
def encode_stroke_width(logo_mask: np.ndarray, data: str,
                        strength: float = 1.5) -> np.ndarray:
    """Encode data via stroke-width modulation along the contour.

    Bit=1 segments are dilated (thickened), bit=0 segments are eroded (thinned).

    Returns a modified binary mask (uint8, 0/255).
    """
    bool_mask = logo_mask.astype(bool)
    contour = _extract_contour(bool_mask)

    if len(contour) == 0:
        audit("lumen.encode_stroke_fail", logger=log, reason="empty_contour")
        return bool_mask.astype(np.uint8) * 255

    bitstream = _build_bitstream(data)
    n_bits = len(bitstream)

    audit("lumen.encode_stroke", logger=log, data_len=len(data),
          bits=n_bits, contour_pts=len(contour))

    dist_transform = ndimage.distance_transform_edt(bool_mask)

    h, w = bool_mask.shape
    new_mask = bool_mask.copy().astype(np.uint8) * 255

    pts_per_bit = max(1, len(contour) // n_bits)
    radius = int(np.ceil(strength)) + 1

    for bit_idx in range(n_bits):
        start = bit_idx * pts_per_bit
        end = start + pts_per_bit if bit_idx < n_bits - 1 else len(contour)
        if start >= len(contour):
            break

        segment = contour[start:end]
        bit_val = bitstream[bit_idx]

        for pt in segment:
            y, x = int(round(pt[0])), int(round(pt[1]))

            if bit_val == 1:
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            if dy * dy + dx * dx <= strength * strength:
                                new_mask[ny, nx] = 255
            else:
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            if dy * dy + dx * dx <= strength * strength:
                                if dist_transform[ny, nx] <= strength:
                                    new_mask[ny, nx] = 0

    audit("lumen.encode_stroke_done", logger=log, bits_encoded=n_bits)
    return new_mask


# ---------------------------------------------------------------------------
# High-level entry points
# ---------------------------------------------------------------------------

@trace
def generate_lumen_code(
    data: str,
    *,
    logo=None,
    use_apple: bool = True,
    method: str = "polar",
    strength: float = 2.0,
    output_size: int = 512,
) -> dict:
    """Generate a Lumen-Code image — data encoded in the logo contour.

    Args:
        data: The string to encode.
        logo: Optional path to a logo image file.
        use_apple: If True (and no logo provided), use built-in Apple logo.
        method: Encoding method ('polar' or 'stroke').
        strength: Displacement strength in pixels.
        output_size: Output image size in pixels.

    Returns:
        dict with keys: image, method, bits_encoded, round_trip_ok, decoded_data
    """
    from qrx.logo import create_apple_silhouette, load_logo, logo_to_binary_mask

    audit("lumen.generate_start", logger=log, data=data[:80], method=method,
          strength=strength, output_size=output_size)

    if logo is not None:
        logo_img = load_logo(logo)
        logo_img = logo_img.resize((output_size, output_size), Image.LANCZOS)
    elif use_apple:
        logo_img = create_apple_silhouette(output_size)
    else:
        raise ValueError("Either logo path or use_apple=True must be provided")

    mask = logo_to_binary_mask(logo_img)

    if method == "polar":
        encoded_mask = encode_polar(mask, data, strength=strength)
    elif method == "stroke":
        encoded_mask = encode_stroke_width(mask, data, strength=strength)
    else:
        raise ValueError(f"Unknown method: {method}")

    bitstream = _build_bitstream(data)
    n_bits = len(bitstream)

    # Render as image (black logo on white background)
    img = Image.fromarray(255 - encoded_mask)
    img = img.convert("L")

    # Attempt round-trip decode
    decoded = None
    round_trip_ok = False
    if method == "polar":
        original_mask_uint8 = mask.astype(np.uint8) * 255
        decoded = decode_polar(encoded_mask, original_mask_uint8, strength=strength)
        round_trip_ok = decoded == data

    audit("lumen.generate_done", logger=log, bits=n_bits,
          round_trip=round_trip_ok, method=method)

    return {
        "image": img,
        "method": method,
        "bits_encoded": n_bits,
        "round_trip_ok": round_trip_ok,
        "decoded_data": decoded,
    }


@trace
def decode_lumen_code(
    image,
    *,
    original_logo=None,
    use_apple: bool = True,
    method: str = "polar",
    strength: float = 2.0,
) -> dict:
    """Decode a Lumen-Code image.

    Args:
        image: PIL Image or path to image file.
        original_logo: Path to original logo for reference.
        use_apple: If True, use Apple logo as reference.
        method: Encoding method ('polar' or 'stroke').
        strength: Displacement strength used during encoding.

    Returns:
        dict with keys: decoded_data, success, method
    """
    from qrx.logo import create_apple_silhouette, load_logo, logo_to_binary_mask

    audit("lumen.decode_start", logger=log, method=method, strength=strength)

    if isinstance(image, str):
        image = Image.open(image)
    if isinstance(image, Image.Image):
        gray = image.convert("L")
        arr = np.array(gray)
        encoded_mask = (arr < 128).astype(np.uint8) * 255
    else:
        encoded_mask = image

    output_size = encoded_mask.shape[0]

    if original_logo is not None:
        logo_img = load_logo(original_logo)
        logo_img = logo_img.resize((output_size, output_size), Image.LANCZOS)
    elif use_apple:
        logo_img = create_apple_silhouette(output_size)
    else:
        return {"decoded_data": None, "success": False, "method": method}

    original_mask = logo_to_binary_mask(logo_img).astype(np.uint8) * 255

    decoded = None
    if method == "polar":
        decoded = decode_polar(encoded_mask, original_mask, strength=strength)

    success = decoded is not None

    audit("lumen.decode_done", logger=log, success=success, method=method)

    return {
        "decoded_data": decoded,
        "success": success,
        "method": method,
    }
