"""M8: Data Fonts — Machine-Readable Typography.

Encodes a URL into the pixel structure of rendered text so that the visual
word (e.g. "APPLE") doubles as a machine-readable data carrier.
"""

import string

from PIL import Image

from qrx.logging import audit, get_logger, trace

log = get_logger("datafont")

# ---------------------------------------------------------------------------
# 1. DNA-ZIP Compression Engine
# ---------------------------------------------------------------------------

STATIC_DICT: dict[int, str] = {
    0x0: "http://",
    0x1: "https://",
    0x2: "apple.com/",
    0x3: "apple.co/",
    0x4: "qr.ai/",
    0x5: "www.",
    0x6: ".com/",
    0x7: ".co/",
}
STATIC_DICT_REV: dict[str, int] = {v: k for k, v in STATIC_DICT.items()}

# Base62 alphabet used for per-character encoding (6 bits each)
BASE62 = string.digits + string.ascii_lowercase + string.ascii_uppercase
BASE62_INDEX: dict[str, int] = {ch: i for i, ch in enumerate(BASE62)}


def _int_to_bits(value: int, width: int) -> list[int]:
    """Convert an integer to a fixed-width list of bits (MSB first)."""
    return [(value >> (width - 1 - i)) & 1 for i in range(width)]


def _bits_to_int(bits: list[int]) -> int:
    """Convert a list of bits (MSB first) back to an integer."""
    n = 0
    for b in bits:
        n = (n << 1) | b
    return n


@trace
def compress_url(url: str) -> list[int]:
    """Compress a URL into a bit list using STATIC_DICT + Base62."""
    # Try to match the longest prefix
    best_token = None
    best_prefix_len = 0
    for prefix, token in sorted(STATIC_DICT_REV.items(), key=lambda x: -len(x[0])):
        if url.startswith(prefix):
            best_token = token
            best_prefix_len = len(prefix)
            break

    bits: list[int] = []
    if best_token is not None:
        bits.extend(_int_to_bits(best_token, 4))
        remainder = url[best_prefix_len:]
    else:
        bits.extend(_int_to_bits(0xF, 4))  # no prefix match
        remainder = url

    # Encode each remaining character as a 6-bit Base62 index.
    # Characters outside Base62 are encoded with a 6-bit escape (0x3F)
    # followed by an 8-bit raw byte.
    for ch in remainder:
        if ch in BASE62_INDEX:
            bits.extend(_int_to_bits(BASE62_INDEX[ch], 6))
        else:
            bits.extend(_int_to_bits(0x3F, 6))  # escape marker
            bits.extend(_int_to_bits(ord(ch) & 0xFF, 8))

    audit("datafont.compress", logger=log, url_len=len(url), bits=len(bits))
    return bits


@trace
def decompress_url(bits: list[int]) -> str:
    """Reverse the compression performed by compress_url."""
    if len(bits) < 4:
        return ""

    token = _bits_to_int(bits[:4])
    pos = 4

    if token == 0xF:
        prefix = ""
    else:
        prefix = STATIC_DICT.get(token, "")

    chars: list[str] = []
    while pos + 6 <= len(bits):
        idx = _bits_to_int(bits[pos:pos + 6])
        pos += 6
        if idx == 0x3F:
            # Escape: next 8 bits are a raw byte
            if pos + 8 <= len(bits):
                raw = _bits_to_int(bits[pos:pos + 8])
                pos += 8
                chars.append(chr(raw))
        else:
            if idx < len(BASE62):
                chars.append(BASE62[idx])

    url = prefix + "".join(chars)
    audit("datafont.decompress", logger=log, url_len=len(url), bits_consumed=pos)
    return url


# ---------------------------------------------------------------------------
# 2. Pixel Font Definitions
# ---------------------------------------------------------------------------

FONT_3PX: dict[str, list[list[int]]] = {
    "A": [[0,1,0],[1,0,1],[1,1,1],[1,0,1],[1,0,1]],
    "B": [[1,1,0],[1,0,1],[1,1,0],[1,0,1],[1,1,0]],
    "C": [[0,1,1],[1,0,0],[1,0,0],[1,0,0],[0,1,1]],
    "D": [[1,1,0],[1,0,1],[1,0,1],[1,0,1],[1,1,0]],
    "E": [[1,1,1],[1,0,0],[1,1,0],[1,0,0],[1,1,1]],
    "F": [[1,1,1],[1,0,0],[1,1,0],[1,0,0],[1,0,0]],
    "G": [[0,1,1],[1,0,0],[1,0,1],[1,0,1],[0,1,1]],
    "H": [[1,0,1],[1,0,1],[1,1,1],[1,0,1],[1,0,1]],
    "I": [[1,1,1],[0,1,0],[0,1,0],[0,1,0],[1,1,1]],
    "J": [[0,0,1],[0,0,1],[0,0,1],[1,0,1],[0,1,0]],
    "K": [[1,0,1],[1,0,1],[1,1,0],[1,0,1],[1,0,1]],
    "L": [[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,1,1]],
    "M": [[1,0,1],[1,1,1],[1,0,1],[1,0,1],[1,0,1]],
    "N": [[1,0,1],[1,1,1],[1,1,1],[1,0,1],[1,0,1]],
    "O": [[0,1,0],[1,0,1],[1,0,1],[1,0,1],[0,1,0]],
    "P": [[1,1,0],[1,0,1],[1,1,0],[1,0,0],[1,0,0]],
    "Q": [[0,1,0],[1,0,1],[1,0,1],[1,1,1],[0,1,1]],
    "R": [[1,1,0],[1,0,1],[1,1,0],[1,0,1],[1,0,1]],
    "S": [[0,1,1],[1,0,0],[0,1,0],[0,0,1],[1,1,0]],
    "T": [[1,1,1],[0,1,0],[0,1,0],[0,1,0],[0,1,0]],
    "U": [[1,0,1],[1,0,1],[1,0,1],[1,0,1],[0,1,0]],
    "V": [[1,0,1],[1,0,1],[1,0,1],[0,1,0],[0,1,0]],
    "W": [[1,0,1],[1,0,1],[1,0,1],[1,1,1],[1,0,1]],
    "X": [[1,0,1],[1,0,1],[0,1,0],[1,0,1],[1,0,1]],
    "Y": [[1,0,1],[1,0,1],[0,1,0],[0,1,0],[0,1,0]],
    "Z": [[1,1,1],[0,0,1],[0,1,0],[1,0,0],[1,1,1]],
    "0": [[0,1,0],[1,0,1],[1,0,1],[1,0,1],[0,1,0]],
    "1": [[0,1,0],[1,1,0],[0,1,0],[0,1,0],[1,1,1]],
    "2": [[1,1,0],[0,0,1],[0,1,0],[1,0,0],[1,1,1]],
    "3": [[1,1,0],[0,0,1],[0,1,0],[0,0,1],[1,1,0]],
    "4": [[1,0,1],[1,0,1],[1,1,1],[0,0,1],[0,0,1]],
    "5": [[1,1,1],[1,0,0],[1,1,0],[0,0,1],[1,1,0]],
    "6": [[0,1,1],[1,0,0],[1,1,0],[1,0,1],[0,1,0]],
    "7": [[1,1,1],[0,0,1],[0,1,0],[0,1,0],[0,1,0]],
    "8": [[0,1,0],[1,0,1],[0,1,0],[1,0,1],[0,1,0]],
    "9": [[0,1,0],[1,0,1],[0,1,1],[0,0,1],[1,1,0]],
}

FONT_4PX: dict[str, list[list[int]]] = {
    "A": [[0,1,1,0],[1,0,0,1],[1,1,1,1],[1,0,0,1],[1,0,0,1],[1,0,0,1]],
    "B": [[1,1,1,0],[1,0,0,1],[1,1,1,0],[1,0,0,1],[1,0,0,1],[1,1,1,0]],
    "C": [[0,1,1,1],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[0,1,1,1]],
    "D": [[1,1,1,0],[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,1,1,0]],
    "E": [[1,1,1,1],[1,0,0,0],[1,1,1,0],[1,0,0,0],[1,0,0,0],[1,1,1,1]],
    "F": [[1,1,1,1],[1,0,0,0],[1,1,1,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]],
    "G": [[0,1,1,1],[1,0,0,0],[1,0,1,1],[1,0,0,1],[1,0,0,1],[0,1,1,0]],
    "H": [[1,0,0,1],[1,0,0,1],[1,1,1,1],[1,0,0,1],[1,0,0,1],[1,0,0,1]],
    "I": [[1,1,1,1],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[1,1,1,1]],
    "J": [[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1],[1,0,0,1],[0,1,1,0]],
    "K": [[1,0,0,1],[1,0,1,0],[1,1,0,0],[1,0,1,0],[1,0,0,1],[1,0,0,1]],
    "L": [[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,1,1,1]],
    "M": [[1,0,0,1],[1,1,1,1],[1,1,1,1],[1,0,0,1],[1,0,0,1],[1,0,0,1]],
    "N": [[1,0,0,1],[1,1,0,1],[1,1,1,1],[1,0,1,1],[1,0,0,1],[1,0,0,1]],
    "O": [[0,1,1,0],[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,0,0,1],[0,1,1,0]],
    "P": [[1,1,1,0],[1,0,0,1],[1,0,0,1],[1,1,1,0],[1,0,0,0],[1,0,0,0]],
    "Q": [[0,1,1,0],[1,0,0,1],[1,0,0,1],[1,0,1,1],[1,0,0,1],[0,1,1,1]],
    "R": [[1,1,1,0],[1,0,0,1],[1,0,0,1],[1,1,1,0],[1,0,1,0],[1,0,0,1]],
    "S": [[0,1,1,1],[1,0,0,0],[0,1,1,0],[0,0,0,1],[0,0,0,1],[1,1,1,0]],
    "T": [[1,1,1,1],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0]],
    "U": [[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,0,0,1],[0,1,1,0]],
    "V": [[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,0,0,1],[0,1,1,0],[0,1,1,0]],
    "W": [[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,1,1,1],[1,1,1,1],[1,0,0,1]],
    "X": [[1,0,0,1],[1,0,0,1],[0,1,1,0],[0,1,1,0],[1,0,0,1],[1,0,0,1]],
    "Y": [[1,0,0,1],[1,0,0,1],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0]],
    "Z": [[1,1,1,1],[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0],[1,1,1,1]],
    "0": [[0,1,1,0],[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,0,0,1],[0,1,1,0]],
    "1": [[0,1,0,0],[1,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[1,1,1,0]],
    "2": [[0,1,1,0],[1,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0],[1,1,1,1]],
    "3": [[1,1,1,0],[0,0,0,1],[0,1,1,0],[0,0,0,1],[0,0,0,1],[1,1,1,0]],
    "4": [[1,0,0,1],[1,0,0,1],[1,1,1,1],[0,0,0,1],[0,0,0,1],[0,0,0,1]],
    "5": [[1,1,1,1],[1,0,0,0],[1,1,1,0],[0,0,0,1],[0,0,0,1],[1,1,1,0]],
    "6": [[0,1,1,0],[1,0,0,0],[1,1,1,0],[1,0,0,1],[1,0,0,1],[0,1,1,0]],
    "7": [[1,1,1,1],[0,0,0,1],[0,0,1,0],[0,1,0,0],[0,1,0,0],[0,1,0,0]],
    "8": [[0,1,1,0],[1,0,0,1],[0,1,1,0],[1,0,0,1],[1,0,0,1],[0,1,1,0]],
    "9": [[0,1,1,0],[1,0,0,1],[1,0,0,1],[0,1,1,1],[0,0,0,1],[0,1,1,0]],
}

FONT_5PX: dict[str, list[list[int]]] = {
    "A": [[0,0,1,0,0],[0,1,0,1,0],[1,0,0,0,1],[1,1,1,1,1],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1]],
    "B": [[1,1,1,1,0],[1,0,0,0,1],[1,1,1,1,0],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,1,1,1,0]],
    "C": [[0,1,1,1,0],[1,0,0,0,1],[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,1],[0,1,1,1,0]],
    "D": [[1,1,1,1,0],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,1,1,1,0]],
    "E": [[1,1,1,1,1],[1,0,0,0,0],[1,1,1,0,0],[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[1,1,1,1,1]],
    "F": [[1,1,1,1,1],[1,0,0,0,0],[1,1,1,0,0],[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0]],
    "G": [[0,1,1,1,0],[1,0,0,0,1],[1,0,0,0,0],[1,0,1,1,1],[1,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0]],
    "H": [[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,1,1,1,1],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1]],
    "I": [[1,1,1,1,1],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[1,1,1,1,1]],
    "J": [[0,0,0,0,1],[0,0,0,0,1],[0,0,0,0,1],[0,0,0,0,1],[0,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0]],
    "K": [[1,0,0,0,1],[1,0,0,1,0],[1,0,1,0,0],[1,1,0,0,0],[1,0,1,0,0],[1,0,0,1,0],[1,0,0,0,1]],
    "L": [[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[1,1,1,1,1]],
    "M": [[1,0,0,0,1],[1,1,0,1,1],[1,0,1,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1]],
    "N": [[1,0,0,0,1],[1,1,0,0,1],[1,0,1,0,1],[1,0,0,1,1],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1]],
    "O": [[0,1,1,1,0],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0]],
    "P": [[1,1,1,1,0],[1,0,0,0,1],[1,0,0,0,1],[1,1,1,1,0],[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0]],
    "Q": [[0,1,1,1,0],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,0,1,0,1],[1,0,0,1,0],[0,1,1,0,1]],
    "R": [[1,1,1,1,0],[1,0,0,0,1],[1,0,0,0,1],[1,1,1,1,0],[1,0,1,0,0],[1,0,0,1,0],[1,0,0,0,1]],
    "S": [[0,1,1,1,0],[1,0,0,0,1],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[1,0,0,0,1],[0,1,1,1,0]],
    "T": [[1,1,1,1,1],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0]],
    "U": [[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0]],
    "V": [[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[0,1,0,1,0],[0,1,0,1,0],[0,0,1,0,0]],
    "W": [[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,0,1,0,1],[1,1,0,1,1],[1,0,0,0,1]],
    "X": [[1,0,0,0,1],[1,0,0,0,1],[0,1,0,1,0],[0,0,1,0,0],[0,1,0,1,0],[1,0,0,0,1],[1,0,0,0,1]],
    "Y": [[1,0,0,0,1],[1,0,0,0,1],[0,1,0,1,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0]],
    "Z": [[1,1,1,1,1],[0,0,0,0,1],[0,0,0,1,0],[0,0,1,0,0],[0,1,0,0,0],[1,0,0,0,0],[1,1,1,1,1]],
    "0": [[0,1,1,1,0],[1,0,0,0,1],[1,0,0,1,1],[1,0,1,0,1],[1,1,0,0,1],[1,0,0,0,1],[0,1,1,1,0]],
    "1": [[0,0,1,0,0],[0,1,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,1,1,1,0]],
    "2": [[0,1,1,1,0],[1,0,0,0,1],[0,0,0,0,1],[0,0,0,1,0],[0,0,1,0,0],[0,1,0,0,0],[1,1,1,1,1]],
    "3": [[0,1,1,1,0],[1,0,0,0,1],[0,0,0,0,1],[0,0,1,1,0],[0,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0]],
    "4": [[0,0,0,1,0],[0,0,1,1,0],[0,1,0,1,0],[1,0,0,1,0],[1,1,1,1,1],[0,0,0,1,0],[0,0,0,1,0]],
    "5": [[1,1,1,1,1],[1,0,0,0,0],[1,1,1,1,0],[0,0,0,0,1],[0,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0]],
    "6": [[0,1,1,1,0],[1,0,0,0,0],[1,0,0,0,0],[1,1,1,1,0],[1,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0]],
    "7": [[1,1,1,1,1],[0,0,0,0,1],[0,0,0,1,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0]],
    "8": [[0,1,1,1,0],[1,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0],[1,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0]],
    "9": [[0,1,1,1,0],[1,0,0,0,1],[1,0,0,0,1],[0,1,1,1,1],[0,0,0,0,1],[0,0,0,0,1],[0,1,1,1,0]],
}

# Data slot positions per glyph — visually "optional" pixel positions
# where hidden data bits can be embedded.
# Each is a list of (row, col) tuples.

DATA_SLOTS_3PX: list[tuple[int, int]] = [
    (0, 0), (0, 2),  # top corners
]

DATA_SLOTS_4PX: list[tuple[int, int]] = [
    (0, 0), (0, 3),  # top corners
    (5, 0), (5, 3),  # bottom corners
]

DATA_SLOTS_5PX: list[tuple[int, int]] = [
    (0, 0), (0, 4),  # top corners
    (6, 0), (6, 4),  # bottom corners
    (3, 0), (3, 4),  # mid-row edges
]

_FONTS = {
    3: FONT_3PX,
    4: FONT_4PX,
    5: FONT_5PX,
}

_DATA_SLOTS = {
    3: DATA_SLOTS_3PX,
    4: DATA_SLOTS_4PX,
    5: DATA_SLOTS_5PX,
}

_GLYPH_DIMS = {
    3: (3, 5),   # width, height
    4: (4, 6),
    5: (5, 7),
}


# ---------------------------------------------------------------------------
# 3. Cross-Character Parity
# ---------------------------------------------------------------------------

@trace
def compute_parity(data_bits: list[int], block_size: int = 8) -> list[int]:
    """Compute XOR parity across blocks of data_bits."""
    parity = [0] * block_size
    for i, bit in enumerate(data_bits):
        parity[i % block_size] ^= bit
    audit("datafont.parity_compute", logger=log, data_len=len(data_bits), parity_len=block_size)
    return parity


@trace
def verify_parity(data_bits: list[int], parity_bits: list[int], block_size: int = 8) -> bool:
    """Verify XOR parity of data_bits against parity_bits."""
    expected = compute_parity(data_bits, block_size)
    ok = expected == parity_bits
    audit("datafont.parity_verify", logger=log, ok=ok)
    return ok


# ---------------------------------------------------------------------------
# 4. Rendering
# ---------------------------------------------------------------------------

@trace
def render_data_text(
    text: str,
    url: str,
    *,
    font_size: int = 5,
    scale: int = 10,
    fg_color: tuple[int, int, int] = (0, 0, 0),
    bg_color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Render *text* as a bitmap image with *url* encoded in its pixel structure."""
    font = _FONTS[font_size]
    slots = _DATA_SLOTS[font_size]
    glyph_w, glyph_h = _GLYPH_DIMS[font_size]
    gap = 1  # 1-pixel gap between glyphs

    # Compress URL and compute parity
    data_bits = compress_url(url)
    parity_bits = compute_parity(data_bits)
    # Frame: 6-bit length prefix + data + 8-bit parity + zero padding
    length_prefix = _int_to_bits(len(data_bits), 6)
    payload = length_prefix + data_bits + parity_bits

    text = text.upper()
    total_slots = len(slots) * len(text)
    bits_available = total_slots
    # Pad with zeros if fewer bits than slots; truncate if too many
    if len(payload) < total_slots:
        bits_to_embed = payload + [0] * (total_slots - len(payload))
    else:
        bits_to_embed = payload[:total_slots]

    # Canvas dimensions (at 1x scale)
    canvas_w = len(text) * glyph_w + (len(text) - 1) * gap
    canvas_h = glyph_h

    img = Image.new("RGB", (canvas_w, canvas_h), bg_color)
    pixels = img.load()

    bit_idx = 0
    for ci, ch in enumerate(text):
        glyph = font.get(ch)
        if glyph is None:
            continue
        x_off = ci * (glyph_w + gap)

        # Build a set of data slot positions for fast lookup
        slot_set: dict[tuple[int, int], int | None] = {}
        for sr, sc in slots:
            if bit_idx < len(bits_to_embed):
                slot_set[(sr, sc)] = bits_to_embed[bit_idx]
                bit_idx += 1
            else:
                slot_set[(sr, sc)] = None

        for r, row in enumerate(glyph):
            for c, val in enumerate(row):
                px = x_off + c
                py = r
                if (r, c) in slot_set and slot_set[(r, c)] is not None:
                    data_bit = slot_set[(r, c)]
                    if val == 1:
                        # Structural pixel — use subtle grayscale encoding
                        if data_bit == 1:
                            color = fg_color  # full black
                        else:
                            color = (180, 180, 180)  # dark gray for bit 0
                    else:
                        # Background pixel — use fg/bg to encode
                        if data_bit == 1:
                            color = fg_color
                        else:
                            color = bg_color
                else:
                    color = fg_color if val == 1 else bg_color
                pixels[px, py] = color

    # Scale up using nearest-neighbor
    scaled = img.resize((canvas_w * scale, canvas_h * scale), Image.NEAREST)

    audit(
        "datafont.render",
        logger=log,
        text=text,
        url_len=len(url),
        bits_used=min(len(bits_to_embed), total_slots),
        bits_available=bits_available,
        scale=scale,
    )
    return scaled


# ---------------------------------------------------------------------------
# 5. Decoding
# ---------------------------------------------------------------------------

@trace
def decode_data_text(
    image: Image.Image,
    *,
    font_size: int = 5,
    scale: int = 10,
    n_chars: int = 5,
) -> str | None:
    """Decode a URL from a data-font rendered image."""
    slots = _DATA_SLOTS[font_size]
    glyph_w, glyph_h = _GLYPH_DIMS[font_size]
    gap = 1

    # Downscale
    canvas_w = n_chars * glyph_w + (n_chars - 1) * gap
    canvas_h = glyph_h
    small = image.resize((canvas_w, canvas_h), Image.NEAREST)
    pixels = small.load()

    all_bits: list[int] = []
    for ci in range(n_chars):
        x_off = ci * (glyph_w + gap)
        for sr, sc in slots:
            px = x_off + sc
            py = sr
            r, g, b = pixels[px, py][:3]
            brightness = (r + g + b) / 3.0
            # <=128 brightness → bit 1, >128 → bit 0
            all_bits.append(1 if brightness <= 128 else 0)

    # Decode framed bit stream: 6-bit length prefix + data + 8-bit parity + padding
    if len(all_bits) < 6 + 8:
        audit("datafont.decode_fail", logger=log, reason="not_enough_bits")
        return None

    data_len = _bits_to_int(all_bits[:6])
    if data_len <= 0 or 6 + data_len + 8 > len(all_bits):
        audit("datafont.decode_fail", logger=log, reason="invalid_length", data_len=data_len)
        return None

    data_bits = all_bits[6:6 + data_len]
    parity_bits = all_bits[6 + data_len:6 + data_len + 8]

    if not verify_parity(data_bits, parity_bits):
        audit("datafont.decode_fail", logger=log, reason="parity_mismatch")
        return None

    url = decompress_url(data_bits)
    audit("datafont.decode_ok", logger=log, url=url[:80])
    return url


# ---------------------------------------------------------------------------
# 6. Main entry point
# ---------------------------------------------------------------------------

@trace
def generate_data_font_demo(
    text: str,
    url: str,
    *,
    font_size: int = 5,
    scale: int = 10,
) -> dict:
    """Encode a URL into rendered text and verify round-trip decoding."""
    data_bits = compress_url(url)
    parity_bits = compute_parity(data_bits)
    # 6-bit length prefix + data + 8-bit parity
    total_payload = 6 + len(data_bits) + len(parity_bits)

    slots = _DATA_SLOTS[font_size]
    total_slots = len(slots) * len(text)

    image = render_data_text(text, url, font_size=font_size, scale=scale)
    decoded_url = decode_data_text(
        image, font_size=font_size, scale=scale, n_chars=len(text),
    )

    round_trip_ok = decoded_url == url
    compression_ratio = len(url) / max(len(data_bits) / 8, 1)

    result = {
        "image": image,
        "text": text.upper(),
        "url": url,
        "font_size": font_size,
        "bits_used": min(total_payload, total_slots),
        "bits_available": total_slots,
        "compression_ratio": round(compression_ratio, 2),
        "round_trip_ok": round_trip_ok,
        "decoded_url": decoded_url,
    }

    audit(
        "datafont.demo",
        logger=log,
        text=text,
        url=url[:80],
        round_trip=round_trip_ok,
        bits_used=result["bits_used"],
        bits_available=total_slots,
    )
    return result
