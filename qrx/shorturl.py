"""M0: Short URL Engine — Base62/numeric key generation and redirect service."""

import json
import string
import threading
from pathlib import Path

from qrx.logging import audit, get_logger, trace

log = get_logger("shorturl")

# Base62 alphabet: 0-9, a-z, A-Z
BASE62_ALPHABET = string.digits + string.ascii_lowercase + string.ascii_uppercase
BASE62_SIZE = len(BASE62_ALPHABET)

# Domain-specific compression dictionary for future use (M7/M8)
DOMAIN_DICTIONARY = {
    0x00: "http://",
    0x01: "https://",
    0x02: "apple.co/",
    0x03: "apple.com/",
    0x04: "qr.ai/",
    0x05: "www.",
}
# Reverse lookup
DOMAIN_DICTIONARY_REV = {v: k for k, v in DOMAIN_DICTIONARY.items()}


@trace
def int_to_base62(n: int) -> str:
    """Convert a non-negative integer to a Base62 string."""
    if n == 0:
        return BASE62_ALPHABET[0]
    result = []
    while n > 0:
        n, remainder = divmod(n, BASE62_SIZE)
        result.append(BASE62_ALPHABET[remainder])
    return "".join(reversed(result))


@trace
def base62_to_int(s: str) -> int:
    """Convert a Base62 string back to an integer."""
    n = 0
    for char in s:
        n = n * BASE62_SIZE + BASE62_ALPHABET.index(char)
    return n


@trace
def int_to_numeric(n: int, min_length: int = 6) -> str:
    """Convert integer to numeric-only key (for QR numeric mode — 30% more efficient)."""
    s = str(n)
    return s.zfill(min_length)


@trace
def generate_key(counter: int, mode: str = "base62") -> str:
    """Generate a short key from a counter value.

    Args:
        counter: Monotonically increasing ID.
        mode: "base62" (default, 6 chars = 9B IDs) or "numeric" (digits only).
    """
    if mode == "numeric":
        return int_to_numeric(counter)
    return int_to_base62(counter)


@trace
def compress_url_prefix(url: str) -> tuple[int | None, str]:
    """Check if URL starts with a known dictionary prefix.

    Returns (token, remainder) if matched, or (None, url) if not.
    """
    for prefix, token in sorted(DOMAIN_DICTIONARY_REV.items(), key=lambda x: -len(x[0])):
        if url.startswith(prefix):
            return token, url[len(prefix):]
    return None, url


@trace
def decompress_url_prefix(token: int, remainder: str) -> str:
    """Reconstruct URL from a dictionary token and remainder."""
    return DOMAIN_DICTIONARY[token] + remainder


class ShortURLStore:
    """Simple JSON-file-backed URL shortener store.

    Thread-safe. For production, replace with a real database.
    """

    def __init__(self, db_path: str = "shorturl_db.json"):
        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        self._data = {"counter": 1, "urls": {}, "reverse": {}}
        if self.db_path.exists():
            with open(self.db_path) as f:
                self._data = json.load(f)
            log.info("Loaded store from %s (%d urls)", self.db_path, len(self._data["urls"]))

    def _save(self):
        with open(self.db_path, "w") as f:
            json.dump(self._data, f, indent=2)

    @trace
    def shorten(self, destination_url: str, mode: str = "base62") -> str:
        """Shorten a URL. Returns the short key.

        If the URL was already shortened, returns the existing key.
        """
        with self._lock:
            # Check if already exists
            if destination_url in self._data["reverse"]:
                existing_key = self._data["reverse"][destination_url]
                audit("url.cache_hit", logger=log, url=destination_url[:80], key=existing_key)
                return existing_key

            key = generate_key(self._data["counter"], mode=mode)
            self._data["counter"] += 1
            self._data["urls"][key] = destination_url
            self._data["reverse"][destination_url] = key
            self._save()
            audit("url.shortened", logger=log, url=destination_url[:80], key=key, mode=mode)
            return key

    @trace
    def resolve(self, key: str) -> str | None:
        """Resolve a short key to its destination URL."""
        result = self._data["urls"].get(key)
        if result is None:
            audit("url.resolve_miss", logger=log, key=key)
        else:
            audit("url.resolved", logger=log, key=key, destination=result[:80])
        return result

    @trace
    def stats(self) -> dict:
        """Return store statistics."""
        return {
            "total_urls": len(self._data["urls"]),
            "next_counter": self._data["counter"],
        }


@trace
def create_redirect_app(store: ShortURLStore, base_domain: str = "qr.ai"):
    """Create a Flask app that serves 301 redirects."""
    from flask import Flask, abort, jsonify, redirect, request

    app = Flask(__name__)

    @app.route("/<key>")
    def redirect_to_url(key):
        destination = store.resolve(key)
        if destination is None:
            audit("redirect.404", logger=log, key=key)
            abort(404)
        audit("redirect.301", logger=log, key=key, destination=destination[:80])
        return redirect(destination, code=301)

    @app.route("/api/shorten", methods=["POST"])
    def shorten_url():
        data = request.get_json()
        if not data or "url" not in data:
            return jsonify({"error": "Missing 'url' field"}), 400
        mode = data.get("mode", "base62")
        key = store.shorten(data["url"], mode=mode)
        short_url = f"{base_domain}/{key}"
        return jsonify({"key": key, "short_url": short_url, "destination": data["url"]})

    @app.route("/api/stats")
    def get_stats():
        return jsonify(store.stats())

    return app


if __name__ == "__main__":
    from qrx.logging import setup_logging
    setup_logging("DEBUG")

    store = ShortURLStore()

    examples = [
        "https://www.apple.com/shop/buy-iphone/iphone-15-pro",
        "https://www.google.com/search?q=qr+codes",
        "https://example.com/very/long/path/to/something?with=params&and=more",
    ]

    for url in examples:
        key_b62 = store.shorten(url, mode="base62")
        key_num = generate_key(base62_to_int(key_b62), mode="numeric")
        token, remainder = compress_url_prefix(url)
        print(f"URL:      {url}")
        print(f"Base62:   qr.ai/{key_b62}  ({len('qr.ai/' + key_b62)} chars)")
        print(f"Numeric:  qr.ai/{key_num}  ({len('qr.ai/' + key_num)} chars)")
        if token is not None:
            print(f"Dict:     token=0x{token:02x} + '{remainder}'")
        print()
