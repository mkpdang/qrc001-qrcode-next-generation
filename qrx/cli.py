"""QR-X CLI — command-line interface for the QR-X toolchain."""

import argparse
import sys
from pathlib import Path

from PIL import Image

from qrx.logging import setup_logging, get_logger, audit

log = get_logger("cli")


def _parse_hex_color(s: str) -> tuple[int, int, int]:
    """Parse a hex colour string (with or without '#') to an RGB tuple."""
    s = s.lstrip("#")
    return tuple(int(s[i : i + 2], 16) for i in (0, 2, 4))


def cmd_generate(args):
    """Generate a QR code."""
    from qrx.generator import generate_qr, generate_all_masks

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.mask == "all":
        results = generate_all_masks(
            data=args.url,
            version=args.version,
            ecc=args.ecc,
            box_size=args.box_size,
            border=args.border,
        )
        for mask_idx, img in results:
            p = output.with_stem(f"{output.stem}_mask{mask_idx}")
            img.save(p)
            print(f"  Mask {mask_idx}: {p}")
        print(f"Generated {len(results)} variants.")
    else:
        mask = int(args.mask) if args.mask is not None else None
        img = generate_qr(
            data=args.url,
            version=args.version,
            ecc=args.ecc,
            mask=mask,
            box_size=args.box_size,
            border=args.border,
        )
        img.save(output)
        print(f"Generated: {output} ({img.size[0]}x{img.size[1]})")


def cmd_verify(args):
    """Verify a QR code image."""
    from qrx.verify import verify

    img = Image.open(args.image)
    results = verify(img, expected_data=args.expected)

    all_pass = True
    for r in results:
        status = "PASS" if r.success else "FAIL"
        if not r.success:
            all_pass = False
        print(f"  [{r.decoder:12s}] {status} | {r.decode_time_ms:6.1f}ms | {r.decoded_data or r.error}")

    sys.exit(0 if all_pass else 1)


def cmd_stress(args):
    """Run stress tests on a QR code image."""
    from qrx.verify import stress_test

    img = Image.open(args.image)
    result = stress_test(img, expected_data=args.expected, decoder=args.decoder)
    print(result.summary())
    sys.exit(0 if result.pass_rate >= 0.8 else 1)


def cmd_bitmap(args):
    """Generate a color-coded bitmap dump showing module types."""
    from qrx.generator import get_module_map, render_bitmap_dump

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    module_map = get_module_map(
        data=args.url,
        version=args.version,
        ecc=args.ecc,
        mask=int(args.mask) if args.mask is not None else None,
    )

    img = render_bitmap_dump(module_map, output_path=str(output))
    size = module_map["size"]
    n_data = len(module_map["data_positions"])
    n_finder = len(module_map["finder_positions"])
    n_align = len(module_map["alignment_positions"])
    n_timing = len(module_map["timing_positions"])
    n_format = len(module_map["format_positions"])

    print(f"QR Version {module_map['version']} ({size}x{size} = {size*size} modules)")
    print(f"  Finder:    {n_finder:4d} modules (red)")
    print(f"  Alignment: {n_align:4d} modules (blue)")
    print(f"  Timing:    {n_timing:4d} modules (green)")
    print(f"  Format:    {n_format:4d} modules (yellow)")
    print(f"  Data+ECC:  {n_data:4d} modules (black/white, gray=safe zone)")
    print(f"Saved to: {output}")


def cmd_shorten(args):
    """Shorten a URL."""
    from qrx.shorturl import ShortURLStore

    store = ShortURLStore()
    key = store.shorten(args.url, mode=args.mode)
    domain = args.domain
    short_url = f"{domain}/{key}"
    print(f"Short URL: {short_url} ({len(short_url)} chars)")
    print(f"Key:       {key}")
    print(f"Mode:      {args.mode}")


def cmd_logo_qr(args):
    """Generate a logo-aware QR code (M2)."""
    from qrx.logo_qr import generate_logo_qr

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    data_color = _parse_hex_color(args.color) if args.color else (0, 0, 0)
    finder_color = _parse_hex_color(args.finder_color) if args.finder_color else None
    logo_color = _parse_hex_color(args.logo_color) if args.logo_color else (0, 0, 0)

    result = generate_logo_qr(
        data=args.url,
        use_apple=args.apple,
        logo=args.logo,
        version=args.version,
        ecc=args.ecc,
        box_size=args.box_size,
        border=args.border,
        shape=args.shape,
        data_color=data_color,
        finder_color=finder_color,
        finder_style=args.finder_style,
        logo_color=logo_color,
    )

    result["image"].save(output)

    budget = result["ecc_budget"]
    print(f"Logo QR generated: {output} ({result['image'].size[0]}x{result['image'].size[1]})")
    print(f"  Version: {budget.version}, ECC: {budget.ecc}, Mask: {result['mask']}")
    print(f"  Shape: {args.shape}, Finder: {args.finder_style}")
    print(f"  ECC budget: {budget.budget_used_pct:.1f}% used "
          f"({budget.logo_covered_modules}/{budget.correctable_modules} modules)")
    if result["scan_ok"] is not None:
        scan_status = "PASS" if result["scan_ok"] else "FAIL"
        print(f"  Scan: {scan_status}")
        for sr in result["scan_results"]:
            tag = "PASS" if sr.success else "FAIL"
            print(f"    [{sr.decoder:12s}] {tag} | {sr.decode_time_ms:.1f}ms | {sr.decoded_data or sr.error}")
    print(f"  Leaf anchor: {result['leaf_anchor']}")
    print(f"  Mask scores (lower = better):")
    for m, s in sorted(result["mask_scores"].items()):
        marker = " <-- best" if m == result["mask"] else ""
        print(f"    Mask {m}: {s:>5d}{marker}")


def cmd_serve(args):
    """Start the redirect server."""
    from qrx.shorturl import ShortURLStore, create_redirect_app

    store = ShortURLStore()
    app = create_redirect_app(store, base_domain=args.domain)
    print(f"Starting redirect server on http://0.0.0.0:{args.port}")
    print(f"Base domain: {args.domain}")
    app.run(host="0.0.0.0", port=args.port, debug=args.debug)


def main():
    parser = argparse.ArgumentParser(prog="qrx", description="QR-X: Next Generation QR Code Toolchain")

    # Global logging flags
    parser.add_argument("-V", "--verbose", action="store_true", help="Enable DEBUG-level logging")
    parser.add_argument("--log-file", default=None, help="Write JSON logs to file")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- generate ---
    p_gen = subparsers.add_parser("generate", help="Generate a QR code")
    p_gen.add_argument("url", help="URL or data to encode")
    p_gen.add_argument("-o", "--output", default="output/qr.png", help="Output file path")
    p_gen.add_argument("-v", "--version", type=int, default=None, help="QR version 1-40 (auto if omitted)")
    p_gen.add_argument("-e", "--ecc", default="H", choices=["L", "M", "Q", "H"], help="Error correction level")
    p_gen.add_argument("-m", "--mask", default=None, help="Mask pattern 0-7, or 'all' for all 8")
    p_gen.add_argument("--box-size", type=int, default=20, help="Module pixel size")
    p_gen.add_argument("--border", type=int, default=4, help="Quiet zone modules")

    # --- verify ---
    p_ver = subparsers.add_parser("verify", help="Verify a QR code image")
    p_ver.add_argument("image", help="Path to QR code image")
    p_ver.add_argument("--expected", default=None, help="Expected decoded data (fails if mismatch)")

    # --- stress ---
    p_stress = subparsers.add_parser("stress", help="Run stress tests on a QR code image")
    p_stress.add_argument("image", help="Path to QR code image")
    p_stress.add_argument("--expected", default=None, help="Expected decoded data")
    p_stress.add_argument("--decoder", default="pyzbar", choices=["pyzbar", "opencv"], help="Decoder to use")

    # --- bitmap ---
    p_bmp = subparsers.add_parser("bitmap", help="Generate color-coded module map")
    p_bmp.add_argument("url", help="URL or data to encode")
    p_bmp.add_argument("-o", "--output", default="output/bitmap.png", help="Output file path")
    p_bmp.add_argument("-v", "--version", type=int, default=None, help="QR version")
    p_bmp.add_argument("-e", "--ecc", default="H", choices=["L", "M", "Q", "H"])
    p_bmp.add_argument("-m", "--mask", default=None, help="Mask pattern 0-7")

    # --- logo-qr ---
    p_logo = subparsers.add_parser("logo-qr", help="Generate logo-aware QR code (M2)")
    p_logo.add_argument("url", help="URL or data to encode")
    p_logo.add_argument("-o", "--output", default="output/logo_qr.png", help="Output file path")
    p_logo.add_argument("--apple", action="store_true", help="Use built-in Apple logo")
    p_logo.add_argument("--logo", default=None, help="Path to custom logo image")
    p_logo.add_argument("-v", "--version", type=int, default=None, help="QR version")
    p_logo.add_argument("-e", "--ecc", default="H", choices=["L", "M", "Q", "H"])
    p_logo.add_argument("--box-size", type=int, default=20, help="Module pixel size")
    p_logo.add_argument("--border", type=int, default=4, help="Quiet zone modules")
    p_logo.add_argument("--shape", default="circle", choices=["square", "circle", "rounded"],
                        help="Data module shape")
    p_logo.add_argument("--color", default=None, help="Data module colour (hex e.g. '000000')")
    p_logo.add_argument("--finder-color", default=None, help="Finder colour (hex)")
    p_logo.add_argument("--finder-style", default="rounded",
                        choices=["standard", "rounded", "dots"], help="Finder pattern style")
    p_logo.add_argument("--logo-color", default=None, help="Logo silhouette colour (hex)")

    # --- shorten ---
    p_short = subparsers.add_parser("shorten", help="Shorten a URL")
    p_short.add_argument("url", help="URL to shorten")
    p_short.add_argument("--mode", default="base62", choices=["base62", "numeric"], help="Key encoding mode")
    p_short.add_argument("--domain", default="qr.ai", help="Short URL domain")

    # --- serve ---
    p_serve = subparsers.add_parser("serve", help="Start the redirect server")
    p_serve.add_argument("--port", type=int, default=8080, help="Port to listen on")
    p_serve.add_argument("--domain", default="qr.ai", help="Base domain for short URLs")
    p_serve.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    # Setup logging before any command runs
    level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=level, log_file=args.log_file)
    audit("cli.start", logger=log, command=args.command, verbose=args.verbose)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "generate": cmd_generate,
        "verify": cmd_verify,
        "stress": cmd_stress,
        "bitmap": cmd_bitmap,
        "logo-qr": cmd_logo_qr,
        "shorten": cmd_shorten,
        "serve": cmd_serve,
    }
    commands[args.command](args)
    audit("cli.done", logger=log, command=args.command)


if __name__ == "__main__":
    main()
