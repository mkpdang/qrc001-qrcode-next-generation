"""M1: Scan-Verify & Stress Test — automated QR code verification and resilience testing."""

import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from pyzbar.pyzbar import decode as pyzbar_decode

from qrx.logging import audit, get_logger, trace

log = get_logger("verify")


@dataclass
class ScanResult:
    """Result of a single scan attempt."""
    success: bool
    decoded_data: str | None = None
    decode_time_ms: float = 0.0
    decoder: str = ""
    error: str | None = None


@dataclass
class StressTestResult:
    """Result of a full stress test battery."""
    original: ScanResult = field(default_factory=lambda: ScanResult(success=False))
    blur_results: dict[str, ScanResult] = field(default_factory=dict)
    brightness_results: dict[str, ScanResult] = field(default_factory=dict)
    rotation_results: dict[str, ScanResult] = field(default_factory=dict)
    occlusion_results: dict[str, ScanResult] = field(default_factory=dict)
    total_tests: int = 0
    total_passed: int = 0

    @property
    def pass_rate(self) -> float:
        return self.total_passed / self.total_tests if self.total_tests > 0 else 0.0

    def summary(self) -> str:
        lines = [
            f"Stress Test Summary: {self.total_passed}/{self.total_tests} passed ({self.pass_rate:.1%})",
            f"  Original:    {'PASS' if self.original.success else 'FAIL'} ({self.original.decode_time_ms:.1f}ms)",
        ]
        for category, results in [
            ("Blur", self.blur_results),
            ("Brightness", self.brightness_results),
            ("Rotation", self.rotation_results),
            ("Occlusion", self.occlusion_results),
        ]:
            passed = sum(1 for r in results.values() if r.success)
            lines.append(f"  {category:12s}: {passed}/{len(results)} passed")
            for name, r in results.items():
                status = "PASS" if r.success else "FAIL"
                lines.append(f"    {name:20s}: {status} ({r.decode_time_ms:.1f}ms)")
        return "\n".join(lines)


@trace
def scan_pyzbar(image: Image.Image) -> ScanResult:
    """Scan a QR code using pyzbar (wraps ZBar)."""
    start = time.perf_counter()
    try:
        results = pyzbar_decode(image)
        elapsed = (time.perf_counter() - start) * 1000

        if results:
            data = results[0].data.decode("utf-8", errors="replace")
            audit("scan.verified", logger=log, decoder="pyzbar/zbar", success=True, time_ms=round(elapsed, 1), data=data[:80])
            return ScanResult(
                success=True,
                decoded_data=data,
                decode_time_ms=elapsed,
                decoder="pyzbar/zbar",
            )
        audit("scan.verified", logger=log, decoder="pyzbar/zbar", success=False, time_ms=round(elapsed, 1), error="No QR code detected")
        return ScanResult(
            success=False,
            decode_time_ms=elapsed,
            decoder="pyzbar/zbar",
            error="No QR code detected",
        )
    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        audit("scan.error", logger=log, decoder="pyzbar/zbar", error=str(e), time_ms=round(elapsed, 1))
        return ScanResult(
            success=False,
            decode_time_ms=elapsed,
            decoder="pyzbar/zbar",
            error=str(e),
        )


@trace
def scan_opencv(image: Image.Image) -> ScanResult:
    """Scan a QR code using OpenCV's built-in QR detector (wraps quirc/wechat)."""
    start = time.perf_counter()
    try:
        arr = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        detector = cv2.QRCodeDetector()
        data, points, _ = detector.detectAndDecode(gray)
        elapsed = (time.perf_counter() - start) * 1000

        if data:
            audit("scan.verified", logger=log, decoder="opencv", success=True, time_ms=round(elapsed, 1), data=data[:80])
            return ScanResult(
                success=True,
                decoded_data=data,
                decode_time_ms=elapsed,
                decoder="opencv",
            )
        audit("scan.verified", logger=log, decoder="opencv", success=False, time_ms=round(elapsed, 1), error="No QR code detected")
        return ScanResult(
            success=False,
            decode_time_ms=elapsed,
            decoder="opencv",
            error="No QR code detected",
        )
    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        audit("scan.error", logger=log, decoder="opencv", error=str(e), time_ms=round(elapsed, 1))
        return ScanResult(
            success=False,
            decode_time_ms=elapsed,
            decoder="opencv",
            error=str(e),
        )


@trace
def verify(image: Image.Image, expected_data: str | None = None) -> list[ScanResult]:
    """Run all available decoders on an image.

    Args:
        image: PIL Image containing a QR code.
        expected_data: If provided, marks result as failure if decoded data doesn't match.

    Returns:
        List of ScanResults, one per decoder.
    """
    results = []
    for scanner in [scan_pyzbar, scan_opencv]:
        result = scanner(image)
        if result.success and expected_data and result.decoded_data != expected_data:
            result.success = False
            result.error = f"Data mismatch: got '{result.decoded_data}', expected '{expected_data}'"
        results.append(result)
    return results


@trace
def apply_blur(image: Image.Image, radius: float) -> Image.Image:
    """Apply Gaussian blur."""
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


@trace
def apply_brightness(image: Image.Image, factor: float) -> Image.Image:
    """Adjust brightness. factor=1.0 is original, <1 darker, >1 brighter."""
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


@trace
def apply_rotation(image: Image.Image, degrees: float) -> Image.Image:
    """Rotate image by degrees (with white background fill)."""
    return image.rotate(degrees, expand=True, fillcolor=(255, 255, 255))


@trace
def apply_occlusion(image: Image.Image, coverage: float = 0.05, position: str = "center") -> Image.Image:
    """Cover a portion of the image with a white rectangle.

    Args:
        coverage: Fraction of image area to cover (0.0-1.0).
        position: "center", "top-left", "bottom-right".
    """
    img = image.copy()
    w, h = img.size
    block_w = int(w * (coverage ** 0.5))
    block_h = int(h * (coverage ** 0.5))
    draw = ImageDraw.Draw(img)

    position_offsets = {
        "top-left": (0, 0),
        "bottom-right": (w - block_w, h - block_h),
    }
    x0, y0 = position_offsets.get(
        position,
        ((w - block_w) // 2, (h - block_h) // 2),
    )

    draw.rectangle([x0, y0, x0 + block_w, y0 + block_h], fill=(255, 255, 255))
    return img


@trace
def stress_test(
    image: Image.Image,
    expected_data: str | None = None,
    decoder: str = "pyzbar",
) -> StressTestResult:
    """Run a full battery of stress tests on a QR code image.

    Tests:
        - Gaussian blur: radius 1, 2, 3, 5
        - Brightness: 0.3 (dark), 0.6 (dim), 1.5 (bright), 2.0 (overexposed)
        - Rotation: ±15°, ±30°, ±45°
        - Occlusion: 5% center, 5% corner, 10% center
    """
    scanner = scan_pyzbar if decoder == "pyzbar" else scan_opencv

    def _scan(img: Image.Image) -> ScanResult:
        result = scanner(img)
        if result.success and expected_data and result.decoded_data != expected_data:
            result.success = False
            result.error = f"Data mismatch: got '{result.decoded_data}', expected '{expected_data}'"
        return result

    result = StressTestResult()

    # Original
    result.original = _scan(image)
    result.total_tests = 1
    result.total_passed = 1 if result.original.success else 0

    # Blur tests
    for radius in [1, 2, 3, 5]:
        name = f"radius={radius}"
        r = _scan(apply_blur(image, radius))
        result.blur_results[name] = r
        result.total_tests += 1
        result.total_passed += 1 if r.success else 0

    # Brightness tests
    for factor in [0.3, 0.6, 1.5, 2.0]:
        name = f"factor={factor}"
        r = _scan(apply_brightness(image, factor))
        result.brightness_results[name] = r
        result.total_tests += 1
        result.total_passed += 1 if r.success else 0

    # Rotation tests
    for degrees in [-15, 15, -30, 30, -45, 45]:
        name = f"{degrees}°"
        r = _scan(apply_rotation(image, degrees))
        result.rotation_results[name] = r
        result.total_tests += 1
        result.total_passed += 1 if r.success else 0

    # Occlusion tests
    for coverage, position in [(0.05, "center"), (0.05, "top-left"), (0.10, "center")]:
        name = f"{coverage:.0%} {position}"
        r = _scan(apply_occlusion(image, coverage, position))
        result.occlusion_results[name] = r
        result.total_tests += 1
        result.total_passed += 1 if r.success else 0

    audit("stress.completed", logger=log,
          decoder=decoder,
          pass_rate=f"{result.pass_rate:.1%}",
          passed=result.total_passed,
          total=result.total_tests)
    return result
