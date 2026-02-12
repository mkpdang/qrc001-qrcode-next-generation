# QR-X: Next Generation QR Code — Technical Plan

## Goal

Build a new QR code system where the Apple logo IS the QR code. A phone scans what looks like an Apple logo and opens a URL.

## Core Principle

**Shorter data = sparser grid = more room for art.**
An 11-char URL like `qr.ai/apple` fits in a Version 2 (25x25) QR with Level H ECC, leaving 30% of modules available for logo integration.

## Critical Constraint: The Short URL is Non-Negotiable

Without a short URL, the entire plan collapses. A full URL like `apple.com/store/iphone-15-pro?utm=...` requires Version 6+ (41x41), making modules tiny and leaving zero room for logo art. **The URL shortener is the foundation — M1 must come before M2.**

---

## Demo Target

**Input:** Apple logo SVG + URL `qr.ai/apple`
**Output:** An image that looks like the Apple logo but scans as a working QR code on any phone.

---

## M0: Short URL Engine (1 week)

The foundation everything depends on. Without Version 1-2 QR codes, the logo will be a pixelated mess.

| # | Task |
|---|------|
| 0.1 | URL shortener micro-service (key → destination, 301 redirects) |
| 0.2 | Base62 key generation (9 billion IDs in 6 chars) |
| 0.3 | Numeric-only key mode (QR numeric encoding is 30% more efficient than alphanumeric) |
| 0.4 | Domain-specific compression dictionary (`apple.co/` → token, for later use in M7) |

**Output:** Any URL shortened to < 15 characters. Guarantees Version 2 QR or smaller.

---

## M1: QR Generation & Scan-Verify Toolchain (1 week)

Build the tools to iterate fast and measure everything.

| # | Task |
|---|------|
| 1.1 | Python QR generator using `segno`/`qrcode` — produce V1/V2/V3 at all ECC levels, all 8 mask patterns |
| 1.2 | **Custom pad codeword injection** — when data is short, QR fills remaining space with pad bytes (`0xEC`/`0x11`). Hack the encoder to use custom pad bytes that produce module patterns visually aligned with the logo *before* Reed-Solomon calculation. This reduces "noise" without costing any ECC budget. |
| 1.3 | Automated scan-verify using `pyzbar`/OpenCV — input image → decoded URL or failure + timing |
| 1.4 | Stress test harness — Gaussian blur, brightness shifts, rotation ±45°, partial occlusion, then re-scan |
| 1.5 | Bit-map dumper — visualize which modules are data vs ECC vs fixed patterns vs "safe to destroy" |
| 1.6 | **Multi-decoder verification** — test against ZBar, ZXing, iPhone native camera, Android native camera (not just one decoder) |

**Output:** CLI tool: `qrx generate --url "qr.ai/apple" --version 2 --ecc H --mask auto` + `qrx verify image.png`

---

## M2: Logo-Aware QR Engine (2 weeks)

Make the QR code aware of the logo and optimize around it.

| # | Task |
|---|------|
| 2.1 | Logo safe-zone calculator — given a logo SVG, compute which QR modules it overlaps at center placement |
| 2.2 | **Geometry-aware mask optimizer** — for each of 8 masks, don't just count destroyed bits; heavily penalize black pixels that land inside the Apple "bite" or outside the apple silhouette boundary. The bite and leaf must stay clean. |
| 2.3 | ECC budget tracker — show exactly how many modules can be sacrificed (Level H = 30%) vs how many the logo needs |
| 2.4 | **Finder pattern integration strategy** — the three 7x7 corner squares are mandatory for standard readers. Design how they coexist with the logo (e.g., style them as part of the frame, round their corners, use brand color) rather than pretending they don't exist |
| 2.5 | Module shape renderer — circles, rounded rects, hexagons, custom shapes (center sampling point stays correct) |
| 2.6 | **Dynamic contrast booster** — Apple's aesthetic is often low-contrast (gray-on-white). QR needs >70% contrast. Add subtle shadow/outline to ensure scannability without ruining flat design |
| 2.7 | Adaptive color — render data modules in brand color, validate contrast ratio >= 7:1 |
| 2.8 | Apple logo overlay — place icon in center 7x7 zone, verify scan works |
| 2.9 | **The "leaf as anchor" technique** — use the Apple leaf as a functional orientation marker or version-info bit for custom decoders (future M6 compatibility) |

**Output:** QR code with Apple logo cleanly embedded. Scans on any phone from 2018+.

**Key risk (from Gemini review):** The Apple bite is the enemy. If a black data module lands in the bite area, the silhouette is ruined. The geometry-aware mask optimizer (2.2) is critical.

---

## M3: Contour-Integrated QR (2 weeks)

Make QR modules follow the logo's curves instead of a rigid grid.

| # | Task |
|---|------|
| 3.1 | SDF (Signed Distance Field) generator for Apple logo — distance from every module center to nearest logo edge |
| 3.2 | **Conservative module warping** — shift module *edges* (not centers) near logo curves to follow the bite and leaf. Keep center sampling points within 40% of module width from grid position (beyond this, scanners misread). |
| 3.3 | Half-module decorative dots — fill gaps between square data modules and curved logo edge with non-functional sub-pixels that smooth the "staircase" effect |
| 3.4 | Edge smoothing buffer — white glow zone around logo prevents data modules from touching logo edges |
| 3.5 | **Sampling-point verification (strict)** — after warping, confirm every module's center point reads correct color with >70% contrast. Auto-revert any module that fails to standard square. |
| 3.6 | Multi-decoder scan-verify (ZBar, ZXing, iPhone camera, Android camera, "2018 budget phone" test) |

**Output:** QR code where modules visually curve around the Apple bite and leaf. Still backwards compatible.

**Key risk (from Gemini review):** Module warping tolerance is ~40% of module width max. Edge warping is safe; center displacement breaks scanners. The auto-revert (3.5) is the safety net.

---

## M4: AI-Blended QR Art (2 weeks)

Use AI to merge QR code and logo into unified artwork.

| # | Task |
|---|------|
| 4.1 | Set up Stable Diffusion + ControlNet QR model (QR Code Monster or QR Code Pattern v2) |
| 4.2 | Pipeline: input base QR (from M2) + Apple logo as style prompt → generate 50-100 artistic variations |
| 4.3 | **Auto-verify loop with multi-decoder** — run each variation through ZBar + ZXing + simulated phone camera, discard any that fail on ANY decoder |
| 4.4 | Similarity scoring — SSIM + perceptual hashing to rank by visual similarity to original Apple logo |
| 4.5 | **Brute-force bit-flipper** — for top candidates, flip individual ECC-safe bits to push visual similarity higher while maintaining scannability. This is the real hero — AI generates the art, bit-flipper ensures the math works. |
| 4.6 | Best-of-N selector — highest logo similarity that scans in < 400ms across all decoders |

**Output:** Image that looks like Apple logo artwork but is a fully functional QR code.

**Key risk (from Gemini review):** AI generation is non-deterministic. You may generate 1000 images with 0 valid scans. The bit-flipper (4.5) compensates but may destroy artistic quality. Expect high compute cost and many iterations.

---

## M5: Non-Square & Mosaic Codes (2 weeks)

Solve the "square QR on a non-square logo" problem.

| # | Task |
|---|------|
| 5.1 | **Structured Append mosaic** — use the QR standard's built-in "Structured Append" feature (up to 16 linked QR codes) instead of just parallel redundancy. Scanner knows "I need all 3 tiles for the full message." More robust than independent duplicates. |
| 5.2 | Also support parallel redundancy mode — 3-5 identical micro QR codes (V1) each containing the same URL, for maximum compatibility |
| 5.3 | Apply contour strategy (M3) per-tile so they blend into logo silhouette |
| 5.4 | Test with wide logos (text wordmarks), tall logos, and circular logos |
| 5.5 | Data Ribbon prototype — 100x4 non-square matrix encoded as thin strip (non-backwards-compatible, custom decoder) |

**Output:** Non-square QR layouts that follow any logo shape. Mosaic works on all phones.

---

## M6: Chroma & Grayscale Steganography (2 weeks)

Hide data in color/grayscale channels so the QR grid becomes invisible.

| # | Task |
|---|------|
| 6.1 | Chroma encoder — embed data bits in hue/saturation shifts (Delta-E < 3.0, invisible to humans) |
| 6.2 | **Hard calibration reference** — embed pure black + pure white control pixels in the quiet zone so the decoder can normalize the image before reading invisible color shifts. Don't rely on logo colors alone for white-balance. |
| 6.3 | **Bayer dithering data hiding** — hide data in the texture/high-frequency noise pattern of a grayscale image. Human eye smooths it out; camera sensor reads it clearly. More robust than simple color-channel shifts. |
| 6.4 | B&W halftone engine — 3x3 sub-module grid: center pixel = true data bit, surrounding 8 = grayscale dithering for logo texture |
| 6.5 | Dual-layer output: luminance ghost layer (legacy scanner compatible) + chroma layer (enhanced data for pro decoder) |
| 6.6 | **Print profile testing** — test on B&W laser, inkjet, and offset. Verify halftone survives ink bleed, dot gain, and low toner across 3+ printer types |
| 6.7 | **Lighting resilience** — test under yellow streetlight, blue office fluorescent, direct sunlight, and 20 lux dim conditions |

**Output:** Full-color Apple logo with invisibly embedded QR data. Also a B&W version for laser printers.

---

## M7: Lumen-Code — The Logo IS the Code (3 weeks)

Break backwards compatibility entirely. No visible QR grid.

| # | Task |
|---|------|
| 7.1 | Polar coordinate encoder — data as subtle "ripples" in logo contour path (r,theta displacements) |
| 7.2 | Stroke-width modulation — thick-to-thin transitions = bits |
| 7.3 | Contour frequency modulation — data as "waves" in stroke thickness |
| 7.4 | Train neural decoder (CNN or ViT): detect Apple logo → find coordinate origin → extract modulated data |
| 7.5 | Synthetic training data generator — thousands of Lumen-Code variants with known payloads |
| 7.6 | **WebAR decoder** — browser-based JS scanning so users don't need to install an app (critical for adoption; an app download kills the UX) |
| 7.7 | **Bootstrap strategy** — a standard QR code (from M2) leads to a WebAR page that then scans Lumen-Codes. Solves the chicken-and-egg: standard QR introduces users to the new format. |
| 7.8 | Test: decode latency < 200ms, +-30deg angles, survives blur and partial occlusion |

**Output:** Pristine Apple logo (no visible code) that a custom decoder reads as a URL. Looks like a 4K brand asset.

---

## M8: Data Fonts — Machine-Readable Typography (3 weeks)

The text "APPLE" itself carries the encoded URL.

| # | Task |
|---|------|
| 8.1 | DNA-ZIP compression engine — static dictionary + Base62 + quaternary encoding (>4:1 ratio on short URLs) |
| 8.2 | Design 3px/4px/5px data-carrying fonts — hide checksum pixels in glyph anatomy (counters, serifs, stem width variations) |
| 8.3 | Cross-character parity chain — Reed-Solomon across glyphs (letter A carries checksum for letter E) |
| 8.4 | **OpenType GSUB ligature computation** — when you type "AP", the font replaces it with a special ligature that has correct data dots pre-rendered for that pair. This is the hardest font engineering task. |
| 8.5 | B&W halftone data hiding within glyphs — survives laser printing |
| 8.6 | Font decoder SDK — camera → isolate font → extract data → verify checksums |
| 8.7 | **Inter-page document parity** — for multi-page B&W docs: each page footer carries metadata + Reed-Solomon parity for neighboring pages. Missing pages are reconstructable. |
| 8.8 | Demo: type "APPLE" in the data font → scanner reads `apple.co/store` from the pixels |

**Output:** A font where typing "APPLE" produces text humans read AND machines decode as a URL.

**Key risk (from Gemini review):** 3px font decode in real-world conditions (crumpled paper, phone camera, bad lighting) is research-level hard. 5px is the realistic starting point. OpenType GSUB may not support the complexity needed — may require a custom rendering engine.

---

## Summary: The Apple Logo Demo Progression

| Milestone | What the user sees | Backwards compatible? |
|-----------|-------------------|----------------------|
| M2 | Apple logo centered in a styled QR grid with shaped modules | Yes |
| M3 | QR modules curve around the Apple bite and leaf | Yes |
| M4 | Artistic image blending Apple logo with QR texture | Yes |
| M5 | Non-square layout following Apple wordmark shape | Yes (mosaic) / No (ribbon) |
| M6 | Full-color Apple logo with invisible embedded data | Partially (luminance layer) |
| M7 | Pure Apple logo, no visible code at all | No (WebAR decoder) |
| M8 | The word "APPLE" in a font that carries the URL | No (custom decoder) |

**M0-M4 (~8 weeks) deliver the most impressive backwards-compatible demo.**
M5-M8 are advanced research milestones with diminishing backwards compatibility but increasing "wow factor."

---

## Apple Logo-Specific Risks & Mitigations

| Risk | Why it matters | Mitigation |
|------|---------------|------------|
| The "bite" gets filled with black modules | Ruins the iconic silhouette | Geometry-aware mask optimizer (M2.2) penalizes modules in bite zone |
| The leaf is too small for valid modules | Scanner sees it as dust/noise | Use leaf as orientation anchor for custom decoders (M2.9) |
| Low contrast (gray-on-white Apple aesthetic) | Cheap cameras can't lock on | Dynamic contrast booster with subtle shadow (M2.6) |
| Finder patterns (3 corner squares) clash with logo | Looks ugly, breaks brand feel | Style them as frame elements with rounded corners and brand color (M2.4) |
| Module warping breaks scannability | Shifted centers misread as wrong bit | Cap displacement at 40% module width, auto-revert failures (M3.5) |
| AI generation produces 0 valid scans | Wasted compute, no output | Bit-flipper post-processing recovers scannability (M4.5) |
| No phones natively scan Lumen-Code | Zero adoption | WebAR bridge + standard QR bootstrap (M7.6, M7.7) |

---

## Technical Techniques Catalog

Techniques gathered from brainstorm + Gemini review, organized by category:

**Compression:**
- Base62 key encoding (9B IDs in 6 chars)
- Numeric-only QR mode (30% more efficient)
- Domain-specific dictionary (`apple.co/` → 4-bit token)
- DNA quaternary encoding (2 bits per pixel via 4 levels)

**QR Optimization:**
- Custom pad codeword injection (visual alignment before ECC)
- Mask pattern brute-forcing (8 patterns, pick best for logo)
- Structured Append (standard multi-QR linking)
- Module shape substitution (circles, hexagons, etc.)

**Visual Integration:**
- SDF-based contour warping
- Half-module decorative sub-pixels
- ControlNet QR AI blending
- Brute-force ECC-safe bit-flipping

**Steganography:**
- Chroma-steganography (Delta-E < 3.0 color shifts)
- Bayer dithering data hiding
- 3x3 sub-module halftone grid
- Luminance ghost layer + chroma data layer

**New Protocols:**
- Polar coordinate contour encoding
- Stroke-width modulation
- Contour frequency modulation
- Data-carrying typography with cross-glyph Reed-Solomon
