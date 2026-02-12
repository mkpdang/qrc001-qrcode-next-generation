# Project Rules

## Architecture

This is a QR Code Next Generation project with three main components:

1. **client/** - Vue.js/Vite frontend (scanner UI)
2. **laravel-app/** - Laravel 12 backend API
3. **notebook/** - Python QR code engine & processing

## Dev Basic Auth

All endpoints are protected by HTTP Basic Auth in dev/staging. Credentials are in `.env`:
- Username: `DEV_BASIC_AUTH_USERNAME` (default: `admin`)
- Password: `DEV_BASIC_AUTH_PASSWORD` (default: `admin123123`)

API requests with `Authorization: Bearer` tokens skip basic auth (they use API key auth).
Health endpoint (`/api/health`) is exempt.

## QR Code Pipeline Architecture

- All QR code generation, scanning, and data extraction logic MUST be handled server-side in Python (`notebook/`). Never implement QR code processing in client-side JavaScript.
- The client (`client/src/`) is a display-only layer: it reads pre-computed data from the API and renders it.
- `notebook/` contains the QR code engine. Changes to scanning or generation go there.
