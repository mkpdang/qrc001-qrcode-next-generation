# Show Image — Open in Windows Photo Viewer

Open an image file in the Windows Photos app (via WSL2 interop).

## Usage

When the user says `/show <path>` or `/show` (no args):

1. If a **path argument** is provided, use that file.
2. If **no argument**, look for the most recently modified `.png` file in `output/`.
3. Convert the WSL path to a Windows path using `wslpath -w`.
4. Open it with `explorer.exe` (which launches Windows Photos).

## Steps

```bash
# Convert WSL path to Windows path and open
WIN_PATH=$(wslpath -w "$FILE_PATH")
explorer.exe "$WIN_PATH"
```

Also display the image inline using the Read tool so the user can see it in the terminal too.

## Example

```
/show output/best_apple.png
```
