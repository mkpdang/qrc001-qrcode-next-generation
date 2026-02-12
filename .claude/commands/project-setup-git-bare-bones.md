# Project Setup - Git Bare Bones

Set up a new project repository with bare-bones structure mirroring the standard three-component architecture.

## Steps

1. **Create GitHub repo** using `gh repo create <name> --public --description "<desc>" --clone`
2. **Rename branch** to `main`: `git branch -m master main`
3. **Ensure repo is at** `/home/ubuntu/<project-name>/main/` (matching the worktree pattern)

4. **Create root config files:**
   - `.gitignore` — vendor, node_modules, .env, __pycache__, client/dist, etc.
   - `.env.example` — app config, DB (MySQL), Redis, mail, Vite
   - `CLAUDE.md` — project rules, architecture overview, pipeline docs
   - `.mcp.json` — MCP server config (start empty `{ "mcpServers": {} }`)

5. **Set up Claude hooks** (notification sounds):
   - Copy `.claude/notify.py` from reference repo or create with 880Hz notify + 660Hz done tones
   - Add hooks to `.claude/settings.local.json`:
     ```json
     "hooks": {
       "Notification": [{ "matcher": "", "hooks": [{ "type": "command", "command": "python3 /home/ubuntu/<project>/main/.claude/notify.py notify &", "timeout": 5 }] }],
       "Stop": [{ "matcher": "", "hooks": [{ "type": "command", "command": "python3 /home/ubuntu/<project>/main/.claude/notify.py done &", "timeout": 5 }] }]
     }
     ```

6. **Create `client/` directory** (Vue.js/Vite frontend):
   - `package.json` — name, scripts (dev/build/preview), alpinejs, tailwindcss, vite
   - `vite.config.js` — tailwind plugin, base `/scanner/`, build to `../laravel-app/public/scanner`, proxy `/api` to localhost:8000
   - `index.html` — minimal HTML entry point
   - `src/main.js` — import style.css, init Alpine
   - `src/style.css` — `@import "tailwindcss";`

7. **Create `laravel-app/`** using `composer create-project laravel/laravel laravel-app`:
   - Remove nested `.git/` directory
   - Remove `vendor/`, `.env`, `database.sqlite` from git staging (keep in .gitignore)

8. **Create `notebook/` directory** (Python engine):
   - `pyproject.toml` — name, version, requires-python >=3.11, minimal deps (opencv, pillow, python-dotenv, requests)

9. **Create `specs/` directory** with `.gitkeep`

10. **Initial commit and push:**
    - `git add` all files (excluding vendor, .env, database.sqlite)
    - Commit with message: "Initial bare-bones project structure for <project-name>"
    - `git push -u origin main`

## Reference Repo

Structure is based on `/home/ubuntu/pas001-passport-ocr-api/main/`:
```
root/
├── .claude/          # Claude hooks & settings
├── .env.example      # Environment template
├── .gitignore
├── .mcp.json         # MCP server config
├── CLAUDE.md         # Project rules
├── client/           # Vue.js/Vite frontend
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   └── src/
├── laravel-app/      # Laravel 12 backend
│   ├── composer.json
│   ├── app/
│   ├── config/
│   ├── database/
│   ├── routes/
│   └── ...
├── notebook/         # Python engine
│   └── pyproject.toml
└── specs/            # Specifications
```
