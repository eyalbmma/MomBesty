# Pre-publication checklist

Use this before making the repository public or widely sharing it. Items are ordered by severity within each section.

---

## 1. Must fix before public

| Issue | Why it matters | File(s) | Recommended fix |
|--------|----------------|---------|-------------------|
| Database SQL export may contain real or realistic user/chat/content data | Public repos retain history; health-adjacent text and identifiers can harm privacy and create legal/reputational risk | `rag.db.sql` | Remove the file from the repo or replace with **schema-only** DDL (no `INSERT` rows). If it was ever committed with sensitive data, consider `git filter-repo` or equivalent history rewrite, then rotate any exposed secrets. |
| Unauthenticated endpoint returns Expo push tokens | Anyone who can guess or obtain a `user_id` can read device push tokens → account/device targeting, notification abuse | `forum_api.py` — route `GET /forum/debug/push-tokens` (`debug_push_tokens`) | **Delete the route** or protect it behind strong authentication and admin-only access; ensure it is never enabled in production builds. |
| Default admin API key when env is unset | Deployments without `CIRCLES_ADMIN_KEY` use a predictable string; admin seed and CRUD become trivially callable | `circles_api.py` — `ADMIN_SEED_KEY` / `ADMIN_KEY` with `os.getenv("CIRCLES_ADMIN_KEY", "change-me")` | **Remove the default**: require `CIRCLES_ADMIN_KEY` at startup and fail fast if missing. Document that operators must set a long random secret. |
| No real authentication; client-supplied `user_id` is trusted | Impersonation, reading/modifying others’ forum posts, tracker data, and notifications | `forum_api.py`, `tracker_api.py`, and related handlers | For a **public** API surface: add proper auth (sessions, JWT, etc.) or **clearly scope** the repo as reference-only and **do not** deploy publicly without auth. At minimum, state this limitation prominently in `README.md` (already noted) and block public deploys until addressed. |
| OpenAI and admin secrets in environment only | Committed keys are a critical incident | Entire repo + git history | Run a secret scan (e.g. `gitleaks`, GitHub secret scanning) on the full history. Ensure `.env` is gitignored and never committed. |

---

## 2. Recommended cleanup

| Issue | Why it matters | File(s) | Recommended fix |
|--------|----------------|---------|-------------------|
| Hardcoded paths for non-DB inputs | Breaks other machines; exposes your local layout | `insert_curated.py` (`JSON_PATH`) | Use environment variables or paths relative to the repo / `pathlib` with documented defaults. |
| Duplicate or broken module | Confuses contributors and tooling; may be imported by mistake | `daily_support_repo.py` (references `now_ts`, `compute_day_index` without defining/importing them) | **Delete** the file or merge into `daily_support_plan.py` and fix imports; grep the repo to ensure nothing imports it. |
| Stray editor artifact | Noise and unprofessional in a public repo | `Untitled` | Delete the file. |
| Push token prefixes and full token lists logged | Logs often leave the host; tokens are credentials | `forum_api.py` (`[PUSH_TOKEN]`, `[PUSH_DEBUG]` prints including `tokens={tokens}`) | Log only non-sensitive metadata (e.g. user id hash, token count); redact or remove token substrings from logs in production. |
| `.gitignore` does not cover large SQL dumps | Risk of re-adding `rag.db.sql`-style files | `.gitignore` | Add patterns such as `*.sql` dumps you use for data only, or `exports/`, `backups/`, and document what belongs in repo vs local. |
| No dependency lockfile | Reproducible installs and security audits are harder | *(missing)* | Add `requirements.txt` (and optionally pin versions) generated from imports actually used by `api.py` and scripts you support. |
| Forum DDL vs API column usage may drift | Runtime errors or failed inserts after fresh DB create | `create_forum_tables.py` vs `forum_api.py` / `rag.db.sql` | Align `CREATE TABLE` with columns the API uses (e.g. `display_name` on posts/comments); add a single migration or setup script you run in CI. |
| `rag.db.sql` in tree while README warns about exports | Easy to accidentally widen exposure in a fork or PR | `rag.db.sql` | After sanitizing, prefer **schema-only** artifact with a name like `schema.sql` and keep large data out of git. |

---

## 3. Nice to have improvements

| Issue | Why it matters | File(s) | Recommended fix |
|--------|----------------|---------|-------------------|
| `DB_PATH` duplicated in many files | Drift risk if one file is updated and others are not | `api.py`, `content_api.py`, `forum_api.py`, `tracker_api.py`, `circles_api.py`, `embed_db.py`, `create_*_tables.py`, etc. | Centralize `DB_PATH` in one module or `os.getenv("DATABASE_PATH", "data/rag.db")` and document in README. |
| OpenAPI UI exposed by default | Expands attack surface and documents internal routes | FastAPI defaults on `api.py` | In production, disable or restrict `/docs`, `/redoc`, `/openapi.json` (middleware or `FastAPI(docs_url=None, ...)` when `ENV=production`). |
| Python version not pinned | CI and contributors may hit subtle incompatibilities | *(missing)* | Add `.python-version` or state `requires-python` in `pyproject.toml` if you adopt one. |
| RAG tables and `embed_db.py` unused by live chat | Confusion about what the product actually does | `api.py`, `embed_db.py`, `rag_clean` | Either **document** “offline only” (README already does) or implement retrieval in `/ask_final`; alternatively trim unused tables/scripts from the public repo if you want a minimal story. |
| No container or process manager config | Harder for others to run consistently | *(missing)* | Optional `Dockerfile` + `docker-compose` with volume for SQLite and env-file example (no real secrets). |
| Circles tables created only via seed/admin | Fresh clone may lack `CREATE TABLE` for circles | `circles_api.py` (inserts only) | Add `create_circles_tables.py` (or SQL migration) that matches seed column sets. |
| CORS allows broad Netlify pattern | Any `*.netlify.app` origin can call the API from a browser | `api.py` | Restrict to your known preview/production hostnames when going public. |

---

*Review this list after any major change to auth, forum push flow, or database layout.*
