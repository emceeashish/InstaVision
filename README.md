## Instagram Influencer Insights

Scraper + AI/ML analysis + Django REST API + Streamlit UI.

### What this does
- Scrapes Instagram public profile and recent posts (up to ~20) without login
- Runs lightweight AI/ML on each post (images + video thumbnails):
  - Tags (scene, activity, lifestyle, tone, cause, etc.)
  - Vibe/ambience (casual, cinematic, cozy, etc.)
  - Quality: BRISQUE score + lighting, visual appeal, consistency
  - Objects (YOLOv8n)
- Serves analyzed JSON via a Django REST API
- Streamlit app to browse profile, posts, reels (thumbnails), and analytics

### Requirements
- Windows 10/11 (tested), Python 3.10+
- GPU optional; works on CPU
- First run downloads ML weights (CLIP, YOLO) – takes a few minutes

### Quick start
```
# 1) Install deps
pip install --upgrade pip
pip install -r requirements.txt

# 2) Run backend (Django)
run_backend.bat
# API will be at http://localhost:8000

# 3) Run frontend (Streamlit)
run_frontend.bat
# UI at http://localhost:8501
```

### CLI utilities
```
# Scrape profile and posts
python scripts/scrape.py --username virat.kohli

# Run AI analysis on scraped data
python scripts/analyze.py --username virat.kohli

# Both in one shot
python scripts/scrape_and_analyze.py --username virat.kohli
```
Analyzed JSON is saved to `instagram_data/<username>_analyzed.json`.

### API endpoints
- `GET /api/profile/<username>/` – returns profile + (if present) overall analysis summary; triggers AI analysis in background if missing
- `GET /api/posts/<username>/?limit=N&media_type=image|video` – returns posts array with `ai_analysis` per item; generates analysis on-demand

### Streamlit features
- Search any username
- Header metrics: followers, following, posts + avg likes, avg comments, engagement rate
- Tabs: Posts | Reels | Analytics
- Per-post chips: tags, vibe, quality label; caption expander per card
- Analytics charts: likes vs comments, vibe distribution, top tags/objects, quality labels

### Design choices & assumptions
- No login is used by default. Scraper leverages Instagram’s public web profile endpoint. If an account is private/blocked, scraping fails gracefully.
- Video analysis uses thumbnails only (fast and reliable). Reels are shown as images; AI works on thumbnails (consistent across profiles and avoids rate limits).
- Tagging uses CLIP with an expanded vocabulary (scenes, lifestyle, activities, tone, social-cause). It’s zero-shot; we trade a bit of precision for coverage and speed.
- Quality is a combination of BRISQUE and simple heuristics (lighting/appeal/consistency). It’s meant for ranking/comparison rather than lab-grade scoring.
- Object detection uses YOLOv8n for speed. It’s conservative (0.5 conf) and returns deduplicated objects.

### Handling missing data
- Backend: guards for missing keys/fields; returns 502 with a helpful message if scrape fails
- Frontend: robust to NaN/None – `ai_analysis` is safely coerced; no crashes if columns are missing
- If first request takes long (model downloads), Streamlit timeouts are increased

### Troubleshooting
- First-run slow: allow time for model/weights download
- Private/blocked profiles: API returns a friendly error; try another username
- Streamlit errors like `segmented_control` or image width:
  - The app falls back to a radio switcher and handles older Streamlit APIs automatically
- If you see “float has no attribute get”: pull latest; frontend now guards `ai_analysis` properly

### Project structure
```
insta_anal/
  backend/            # Django + DRF API
  core/               # Scraper and AI analysis
  frontend/           # Streamlit UI
  scripts/            # CLI wrappers
  instagram_data/     # JSON outputs
```

### Security and rate limits
- No credentials stored; public-only scraping
- Respectful pacing and limited post count
- Avoids aggressive video fetching; uses thumbnails for reliability

### Roadmap (nice-to-have)
- Account-level historical engagement trends
- Per-post detail route/page inside Streamlit
- Export CSV of analytics

### License
MIT (feel free to adapt for your own research/portfolio)
