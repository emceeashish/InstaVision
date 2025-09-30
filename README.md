## Instagram Influencer Profile - Scraper, AI Analysis, Django API, Streamlit UI

### Prerequisites
- Python 3.10+
- Windows 10/11 (or macOS/Linux)
- Recommended: Create a virtualenv

### Setup
```bash
setup_venv.bat
```

### Project Structure
```
insta_anal/
  core/                      # reusable scraping + AI analysis logic
  scripts/                   # CLI wrappers for manual runs
  backend/                   # Django + DRF API
  frontend/                  # Streamlit app
  instagram_data/            # JSON data output (profile, posts, analyzed)
```

### Quickstart (CLI)
```bash
# Scrape data
python scripts/scrape.py --username virat.kohli

# Run AI analysis
python scripts/analyze.py --username virat.kohli

# Or do both
python scripts/scrape_and_analyze.py --username virat.kohli
```

### Run Django API (Windows)
```bash
run_backend.bat
```
Then open `http://localhost:8000/api/profile/virat.kohli/` or `http://localhost:8000/api/posts/virat.kohli/`.

If analyzed data does not exist, the API will scrape + analyze on-demand (first request may take time to download models and run inference).

### Run Streamlit UI (Windows)
```bash
run_frontend.bat
```
Then open `http://localhost:8501` in your browser.

- Enter any Instagram username.
- The app fetches `profile` and `posts` from the Django API and displays:
  - Profile card (followers, following, posts, bio, profile picture)
  - Last 10 posts (images)
  - Last 5 reels (videos)
  - AI analysis tags, vibe, quality, objects, events per item
  - Charts for likes vs comments, vibe distribution, top tags/objects, quality labels

### Notes
- Heavy ML models (CLIP, YOLO) will download on the first run.
- If video URLs are not directly available, analysis falls back to thumbnails.
- Data is stored under `instagram_data/` as `<username>_profile.json`, `<username>_posts.json`, and `<username>_analyzed.json`.
- For private or blocked accounts, scraping may fail and API will return an error with details.
