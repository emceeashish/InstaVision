from pathlib import Path
import json
from typing import Tuple, Dict, Any, List, Optional

from django.conf import settings

from core.scraper import scrape_instagram_profile, save_data

import os


def get_paths(username: str) -> Tuple[Path, Path, Path]:
    data_dir = Path(settings.DATA_DIR)
    profile = data_dir / f"{username}_profile.json"
    posts = data_dir / f"{username}_posts.json"
    analyzed = data_dir / f"{username}_analyzed.json"
    return profile, posts, analyzed


# ---------- Analysis helpers ----------

def analysis_exists(username: str) -> bool:
    _, _, analyzed_path = get_paths(username)
    return analyzed_path.exists()


def load_analyzed(username: str) -> Dict[str, Any]:
    _, _, analyzed_path = get_paths(username)
    with open(analyzed_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_scraped(username: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    profile_path, posts_path, _ = get_paths(username)
    if not profile_path.exists() or not posts_path.exists():
        raise FileNotFoundError("Scraped files not found")
    with open(profile_path, 'r', encoding='utf-8') as f:
        profile = json.load(f)
    with open(posts_path, 'r', encoding='utf-8') as f:
        posts = json.load(f)
    return profile, posts


def ensure_scraped(username: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    profile_path, posts_path, _ = get_paths(username)
    if profile_path.exists() and posts_path.exists():
        return load_scraped(username)

    profile, posts = scrape_instagram_profile(username)
    if not profile or posts is None:
        raise RuntimeError("Failed to scrape Instagram (profile may be private or rate-limited)")
    save_data(profile, posts, username, out_dir=str(settings.DATA_DIR))
    return profile, posts


def run_analysis_async(username: str) -> None:
    import threading

    def _task():
        try:
            from core.ai_pipeline import run_ai_analysis
            run_ai_analysis(username=username, data_folder=str(settings.DATA_DIR), output_folder=str(settings.DATA_DIR))
        except Exception:
            pass

    t = threading.Thread(target=_task, daemon=True)
    t.start()


def ensure_analyzed(username: str) -> Dict[str, Any]:
    profile_path, posts_path, analyzed_path = get_paths(username)

    if analyzed_path.exists():
        with open(analyzed_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    if not (profile_path.exists() and posts_path.exists()):
        profile, posts = scrape_instagram_profile(username)
        if not profile or posts is None:
            raise RuntimeError("Failed to scrape Instagram (profile may be private or rate-limited)")
        save_data(profile, posts, username, out_dir=str(settings.DATA_DIR))

    from core.ai_pipeline import run_ai_analysis
    analyzed = run_ai_analysis(username=username, data_folder=str(settings.DATA_DIR), output_folder=str(settings.DATA_DIR))
    if analyzed is None:
        raise RuntimeError("AI analysis failed")
    return analyzed
