import requests
import json
import time
import os
from datetime import datetime
from typing import Tuple, List, Dict, Any


DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "X-IG-App-ID": "936619743392459",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "application/json, text/plain, */*",
}


def _fetch_profile_json(session: requests.Session, username: str, timeout: int = 20) -> Dict[str, Any] | None:
    endpoints = [
        f"https://i.instagram.com/api/v1/users/web_profile_info/?username={username}",
        f"https://www.instagram.com/api/v1/users/web_profile_info/?username={username}",
    ]
    backoff = 2
    for attempt in range(1, 5):
        for api_url in endpoints:
            try:
                headers = {
                    **DEFAULT_HEADERS,
                    "Referer": f"https://www.instagram.com/{username}/",
                    "Origin": "https://www.instagram.com",
                }
                session.headers.update(headers)
                resp = session.get(api_url, timeout=timeout)
                status = resp.status_code
                if status == 200:
                    try:
                        return resp.json()
                    except json.JSONDecodeError:
                        print(f"JSON decode failed on attempt {attempt} url {api_url}. First 200 chars: {resp.text[:200]}")
                elif status in (429, 400, 403, 404, 500, 502, 503):
                    print(f"HTTP {status} from {api_url} (attempt {attempt}).")
                    # 429: rate limited, backoff; 404: user not found, stop after trying both
                else:
                    print(f"Unexpected status {status} from {api_url} (attempt {attempt}).")
            except requests.exceptions.RequestException as e:
                print(f"Request error on {api_url} (attempt {attempt}): {e}")
        time.sleep(backoff)
        backoff = min(backoff * 2, 10)
    return None


def scrape_instagram_profile(username: str) -> Tuple[Dict[str, Any] | None, List[Dict[str, Any]] | None]:
    """
    Scrapes Instagram profile data and up to ~20 recent posts for a given username using
    public web endpoints with retries and fallbacks.
    Returns (profile_info, posts) or (None, None) on error.
    """
    print(f"Fetching data for @{username}...")
    session = requests.Session()

    data = _fetch_profile_json(session, username)
    if data is None:
        print("Failed to fetch profile JSON after retries.")
        return None, None

    try:
        user_data = data['data']['user']
    except Exception as e:
        print(f"Unexpected response structure. Missing user data: {e}. Keys: {list(data.keys())}")
        return None, None

    profile_info = {
        'id': user_data.get('id'),
        'username': user_data.get('username'),
        'full_name': user_data.get('full_name'),
        'biography': user_data.get('biography', ''),
        'profile_pic_url': user_data.get('profile_pic_url_hd') or user_data.get('profile_pic_url'),
        'followers_count': user_data.get('edge_followed_by', {}).get('count', 0),
        'following_count': user_data.get('edge_follow', {}).get('count', 0),
        'posts_count': user_data.get('edge_owner_to_timeline_media', {}).get('count', 0),
        'is_private': user_data.get('is_private', False),
        'is_verified': user_data.get('is_verified', False),
        'scraped_at': datetime.now().isoformat()
    }

    posts: List[Dict[str, Any]] = []
    edges = (user_data.get('edge_owner_to_timeline_media') or {}).get('edges', [])
    for index, edge in enumerate(edges):
        if index >= 20:
            break
        node = edge.get('node') or {}
        post_data = {
            'id': node.get('id'),
            'shortcode': node.get('shortcode'),
            'post_url': f"https://instagram.com/p/{node.get('shortcode')}/" if node.get('shortcode') else None,
            'thumbnail_src': node.get('thumbnail_src') or node.get('display_url'),
            'is_video': node.get('is_video', False),
            'likes_count': (node.get('edge_liked_by') or {}).get('count', 0),
            'comments_count': (node.get('edge_media_to_comment') or {}).get('count', 0),
            'caption': ((node.get('edge_media_to_caption') or {}).get('edges') or [{"node": {"text": ""}}])[0]['node'].get('text', ''),
            'taken_at_timestamp': node.get('taken_at_timestamp')
        }
        posts.append(post_data)
        time.sleep(0.5)

    return profile_info, posts


def save_data(profile_info: Dict[str, Any], posts: List[Dict[str, Any]], username: str, out_dir: str = 'instagram_data') -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f'{username}_profile.json'), 'w', encoding='utf-8') as f:
        json.dump(profile_info, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, f'{username}_posts.json'), 'w', encoding='utf-8') as f:
        json.dump(posts, f, indent=2, ensure_ascii=False)
    print(f"Saved JSON to '{out_dir}'")
