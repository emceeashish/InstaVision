import os
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from typing import Dict, Any, List

st.set_page_config(page_title="Instagram Influencer Insights", layout="wide")

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def proxied(url: str | None) -> str | None:
    if not url:
        return None
    return f"{API_BASE_URL}/api/image-proxy/?url={requests.utils.quote(url, safe='')}"


# ----- Streamlit compatibility helpers -----

def show_image(container, src: str | None, width: int | None = None):
    if not src:
        return
    try:
        if width is not None:
            return container.image(src, width=width)
        return container.image(src, use_container_width=True)
    except TypeError:
        if width is not None:
            return container.image(src, width=width)
        return container.image(src, use_column_width=True)


def format_compact(num: int | float | None) -> str:
    try:
        n = float(num or 0)
    except Exception:
        return "0"
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return f"{int(n):,}"


def inject_css():
    if st.session_state.get("_css_injected"):
        return
    st.session_state["_css_injected"] = True
    st.markdown(
        """
        <style>
        .chip { display:inline-block; padding:2px 10px; border-radius:999px; margin:2px; font-size:12px; border:1px solid #374151; background:#1f2937; color:#e5e7eb; }
        .chip.vibe { background:#0ea5e9; border-color:#0284c7; color:#fff; }
        .chip.quality { background:#10b981; border-color:#059669; color:#fff; }
        .card-meta { font-size:12px; color:#9ca3af; margin-top:6px; }
        .post-link { font-size:13px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# Utility to coerce ai_analysis to dict safely

def safe_ai(analysis):
    if isinstance(analysis, dict):
        return analysis
    return {}


@st.cache_data(show_spinner=False)
def fetch_profile(username: str) -> Dict[str, Any] | None:
    try:
        url = f"{API_BASE_URL}/api/profile/{username}/"
        r = requests.get(url, timeout=120)
        if r.status_code == 200:
            return r.json()
        r = requests.get(url, timeout=180)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def fetch_posts(username: str, limit: int | None = None, media_type: str | None = None) -> Dict[str, Any] | None:
    try:
        params = {}
        if limit:
            params["limit"] = str(limit)
        if media_type in ("image", "video"):
            params["media_type"] = media_type
        url = f"{API_BASE_URL}/api/posts/{username}/"
        r = requests.get(url, params=params, timeout=300)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None


def normalize_posts(posts: List[Dict[str, Any]]) -> pd.DataFrame:
    if not posts:
        return pd.DataFrame([])
    df = pd.DataFrame(posts)
    for col in ["likes_count", "comments_count"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    if "is_video" not in df.columns:
        df["is_video"] = False
    else:
        df["is_video"] = df["is_video"].fillna(False).astype(bool)
    if "taken_at_timestamp" in df.columns:
        df["taken_at"] = pd.to_datetime(df["taken_at_timestamp"], unit="s", errors="coerce")
    else:
        df["taken_at"] = pd.NaT

    # Ensure ai_analysis column exists and is dict-like
    if "ai_analysis" not in df.columns:
        df["ai_analysis"] = [{} for _ in range(len(df))]

    ai_cols = {
        "ai_vibe": lambda a: safe_ai(a).get("vibe"),
        "ai_quality_score": lambda a: (safe_ai(a).get("quality") or {}).get("score"),
        "ai_quality_label": lambda a: (safe_ai(a).get("quality") or {}).get("label"),
        "ai_tags": lambda a: safe_ai(a).get("tags"),
        "ai_objects": lambda a: safe_ai(a).get("objects"),
        "ai_events": lambda a: safe_ai(a).get("events"),
        "ai_media_type": lambda a: safe_ai(a).get("media_type"),
    }
    for new_col, fn in ai_cols.items():
        df[new_col] = df["ai_analysis"].apply(fn)

    return df


def render_profile_header(profile: Dict[str, Any], summary: Dict[str, Any], df_all: pd.DataFrame):
    inject_css()
    left, mid, right = st.columns([1, 3, 3])
    with left:
        show_image(st, proxied(profile.get("profile_pic_url")), width=110)
    with mid:
        full_name = profile.get('full_name') or profile.get('username', '')
        username = profile.get('username', '')
        st.markdown(f"### {full_name}")
        st.caption(f"@{username}")
        if profile.get("biography"):
            st.caption(profile.get("biography"))
    with right:
        m1, m2, m3 = st.columns(3)
        m1.metric("Followers", format_compact(profile.get('followers_count')))
        m2.metric("Following", format_compact(profile.get('following_count')))
        m3.metric("Posts", format_compact(profile.get('posts_count')))

    # Second row: engagement metrics
    likes_avg = float(df_all["likes_count"].mean()) if not df_all.empty else 0.0
    comments_avg = float(df_all["comments_count"].mean()) if not df_all.empty else 0.0
    followers = int(profile.get("followers_count") or 0)
    eng_rate = ((likes_avg + comments_avg) / max(1, followers)) * 100.0 if followers else 0.0

    m1, m2, m3 = st.columns(3)
    m1.metric("Avg likes/post", format_compact(int(round(likes_avg))))
    m2.metric("Avg comments/post", format_compact(int(round(comments_avg))))
    m3.metric("Engagement rate", f"{eng_rate:.2f}%")


def render_chips(tags: List[str] | None, vibe: str | None, quality: str | None):
    inject_css()
    chips = []
    if tags:
        for t in tags[:6]:
            chips.append(f"<span class='chip'>{t}</span>")
    if vibe:
        chips.append(f"<span class='chip vibe'>vibe: {vibe}</span>")
    if quality:
        chips.append(f"<span class='chip quality'>quality: {quality}</span>")
    if chips:
        html = "<div class='chip-row'>" + " ".join(chips) + "</div>"
        html += "<div style='clear:both; height:0;'></div>"
        st.markdown(html, unsafe_allow_html=True)


def posts_grid(items: List[Dict[str, Any]], cols: int = 5):
    if not items:
        st.info("No posts available.")
        return
    rows = (len(items) + cols - 1) // cols
    idx = 0
    for _ in range(rows):
        columns = st.columns(cols)
        for c in columns:
            if idx >= len(items):
                break
            post = items[idx]
            thumb = proxied(post.get("thumbnail_src"))
            url = post.get("post_url")
            likes = format_compact(post.get("likes_count", 0))
            comments = format_compact(post.get("comments_count", 0))
            ai = safe_ai(post.get("ai_analysis"))
            show_image(c, thumb)
            c.markdown(f"<div class='card-meta'>❤️ {likes}&nbsp;&nbsp;💬 {comments}</div>", unsafe_allow_html=True)
            if url:
                c.markdown(f"[Open ↗]({url})")
            with c.container():
                render_chips(ai.get("tags"), ai.get("vibe"), (ai.get("quality") or {}).get("label"))
            # Use zero-width spaces to make labels unique without showing an id
            zws = "\u200B" * (idx % 7)
            label = f"Caption{zws}"
            with c.expander(label, expanded=False, key=f"caption_post_{idx}"):
                caption = post.get("caption") or "No caption"
                st.write(caption)
            idx += 1


def reels_grid(items: List[Dict[str, Any]], cols: int = 5):
    if not items:
        st.info("No reels available.")
        return
    rows = (len(items) + cols - 1) // cols
    idx = 0
    for _ in range(rows):
        columns = st.columns(cols)
        for c in columns:
            if idx >= len(items):
                break
            post = items[idx]
            thumb = proxied(post.get("thumbnail_src"))
            url = post.get("post_url")
            likes = format_compact(post.get("likes_count", 0))
            comments = format_compact(post.get("comments_count", 0))
            ai = safe_ai(post.get("ai_analysis"))
            show_image(c, thumb)
            c.markdown(f"<div class='card-meta'>❤️ {likes}&nbsp;&nbsp;💬 {comments}</div>", unsafe_allow_html=True)
            if url:
                c.markdown(f"[Open ↗]({url})")
            with c.container():
                render_chips(ai.get("tags"), ai.get("vibe"), (ai.get("quality") or {}).get("label"))
            zws = "\u200B" * (idx % 7)
            label = f"Caption{zws}"
            with c.expander(label, expanded=False, key=f"caption_reel_{idx}"):
                caption = post.get("caption") or "No caption"
                st.write(caption)
            idx += 1


def analytics_section(df: pd.DataFrame):
    if df.empty:
        st.info("No data to analyze.")
        return

    tab1, tab2, tab3, tab4 = st.tabs(["Engagement", "Vibes", "Tags & Objects", "Quality"])

    with tab1:
        scatter = px.scatter(
            df,
            x="likes_count",
            y="comments_count",
            color=df["is_video"].map({True: "video", False: "image"}),
            labels={"color": "type"},
            title="Likes vs Comments"
        )
        st.plotly_chart(scatter, use_container_width=True)

    with tab2:
        vibe_counts = df["ai_vibe"].dropna().value_counts().reset_index()
        vibe_counts.columns = ["vibe", "count"]
        st.plotly_chart(px.bar(vibe_counts, x="vibe", y="count", title="Vibe distribution"), use_container_width=True)

    with tab3:
        def explode_list_col(series):
            values = []
            for x in series.dropna().tolist():
                if isinstance(x, list):
                    values.extend(x)
            return pd.Series(values)
        tag_series = explode_list_col(df["ai_tags"]) if "ai_tags" in df.columns else pd.Series([])
        obj_series = explode_list_col(df["ai_objects"]) if "ai_objects" in df.columns else pd.Series([])
        col1, col2 = st.columns(2)
        with col1:
            if not tag_series.empty:
                top_tags = tag_series.value_counts().head(15).reset_index()
                top_tags.columns = ["tag", "count"]
                st.plotly_chart(px.bar(top_tags, x="tag", y="count", title="Top tags"), use_container_width=True)
            else:
                st.info("No tags available.")
        with col2:
            if not obj_series.empty:
                top_objs = obj_series.value_counts().head(15).reset_index()
                top_objs.columns = ["object", "count"]
                st.plotly_chart(px.bar(top_objs, x="object", y="count", title="Top objects"), use_container_width=True)
            else:
                st.info("No objects available.")

    with tab4:
        if "ai_quality_label" in df.columns and df["ai_quality_label"].notna().any():
            q_counts = df["ai_quality_label"].dropna().value_counts().reset_index()
            q_counts.columns = ["quality", "count"]
            st.plotly_chart(px.bar(q_counts, x="quality", y="count", title="Quality labels"), use_container_width=True)
        else:
            st.info("No quality scores available.")


# Sidebar controls
st.sidebar.header("Influencer Lookup")
username_input = st.sidebar.text_input("Instagram username", value=st.session_state.get("username", ""))
fetch_clicked = st.sidebar.button("Fetch data")

if fetch_clicked and username_input:
    st.session_state["username"] = username_input.strip()

username = st.session_state.get("username")

if not username:
    st.title("Instagram Influencer Insights")
    st.info("Enter a username in the left sidebar and click Fetch data.")
    st.stop()

with st.spinner("Loading profile and posts..."):
    profile_payload = fetch_profile(username)
    posts_payload = fetch_posts(username, limit=None, media_type=None)

if not profile_payload:
    st.error("Failed to load profile. The account may be private or rate-limited, or the backend is not running.")
    st.stop()

profile = profile_payload.get("profile", {})
posts = (posts_payload or {}).get("posts", [])
df_all = normalize_posts(posts)

# Header with metrics
render_profile_header(profile, profile_payload.get("analysis_summary", {}), df_all)

# View switcher
try:
    view = st.segmented_control("View", options=["Posts", "Reels", "Analytics"], default="Posts")
except Exception:
    view = st.radio("View", options=["Posts", "Reels", "Analytics"], index=0, horizontal=True)

if view == "Posts":
    images_df = df_all[df_all["is_video"] == False].sort_values(by="taken_at", ascending=False)
    posts_grid(images_df.to_dict("records")[:15], cols=5)
elif view == "Reels":
    reels_df = df_all[df_all["is_video"] == True].sort_values(by="taken_at", ascending=False)
    reels_grid(reels_df.to_dict("records")[:15], cols=5)
else:
    analytics_section(df_all)
