import json
import os
import requests
from PIL import Image
from io import BytesIO
import torch
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO
from piq import brisque
import cv2
import numpy as np
from tqdm import tqdm
import tempfile
import re
from typing import Tuple, List, Dict, Any


def safe_filename_from_url(url: str) -> str:
    name = re.sub(r'[^a-zA-Z0-9._-]+', '_', url.split('/')[-1])[:80]
    return name or "video.mp4"


def download_file(url: str, timeout: int = 15) -> str:
    r = requests.get(url, timeout=timeout, stream=True)
    r.raise_for_status()
    suffix = ".mp4" if ".mp4" in url.lower() else ""
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return path


def open_video_capture(source: str):
    cap = cv2.VideoCapture(source)
    if cap.isOpened():
        return cap
    if source.startswith("http"):
        try:
            tmp_path = download_file(source)
            cap = cv2.VideoCapture(tmp_path)
            if cap.isOpened():
                return cap
        except Exception:
            pass
    return None


def extract_keyframes_opencv(
    source: str,
    max_frames: int = 5,
    stride_seconds: float = 1.5,
    use_hist: bool = True,
    hist_thresh: float = 0.5
):
    cap = open_video_capture(source)
    if cap is None:
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(int(fps * stride_seconds), 1)

    frames = []
    prev_hist = None
    for idx in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if use_hist:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            if prev_hist is not None:
                diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                if diff < hist_thresh:
                    continue
            prev_hist = hist
        frames.append(Image.fromarray(rgb))
        if len(frames) >= max_frames:
            break

    cap.release()
    return frames


class InstagramAIAnalyzer:
    def __init__(self, device: str | None = None, clip_model_id: str = "openai/clip-vit-base-patch32"):
        print("Initializing AI Models...")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.clip_model = CLIPModel.from_pretrained(clip_model_id).to(self.device)
        self.clip_model.eval()
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_id)

        self.yolo_model = YOLO("yolov8n.pt")

        # Expanded categories for better coverage
        self.base_tag_categories = [
            # Scenes & settings
            "city skyline","street style","indoor room","studio backdrop","beach","mountains","forest","cafe","gym",
            # People & portraits
            "selfie","portrait","group photo","close-up","full body","candid","smile",
            # Fashion & lifestyle
            "fashion","outfit","makeup","accessories","watch","sunglasses","footwear","bag","luxury",
            # Activities & sports
            "travel","hiking","running","workout","dance","performance","party","wedding","graduation",
            # Food & objects
            "food","dessert","coffee","technology","gadget","car","motorcycle","animal","pet",
            # Art & media
            "art","painting","poster","music","microphone","stage",
            # Tone & purpose
            "minimalist","aesthetic","brand promo","product showcase","behind the scenes",
            # Social causes
            "charity","volunteer","ngo","community aid","relief","rescue","donation","awareness",
        ]
        self.vibe_categories = [
            "casual everyday","aesthetic artistic","luxury lavish","energetic dynamic","calm peaceful","dark moody",
            "bright vibrant","professional","party celebration","romantic","adventure","minimalist",
            "nostalgic","cinematic","warm cozy","cool modern"
        ]
        self.video_event_categories = [
            "person dancing","people talking","car moving","beach scene","city walking","eating food","working out","singing",
            "playing sports","driving","party celebration","performance","shopping","cooking food","traveling","celebrating"
        ]

        def prompt_labels(labels, template="This is a photo of a {}."):
            return [template.format(l) for l in labels]

        self.tag_prompts = prompt_labels(self.base_tag_categories, "This is a photo of {}.")
        self.vibe_prompts = prompt_labels(self.vibe_categories, "This is a photo with a {} vibe.")
        self.event_prompts = prompt_labels(self.video_event_categories, "This is a video frame of {}.")

        self.tag_text_inputs = self.clip_processor(text=self.tag_prompts, return_tensors="pt", padding=True).to(self.device)
        self.vibe_text_inputs = self.clip_processor(text=self.vibe_prompts, return_tensors="pt", padding=True).to(self.device)
        self.event_text_inputs = self.clip_processor(text=self.event_prompts, return_tensors="pt", padding=True).to(self.device)

        print(f"AI Models loaded on {self.device}!")

    def download_image(self, url):
        try:
            if not url:
                return None
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            return Image.open(BytesIO(r.content)).convert("RGB")
        except Exception as e:
            print(f"Error downloading image: {e}")
            return None

    @torch.no_grad()
    def _clip_probs_from_prompts(self, image: Image.Image, text_inputs, image_size: int = 384):
        pixel_inputs = self.clip_processor(images=image, return_tensors="pt", do_resize=True, size=image_size).to(self.device)
        outputs = self.clip_model(input_ids=text_inputs["input_ids"], attention_mask=text_inputs["attention_mask"], pixel_values=pixel_inputs["pixel_values"])
        probs = outputs.logits_per_image.softmax(dim=1)
        return probs[0].detach().cpu()

    def _select_multilabel(self, probs: torch.Tensor, labels: list[str], threshold: float = 0.18, top_k_fallback: int = 7):
        idx = (probs >= threshold).nonzero(as_tuple=True)[0].tolist()
        if not idx:
            idx = probs.topk(top_k_fallback).indices.tolist()
        return [labels[i] for i in idx]

    def _get_clip_tags(self, image: Image.Image, threshold: float = 0.18):
        probs = self._clip_probs_from_prompts(image, self.tag_text_inputs)
        return self._select_multilabel(probs, self.base_tag_categories, threshold=threshold, top_k_fallback=7)

    def _get_clip_vibe(self, image: Image.Image):
        probs = self._clip_probs_from_prompts(image, self.vibe_text_inputs)
        return self.vibe_categories[int(probs.argmax())]

    def _get_clip_events(self, image: Image.Image, threshold: float = 0.20):
        probs = self._clip_probs_from_prompts(image, self.event_text_inputs)
        return self._select_multilabel(probs, self.video_event_categories, threshold=threshold, top_k_fallback=3)

    def _get_quality_details(self, image: Image.Image):
        # Base BRISQUE score
        try:
            arr = np.asarray(image).astype(np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
            score = float(brisque(tensor, data_range=1.0))
            if score < 20:
                label = "excellent"
            elif score < 40:
                label = "good"
            elif score < 60:
                label = "average"
            else:
                label = "poor"
        except Exception as e:
            print(f"Error in quality assessment: {e}")
            score, label = 0.0, "unknown"

        # Heuristics for lighting/appeal/consistency using simple image stats
        try:
            gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
            brightness = float(np.mean(gray))
            contrast = float(np.std(gray))
            lighting = "well-lit" if brightness > 140 else "dim" if brightness < 80 else "balanced"
            appeal = "high" if (label in ["excellent","good"] and contrast > 50) else "medium" if contrast > 30 else "low"
            # Consistency proxy: edge density
            edges = cv2.Canny(gray, 100, 200)
            edge_density = float(edges.mean())
            consistency = "clean" if edge_density < 10 else "detailed" if edge_density < 25 else "busy"
        except Exception:
            lighting, appeal, consistency = "unknown","unknown","unknown"

        return {"score": score, "label": label, "lighting": lighting, "visual_appeal": appeal, "consistency": consistency}

    def _get_objects(self, image: Image.Image, conf: float = 0.5):
        try:
            results = self.yolo_model.predict(image, conf=conf, verbose=False)
            objects: List[str] = []
            for res in results:
                names = res.names
                for box in res.boxes:
                    if float(box.conf[0]) >= conf:
                        cls = int(box.cls[0])
                        objects.append(names.get(cls, str(cls)))
            return sorted(set(objects))
        except Exception as e:
            print(f"Error in object detection: {e}")
            return []

    def analyze_image_post(self, image_url: str):
        image = self.download_image(image_url)
        if image is None:
            return None
        return {
            "tags": self._get_clip_tags(image, threshold=0.18),
            "vibe": self._get_clip_vibe(image),
            "quality": self._get_quality_details(image),
            "objects": self._get_objects(image),
            "media_type": "image"
        }

    def analyze_video_post(self, video_source: str | None, thumbnail_url: str | None, max_keyframes: int = 4):
        if thumbnail_url:
            analysis = self.analyze_image_post(thumbnail_url)
            if analysis:
                analysis["media_type"] = "video"
                analysis["analysis_note"] = "thumbnail_only"
            return analysis
        return None


def load_json_data(username: str, data_folder: str = "instagram_data") -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    profile_file = os.path.join(data_folder, f"{username}_profile.json")
    posts_file = os.path.join(data_folder, f"{username}_posts.json")

    if not os.path.exists(profile_file):
        raise FileNotFoundError(f"Profile file {profile_file} not found")
    if not os.path.exists(posts_file):
        raise FileNotFoundError(f"Posts file {posts_file} not found")

    with open(profile_file, "r", encoding="utf-8") as f:
        profile_data = json.load(f)
    with open(posts_file, "r", encoding="utf-8") as f:
        posts_data = json.load(f)
    return profile_data, posts_data


def run_ai_analysis(username: str, data_folder: str = "instagram_data", output_folder: str = "instagram_data") -> Dict[str, Any] | None:
    print(f"Starting AI Analysis for @{username}...")

    try:
        profile_data, posts_data = load_json_data(username, data_folder)
        print(f"Loaded {len(posts_data)} posts for analysis")
    except Exception as e:
        print(f"Failed to load data: {e}")
        return None

    analyzer = InstagramAIAnalyzer()

    analyzed_posts: List[Dict[str, Any]] = []
    success_count = 0

    for post in tqdm(posts_data, desc="Analyzing posts"):
        try:
            if post.get("is_video"):
                # Thumbnail-only analysis for videos
                analysis = analyzer.analyze_video_post(video_source=None, thumbnail_url=post.get("thumbnail_src"))
            else:
                analysis = analyzer.analyze_image_post(post.get("thumbnail_src"))

            if analysis:
                post["ai_analysis"] = analysis
                success_count += 1
            analyzed_posts.append(post)
        except Exception as e:
            print(f"Error analyzing post {post.get('id')}: {e}")
            analyzed_posts.append(post)

    output_data: Dict[str, Any] = {
        "profile": profile_data,
        "posts": analyzed_posts,
        "analysis_summary": {
            "total_posts": len(posts_data),
            "successful_analysis": success_count,
            "success_rate": f"{(success_count / max(1, len(posts_data))) * 100:.1f}%"
        }
    }

    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f"{username}_analyzed.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"AI Analysis completed: {output_file}")
    return output_data
