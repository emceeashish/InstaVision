import json
import os
import requests
from PIL import Image
from io import BytesIO
import cv2
import numpy as np
from tqdm import tqdm
import tempfile
import re
from typing import Tuple, List, Dict, Any, Optional
import onnxruntime as ort
import pathlib


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
    def __init__(self, device: str | None = None, clip_model_id: str | None = None):
        print("Initializing lightweight AI pipeline...")
        # Minimize CPU thread usage and memory for cv2/NumPy/ONNXRuntime on Render free plan
        try:
            cv2.setNumThreads(1)
        except Exception:
            pass
        try:
            cv2.ocl.setUseOpenCL(False)
        except Exception:
            pass

        # Detection thresholds (tunable via env)
        self.det_conf_thresh: float = float(os.getenv("DET_CONF_THRESH", "0.25"))
        self.det_iou_thresh: float = float(os.getenv("DET_IOU_THRESH", "0.45"))
        self.allow_autodownload: bool = os.getenv("ALLOW_AUTODOWNLOAD", "1") not in ("0", "false", "False")

        # COCO class names (80 classes)
        self.coco_classes: List[str] = [
            "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
            "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
            "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
            "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
            "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
            "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed",
            "dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
            "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
        ]

        # Try to initialize ONNX Runtime YOLOv8n INT8 session
        self.ort_session: Optional[ort.InferenceSession] = None
        self._onnx_input_name: Optional[str] = None
        self._onnx_img_size: int = int(os.getenv("YOLOV8_ONNX_INPUT", "640"))
        models_dir = os.path.join("models")
        os.makedirs(models_dir, exist_ok=True)
        yolo_path = os.getenv("YOLOV8_ONNX_PATH", os.path.join(models_dir, "yolov8n_int8.onnx"))
        if not os.path.exists(yolo_path) and self.allow_autodownload:
            # If INT8 not provided, fallback to FP32 yolov8n.onnx from Ultralytics assets
            fallback_yolo = os.path.join(models_dir, "yolov8n.onnx")
            yolo_url = os.getenv("YOLOV8_ONNX_URL", "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx")
            try:
                self._download_file(yolo_url, fallback_yolo, max_mb=30)
                yolo_path = fallback_yolo
                print(f"Downloaded YOLO ONNX to {yolo_path}")
            except Exception as e:
                print(f"Autodownload YOLO ONNX failed: {e}")

        if os.path.exists(yolo_path):
            try:
                self._init_onnx_session(yolo_path)
                print("ONNX Runtime YOLOv8 session initialized")
            except Exception as e:
                print(f"Failed to init ONNX Runtime session: {e}")
        else:
            print(f"YOLO ONNX model not found at {yolo_path}. Object tags/objects will be empty.")

        # Optional OpenCV DNN SSD MobileNet fallback
        self.cv_dnn: Optional[cv2.dnn_DetectionModel] = None
        ssd_model = os.getenv("OPENCV_SSD_MODEL_PATH", "")
        ssd_cfg = os.getenv("OPENCV_SSD_CONFIG_PATH", "")
        if not self.ort_session and ssd_model and ssd_cfg and os.path.exists(ssd_model) and os.path.exists(ssd_cfg):
            try:
                net = cv2.dnn.readNetFromTensorflow(ssd_model, ssd_cfg)
                model = cv2.dnn_DetectionModel(net)
                model.setInputSize(320, 320)
                model.setInputScale(1.0 / 127.5)
                model.setInputMean((127.5, 127.5, 127.5))
                model.setInputSwapRB(True)
                self.cv_dnn = model
                print("OpenCV DNN SSD fallback initialized")
            except Exception as e:
                print(f"Failed to init OpenCV DNN SSD: {e}")

        # BRISQUE model paths (optional)
        self.brisque_model_path = os.getenv("BRISQUE_MODEL_PATH", os.path.join(models_dir, "brisque_model_live.yml"))
        self.brisque_range_path = os.getenv("BRISQUE_RANGE_PATH", os.path.join(models_dir, "brisque_range_live.yml"))
        if not (os.path.exists(self.brisque_model_path) and os.path.exists(self.brisque_range_path)):
            if self.allow_autodownload:
                try:
                    self._download_file(
                        os.getenv(
                            "BRISQUE_MODEL_URL",
                            "https://raw.githubusercontent.com/opencv/opencv_contrib/4.x/modules/quality/samples/brisque_model_live.yml"
                        ),
                        self.brisque_model_path,
                        max_mb=2,
                    )
                    self._download_file(
                        os.getenv(
                            "BRISQUE_RANGE_URL",
                            "https://raw.githubusercontent.com/opencv/opencv_contrib/4.x/modules/quality/samples/brisque_range_live.yml"
                        ),
                        self.brisque_range_path,
                        max_mb=2,
                    )
                except Exception as e:
                    print(f"Autodownload BRISQUE assets failed: {e}")
        if not (os.path.exists(self.brisque_model_path) and os.path.exists(self.brisque_range_path)):
            # If still not present, heuristics will be used
            self.brisque_model_path = None
            self.brisque_range_path = None
        print("Lightweight AI ready!")

    def _init_onnx_session(self, model_path: str):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CPUExecutionProvider"]
        self.ort_session = ort.InferenceSession(model_path, sess_options=so, providers=providers)
        self._onnx_input_name = self.ort_session.get_inputs()[0].name

    def _download_file(self, url: str, dst_path: str, timeout: int = 30, max_mb: int | None = None) -> None:
        os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
        resp = requests.get(url, stream=True, timeout=timeout)
        resp.raise_for_status()
        size = 0
        with open(dst_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                f.write(chunk)
                size += len(chunk)
                if max_mb is not None and size > max_mb * 1024 * 1024:
                    raise RuntimeError(f"Download exceeded {max_mb} MB limit")
        # Basic sanity check
        if os.path.getsize(dst_path) == 0:
            raise RuntimeError("Downloaded file is empty")

    def _letterbox(self, img: np.ndarray, new_size: int = 640, color: Tuple[int, int, int] = (114, 114, 114)) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        h, w = img.shape[:2]
        r = min(new_size / h, new_size / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        top = (new_size - nh) // 2
        bottom = new_size - nh - top
        left = (new_size - nw) // 2
        right = new_size - nw - left
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return padded, r, (left, top)

    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep: List[int] = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            inds = np.where(iou <= iou_thr)[0]
            order = order[inds + 1]
        return keep

    def _detect_with_onnx(self, image: Image.Image, conf: Optional[float] = None) -> List[Tuple[str, float]]:
        if not self.ort_session:
            return []
        conf_thr = float(conf if conf is not None else self.det_conf_thresh)
        img = np.asarray(image)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        size = self._onnx_img_size
        padded, r, (left, top) = self._letterbox(img, new_size=size)
        padded_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        inp = padded_rgb.astype(np.float32) / 255.0
        inp = np.transpose(inp, (2, 0, 1))[None, ...]
        outputs = self.ort_session.run(None, {self._onnx_input_name: inp})
        pred = outputs[0]
        if pred.ndim == 3:
            pred = np.squeeze(pred, axis=0)
        # Expected shapes: (8400, 84) or (84, 8400)
        if pred.shape[0] in (84, 85):
            pred = pred.transpose(1, 0)
        # Now pred shape: (num, 84/85)
        num_classes = pred.shape[1] - 4
        boxes = pred[:, :4]
        if boxes.max() <= 1.5:
            # If center format (cx, cy, w, h)
            cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            x1 = cx - bw / 2
            y1 = cy - bh / 2
            x2 = cx + bw / 2
            y2 = cy + bh / 2
            boxes = np.stack([x1, y1, x2, y2], axis=1)
        cls_scores = pred[:, 4:4 + num_classes]
        scores = cls_scores.max(axis=1)
        cls_ids = cls_scores.argmax(axis=1)
        mask = scores >= conf_thr
        boxes = boxes[mask]
        scores = scores[mask]
        cls_ids = cls_ids[mask]
        if boxes.size == 0:
            return []
        # Scale boxes back to original image
        boxes[:, [0, 2]] -= left
        boxes[:, [1, 3]] -= top
        boxes /= r
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, img.shape[1])
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, img.shape[0])
        keep = self._nms(boxes, scores, self.det_iou_thresh)
        names: List[Tuple[str, float]] = []
        for i in keep:
            cid = int(cls_ids[i])
            name = self.coco_classes[cid] if 0 <= cid < len(self.coco_classes) else str(cid)
            names.append((name, float(scores[i])))
        return names

    def _detect_with_opencv_ssd(self, image: Image.Image, conf: Optional[float] = None) -> List[Tuple[str, float]]:
        if not self.cv_dnn:
            return []
        conf_thr = float(conf if conf is not None else self.det_conf_thresh)
        img = np.asarray(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        classes, scores, boxes = self.cv_dnn.detect(img, confThreshold=conf_thr, nmsThreshold=self.det_iou_thresh)
        results: List[Tuple[str, float]] = []
        for cid, score in zip(classes.flatten().tolist(), scores.flatten().tolist()):
            idx = int(cid) - 1  # SSD labels often start at 1
            name = self.coco_classes[idx] if 0 <= idx < len(self.coco_classes) else str(idx)
            results.append((name, float(score)))
        return results

    def detect_objects_tags(self, image: Image.Image, conf: float = 0.35) -> List[str]:
        try:
            dets = self._detect_with_onnx(image, conf=conf)
            if not dets:
                dets = self._detect_with_opencv_ssd(image, conf=conf)
            names = sorted(set([n for n, s in dets]))
            return names
        except Exception as e:
            print(f"Error in object detection: {e}")
            return []

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

    def _get_clip_tags(self, image: Image.Image, threshold: float = 0.18):
        # Backward-compatible wrapper: returns object detection tags
        conf = max(min(threshold, 0.9), 0.01)
        return self.detect_objects_tags(image, conf=conf)

    def rule_based_vibe(self, image: Image.Image) -> str:
        try:
            arr = np.asarray(image)
            hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)
            s_mean = float(np.mean(s))
            v_mean = float(np.mean(v))
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            if v_mean < 80:
                return "dark moody"
            if v_mean > 170 and s_mean > 100:
                return "bright vibrant"
            if s_mean < 60 and 90 <= v_mean <= 170:
                return "minimalist"
            if lap_var > 700 and s_mean > 80:
                return "energetic dynamic"
            return "calm peaceful"
        except Exception:
            return "minimalist"

    def _get_clip_vibe(self, image: Image.Image):
        return self.rule_based_vibe(image)

    def _get_quality_details(self, image: Image.Image):
        # Option A: OpenCV BRISQUE (requires opencv-contrib and model/range files)
        score: float = 0.0
        label: str = "unknown"
        used_brisque = False
        try:
            if self.brisque_model_path and self.brisque_range_path and hasattr(cv2, "quality"):
                arr = np.asarray(image)
                # BRISQUE expects BGR
                bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                try:
                    # Newer API (create + compute)
                    brq = cv2.quality.QualityBRISQUE_create(self.brisque_model_path, self.brisque_range_path)
                    score = float(brq.compute(bgr)[0])
                    used_brisque = True
                except Exception:
                    # Static API fallback
                    s = cv2.quality.QualityBRISQUE_compute(bgr, self.brisque_model_path, self.brisque_range_path)
                    score = float(s[0] if isinstance(s, (list, tuple, np.ndarray)) else s)
                    used_brisque = True
            if used_brisque:
                if score < 20:
                    label = "excellent"
                elif score < 40:
                    label = "good"
                elif score < 60:
                    label = "average"
                else:
                    label = "poor"
        except Exception as e:
            print(f"BRISQUE failed: {e}")
            used_brisque = False

        # Option B: Heuristics fallback
        try:
            gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
            brightness = float(np.mean(gray))
            contrast = float(np.std(gray))
            lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            if not used_brisque:
                # Compose a simple score (lower is better to mimic BRISQUE style)
                # Scale components to comparable ranges
                inv_sharp = max(0.0, 1000.0 - min(lap_var, 1000.0)) / 10.0
                mid_brightness_penalty = abs(brightness - 128.0) / 3.0
                low_contrast_penalty = max(0.0, 50.0 - min(contrast, 50.0))
                score = float(inv_sharp + mid_brightness_penalty + low_contrast_penalty)
                if score < 20:
                    label = "excellent"
                elif score < 40:
                    label = "good"
                elif score < 60:
                    label = "average"
                else:
                    label = "poor"

            lighting = "well-lit" if brightness > 140 else "dim" if brightness < 80 else "balanced"
            appeal = "high" if (label in ["excellent", "good"] and contrast > 45) else "medium" if contrast > 25 else "low"
            edges = cv2.Canny(gray, 100, 200)
            edge_density = float(edges.mean())
            consistency = "clean" if edge_density < 10 else "detailed" if edge_density < 25 else "busy"
        except Exception:
            lighting, appeal, consistency = "unknown", "unknown", "unknown"

        return {"score": float(score), "label": label, "lighting": lighting, "visual_appeal": appeal, "consistency": consistency}

    def _get_objects(self, image: Image.Image, conf: float = 0.5):
        try:
            dets = self._detect_with_onnx(image, conf=conf)
            if not dets:
                dets = self._detect_with_opencv_ssd(image, conf=conf)
            return sorted(set([n for n, s in dets]))
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
