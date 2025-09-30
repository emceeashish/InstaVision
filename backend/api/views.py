from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view
from django.http import HttpResponse, FileResponse

import requests
from pathlib import Path

from .utils import ensure_analyzed, ensure_scraped, analysis_exists, load_analyzed, run_analysis_async
from .serializers import ProfileResponseSerializer, PostsResponseSerializer


class ProfileView(APIView):
    def get(self, request, username: str):
        try:
            profile, posts = ensure_scraped(username)
            if not analysis_exists(username):
                run_analysis_async(username)
            # Removed video prefetching; thumbnails-only approach
        except RuntimeError as e:
            return Response({"detail": str(e)}, status=status.HTTP_502_BAD_GATEWAY)
        except Exception as e:
            return Response({"detail": f"Unexpected error: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        summary = {"total_posts": 0, "successful_analysis": 0, "success_rate": "0.0%"}
        try:
            if analysis_exists(username):
                summary = load_analyzed(username).get("analysis_summary", summary)
        except Exception:
            pass

        resp = {
            "profile": profile,
            "analysis_summary": summary,
        }
        serializer = ProfileResponseSerializer(data=resp)
        serializer.is_valid(raise_exception=True)
        return Response(serializer.data)


class PostsView(APIView):
    def get(self, request, username: str):
        try:
            data = ensure_analyzed(username)
        except RuntimeError as e:
            return Response({"detail": str(e)}, status=status.HTTP_502_BAD_GATEWAY)
        except Exception as e:
            return Response({"detail": f"Unexpected error: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        posts = data.get("posts", [])

        limit = request.query_params.get('limit')
        media_type = request.query_params.get('media_type')

        filtered = posts
        if media_type in ("image", "video"):
            filtered = [p for p in posts if (p.get("is_video") and media_type == "video") or (not p.get("is_video") and media_type == "image")]
        if limit:
            try:
                n = int(limit)
                filtered = filtered[:n]
            except ValueError:
                pass

        # Removed background video prefetching; thumbnails-only approach
        resp = {
            "posts": filtered,
            "count": len(filtered),
            "analysis_summary": data.get("analysis_summary", {}),
        }
        serializer = PostsResponseSerializer(data=resp)
        serializer.is_valid(raise_exception=True)
        return Response(serializer.data)


@api_view(["GET"])
def image_proxy(request):
    url = request.query_params.get("url")
    if not url:
        return Response({"detail": "Missing url param"}, status=status.HTTP_400_BAD_REQUEST)
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            "Referer": "https://www.instagram.com/",
        }
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        resp = HttpResponse(r.content, content_type=r.headers.get("Content-Type", "image/jpeg"))
        resp["Cache-Control"] = "public, max-age=86400"
        return resp
    except Exception as e:
        return Response({"detail": f"Failed to fetch image: {e}"}, status=status.HTTP_502_BAD_GATEWAY)
