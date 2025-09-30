from rest_framework import serializers


class AnalysisSummarySerializer(serializers.Serializer):
    total_posts = serializers.IntegerField()
    successful_analysis = serializers.IntegerField()
    success_rate = serializers.CharField()


class ProfileResponseSerializer(serializers.Serializer):
    profile = serializers.DictField()
    analysis_summary = AnalysisSummarySerializer()


class PostsResponseSerializer(serializers.Serializer):
    posts = serializers.ListField(child=serializers.DictField())
    count = serializers.IntegerField()
    analysis_summary = AnalysisSummarySerializer()
