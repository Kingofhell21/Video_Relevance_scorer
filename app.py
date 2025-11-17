import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import tempfile
import os
import re
import warnings

warnings.filterwarnings('ignore')

# AI/ML Imports - with better error handling
try:
    from sentence_transformers import SentenceTransformer, util

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    st.error("Sentence transformers not available. Please install: pip install sentence-transformers")

try:
    from transformers import pipeline

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    st.warning("Transformers not available. Promotional detection will use basic keyword matching.")

try:
    import yt_dlp

    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False
    st.warning("yt-dlp not available. YouTube transcript extraction disabled.")


# Remove problematic whisper import and provide alternative
class VideoRelevanceScorer:
    def __init__(self):
        self.sentence_model = None
        self.promotion_classifier = None

        # Initialize models with error handling
        try:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                st.success("✅ Semantic analysis model loaded")
            else:
                st.error("❌ Sentence transformers not available - using basic text analysis")
        except Exception as e:
            st.error(f"❌ Error loading sentence model: {e}")

        try:
            if TRANSFORMERS_AVAILABLE:
                # Use a simpler model for promotional detection
                self.promotion_classifier = pipeline(
                    "text-classification",
                    model="distilbert-base-uncased"
                )
                st.success("✅ Promotional content detector loaded")
            else:
                st.warning("⚠️ Using basic keyword-based promotional detection")
        except Exception as e:
            st.warning(f"⚠️ Could not load promotional classifier: {e}. Using keyword-based approach.")

    def extract_youtube_transcript(self, video_url):
        """Extract transcript from YouTube video"""
        if not YT_DLP_AVAILABLE:
            st.error("YouTube transcript extraction requires yt-dlp. Install with: pip install yt-dlp")
            return None

        try:
            # Configure yt-dlp for transcript extraction
            ydl_opts = {
                'writesubtitles': True,
                'writeautomaticsub': True,
                'subtitleslangs': ['en'],
                'skip_download': True,
                'quiet': True
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)

                # Extract transcript from available subtitles
                transcript_text = ""

                # Try manual subtitles first
                if 'subtitles' in info and 'en' in info['subtitles']:
                    for sub in info['subtitles']['en']:
                        if 'data' in sub:
                            transcript_text += sub['data'] + " "

                # Try automatic captions
                elif 'automatic_captions' in info and 'en' in info['automatic_captions']:
                    for sub in info['automatic_captions']['en']:
                        if 'data' in sub:
                            transcript_text += sub['data'] + " "

                if transcript_text.strip():
                    return transcript_text.strip()
                else:
                    st.warning("No English subtitles found for this video")
                    return None

        except Exception as e:
            st.error(f"Error extracting YouTube transcript: {str(e)}")
            return None

    def segment_transcript(self, transcript, segment_length=200):
        """Split transcript into segments for analysis"""
        if not transcript:
            return []

        words = transcript.split()
        segments = []

        for i in range(0, len(words), segment_length):
            segment = ' '.join(words[i:i + segment_length])
            segments.append({
                'text': segment,
                'start_word': i,
                'end_word': min(i + segment_length, len(words))
            })

        return segments

    def calculate_semantic_similarity(self, reference_text, content_segments):
        """Calculate semantic similarity between reference and content segments"""
        if not self.sentence_model:
            # Fallback: basic text similarity using word overlap
            return self.calculate_basic_similarity(reference_text, content_segments)

        try:
            # Encode reference text
            reference_embedding = self.sentence_model.encode([reference_text])

            # Encode content segments
            segment_texts = [segment['text'] for segment in content_segments]
            segment_embeddings = self.sentence_model.encode(segment_texts)

            # Calculate cosine similarities
            similarities = util.pytorch_cos_sim(reference_embedding, segment_embeddings)[0]

            return similarities.tolist()
        except Exception as e:
            st.warning(f"Semantic similarity failed, using basic method: {e}")
            return self.calculate_basic_similarity(reference_text, content_segments)

    def calculate_basic_similarity(self, reference_text, content_segments):
        """Fallback basic similarity using word overlap"""
        reference_words = set(reference_text.lower().split())
        similarities = []

        for segment in content_segments:
            segment_words = set(segment['text'].lower().split())
            if len(reference_words) == 0 or len(segment_words) == 0:
                similarities.append(0.0)
                continue

            overlap = len(reference_words.intersection(segment_words))
            similarity = overlap / len(reference_words.union(segment_words))
            # Scale similarity to make it more meaningful
            scaled_similarity = min(similarity * 3, 1.0)
            similarities.append(scaled_similarity)

        return similarities

    def detect_promotional_content(self, text):
        """Detect promotional or marketing content"""
        try:
            # Enhanced keyword-based approach
            promotional_keywords = [
                'buy now', 'discount', 'offer', 'limited time', 'subscribe',
                'click here', 'link in description', 'sponsor', 'promotion',
                'affiliate', 'coupon code', 'special offer', 'sale', 'purchase',
                'check out', 'visit our website', 'use code', 'get it now',
                'limited offer', 'exclusive deal', 'shop now', 'buy today',
                'discount code', 'promo code', 'limited supply', 'act now',
                'sign up', 'join now', 'free trial', 'money back', 'best price',
                'lowest price', 'hot deal', 'flash sale', 'big sale', 'clearance'
            ]

            text_lower = text.lower()
            promotional_score = sum(1 for keyword in promotional_keywords
                                    if keyword in text_lower)

            # Normalize score (0-1)
            normalized_score = min(promotional_score / 8, 1.0)  # Cap at 1.0

            return normalized_score
        except:
            return 0.0

    def analyze_relevance(self, title, description, transcript):
        """Main analysis function"""
        if not transcript:
            st.error("No transcript provided for analysis")
            return None

        if not title and not description:
            st.error("Please provide at least a title or description")
            return None

        # Segment transcript
        segments = self.segment_transcript(transcript)

        if not segments:
            st.error("Could not segment transcript for analysis")
            return None

        # Prepare reference texts
        reference_texts = []
        if title:
            reference_texts.append(title)
        if description:
            reference_texts.append(description)

        combined_reference = " ".join(reference_texts)

        # Calculate similarities
        similarities = self.calculate_semantic_similarity(combined_reference, segments)

        # Analyze each segment
        analysis_results = []
        total_relevance_score = 0
        total_promotional_score = 0

        for i, segment in enumerate(segments):
            similarity_score = similarities[i] if i < len(similarities) else 0.0
            promotional_score = self.detect_promotional_content(segment['text'])

            # Classify segment
            if promotional_score > 0.3:
                segment_type = "promotional"
            elif similarity_score > 0.6:
                segment_type = "relevant"
            elif similarity_score > 0.3:
                segment_type = "moderately_relevant"
            else:
                segment_type = "irrelevant"

            segment_result = {
                'segment_id': i,
                'text': segment['text'],
                'similarity_score': similarity_score,
                'promotional_score': promotional_score,
                'segment_type': segment_type,
                'start_word': segment['start_word'],
                'end_word': segment['end_word']
            }

            analysis_results.append(segment_result)
            total_relevance_score += similarity_score
            total_promotional_score += promotional_score

        # Calculate overall scores
        avg_relevance = total_relevance_score / len(segments) if segments else 0
        avg_promotional = total_promotional_score / len(segments) if segments else 0

        # Convert to percentage (0-100) with promotional penalty
        base_score = avg_relevance * 100
        promotional_penalty = avg_promotional * 30
        overall_score = max(0, min(100, base_score - promotional_penalty))

        return {
            'overall_score': overall_score,
            'segments': analysis_results,
            'avg_relevance': avg_relevance,
            'avg_promotional': avg_promotional,
            'total_segments': len(segments)
        }

    def generate_justification(self, analysis_result, title):
        """Generate justification for the relevance score"""
        if not analysis_result:
            return "Analysis failed - no results available"

        relevant_segments = len([s for s in analysis_result['segments']
                                 if s['segment_type'] in ['relevant', 'moderately_relevant']])
        promotional_segments = len([s for s in analysis_result['segments']
                                    if s['segment_type'] == 'promotional'])

        total_segments = analysis_result['total_segments']
        relevance_percentage = (relevant_segments / total_segments) * 100 if total_segments > 0 else 0

        justification_parts = []

        score = analysis_result['overall_score']
        if score >= 80:
            justification_parts.append(f"Content strongly matches the title '{title}'")
        elif score >= 60:
            justification_parts.append(f"Content generally aligns with the title '{title}'")
        elif score >= 40:
            justification_parts.append(f"Content partially matches the title '{title}'")
        else:
            justification_parts.append(f"Content shows weak alignment with the title '{title}'")

        if promotional_segments > 0:
            justification_parts.append(f"Contains {promotional_segments} promotional segments")

        justification_parts.append(f"{relevance_percentage:.1f}% of content is relevant")

        return ". ".join(justification_parts) + f". Final score: {analysis_result['overall_score']:.1f}%"


def create_relevance_heatmap(analysis_results):
    """Create a heatmap visualization of relevance scores"""
    if not analysis_results or not analysis_results['segments']:
        st.warning("No data available for heatmap")
        return go.Figure()

    segments_df = pd.DataFrame(analysis_results['segments'])

    fig = go.Figure(data=go.Heatmap(
        z=segments_df['similarity_score'],
        x=segments_df['segment_id'],
        colorscale='RdYlGn',
        hovertext=segments_df['segment_type'],
        hovertemplate='Segment: %{x}<br>Score: %{z:.2f}<br>Type: %{hovertext}<extra></extra>'
    ))

    fig.update_layout(
        title='Relevance Heatmap by Segment',
        xaxis_title='Segment Number',
        yaxis_title='',
        height=300
    )

    return fig


def create_segment_analysis_chart(analysis_results):
    """Create a bar chart showing segment analysis"""
    if not analysis_results or not analysis_results['segments']:
        st.warning("No data available for segment analysis")
        return px.bar(title="No data available")

    segments_df = pd.DataFrame(analysis_results['segments'])

    # Count segment types
    type_counts = segments_df['segment_type'].value_counts()

    fig = px.bar(
        x=type_counts.index,
        y=type_counts.values,
        title='Content Segment Analysis',
        labels={'x': 'Segment Type', 'y': 'Count'},
        color=type_counts.index,
        color_discrete_map={
            'relevant': 'green',
            'moderately_relevant': 'yellow',
            'irrelevant': 'red',
            'promotional': 'orange'
        }
    )

    return fig


def main():
    st.set_page_config(
        page_title="Video Relevance Scorer",
        page_icon="🎬",
        layout="wide"
    )

    st.title("🎬 AI-Powered Video Content Relevance Evaluator")
    st.markdown("""
    Analyze how well your video content matches its title and description.
    Provide a YouTube URL or paste a transcript to get started!
    """)

    # Display dependency status
    with st.expander("🔧 Dependency Status", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("🤖 Semantic Analysis:", "✅" if SENTENCE_TRANSFORMERS_AVAILABLE else "❌")
        with col2:
            st.write("📺 YouTube Extraction:", "✅" if YT_DLP_AVAILABLE else "❌")
        with col3:
            st.write("📊 Advanced Detection:", "✅" if TRANSFORMERS_AVAILABLE else "⚠️")

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            st.error("""
            **Required installation:** 
            ```bash
            pip install sentence-transformers
            ```
            Without this, basic text analysis will be used instead of AI-powered semantic analysis.
            """)

    # Initialize scorer
    if 'scorer' not in st.session_state:
        with st.spinner("Loading AI models..."):
            st.session_state.scorer = VideoRelevanceScorer()

    # Input methods
    input_method = st.radio(
        "Choose input method:",
        ["Manual Input", "YouTube URL", "Sample Data"],
        horizontal=True
    )

    title = ""
    description = ""
    transcript = ""

    if input_method == "YouTube URL":
        youtube_url = st.text_input("Enter YouTube URL:", placeholder="https://www.youtube.com/watch?v=...")
        if youtube_url:
            with st.spinner("Extracting transcript from YouTube..."):
                transcript = st.session_state.scorer.extract_youtube_transcript(youtube_url)
                if transcript:
                    st.success("✅ Transcript extracted successfully!")
                    with st.expander("View Extracted Transcript"):
                        st.text_area("Transcript", transcript, height=150, key="youtube_transcript")
                else:
                    st.error("❌ Could not extract transcript. Please try another video or use manual input.")

    elif input_method == "Sample Data":
        st.info("Using sample data for demonstration")
        title = "Introduction to Machine Learning"
        description = "A comprehensive introduction to machine learning concepts and algorithms"
        transcript = """
        Welcome to this introduction to machine learning. Machine learning is a subset of artificial intelligence 
        that focuses on algorithms that can learn from data. Today we'll cover supervised learning, unsupervised learning, 
        and reinforcement learning. But first, I want to tell you about this amazing new product that can help you learn 
        faster. It's called SuperLearn Pro and for a limited time, you can get 50% off if you use the code LEARN50. 
        Now back to machine learning. In supervised learning, we have labeled data and the algorithm learns to map 
        inputs to outputs. This is different from unsupervised learning where we don't have labels. 
        Click the link below to subscribe to our channel for more videos. Reinforcement learning is another type where 
        an agent learns through trial and error. That's all for today, don't forget to like and subscribe!
        """

    # Input forms
    with st.form("video_input_form"):
        st.subheader("📝 Video Information")

        col1, col2 = st.columns(2)

        with col1:
            title = st.text_input("Video Title*", value=title,
                                  placeholder="Enter video title",
                                  help="The title of the video to analyze")
            description = st.text_area("Video Description", value=description,
                                       placeholder="Enter video description (optional)",
                                       height=100,
                                       help="Description of the video content")

        with col2:
            transcript = st.text_area("Video Transcript*", value=transcript,
                                      placeholder="Paste transcript here or use YouTube extraction above",
                                      height=200,
                                      help="The full transcript of the video content")

        analyze_btn = st.form_submit_button("🎯 Analyze Relevance", use_container_width=True)

    # Analysis
    if analyze_btn:
        if not title:
            st.error("❌ Please provide a video title")
        elif not transcript:
            st.error("❌ Please provide a video transcript")
        else:
            with st.spinner("Analyzing content relevance..."):
                analysis_results = st.session_state.scorer.analyze_relevance(
                    title, description, transcript
                )

            if analysis_results:
                # Display results
                st.markdown("---")
                st.header("📊 Analysis Results")

                # Score and justification
                col1, col2, col3 = st.columns(3)

                with col1:
                    score = analysis_results['overall_score']
                    score_color = "green" if score > 70 else "orange" if score > 50 else "red"
                    st.markdown(f"""
                    <div style='text-align: center; padding: 20px; border: 2px solid {score_color}; border-radius: 10px;'>
                        <h3 style='color: {score_color}; margin: 0;'>Overall Relevance Score</h3>
                        <h1 style='color: {score_color}; font-size: 48px; margin: 10px 0;'>{score:.1f}%</h1>
                        <p style='margin: 0;'>{'High Relevance' if score > 70 else 'Moderate Relevance' if score > 50 else 'Low Relevance'}</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    avg_rel = analysis_results['avg_relevance']
                    st.metric(
                        "Average Similarity",
                        f"{avg_rel:.3f}",
                        help="Average semantic similarity between content and title/description"
                    )

                with col3:
                    avg_promo = analysis_results['avg_promotional']
                    st.metric(
                        "Promotional Content",
                        f"{avg_promo:.3f}",
                        delta="High Promotion" if avg_promo > 0.3 else "Low Promotion",
                        delta_color="inverse",
                        help="Amount of promotional content detected"
                    )

                # Justification
                justification = st.session_state.scorer.generate_justification(analysis_results, title)
                st.info(f"**📝 Justification:** {justification}")

                # Visualizations
                st.subheader("📈 Visual Analysis")

                tab1, tab2, tab3 = st.tabs(["Relevance Heatmap", "Segment Analysis", "Detailed Breakdown"])

                with tab1:
                    heatmap_fig = create_relevance_heatmap(analysis_results)
                    st.plotly_chart(heatmap_fig, use_container_width=True)
                    st.caption(
                        "Heatmap shows relevance scores across video segments. Green = highly relevant, Red = irrelevant")

                with tab2:
                    chart_fig = create_segment_analysis_chart(analysis_results)
                    st.plotly_chart(chart_fig, use_container_width=True)
                    st.caption("Breakdown of content segments by relevance category")

                with tab3:
                    segments_df = pd.DataFrame(analysis_results['segments'])

                    # Color coding for display
                    def style_segment_type(val):
                        color_map = {
                            'relevant': 'background-color: #d4edda; color: #155724;',
                            'moderately_relevant': 'background-color: #fff3cd; color: #856404;',
                            'promotional': 'background-color: #ffeaa7; color: #856404;',
                            'irrelevant': 'background-color: #f8d7da; color: #721c24;'
                        }
                        return color_map.get(val, '')

                    # Display first few segments for readability
                    display_df = segments_df[
                        ['segment_id', 'segment_type', 'similarity_score', 'promotional_score']].head(15)
                    styled_df = display_df.style.applymap(style_segment_type, subset=['segment_type'])

                    st.dataframe(styled_df, use_container_width=True)

                    if len(segments_df) > 15:
                        st.info(
                            f"Showing first 15 of {len(segments_df)} segments. Download full report for complete analysis.")

                # Export results
                st.subheader("💾 Export Results")

                # Create downloadable report
                report = f"""
VIDEO RELEVANCE ANALYSIS REPORT
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Title: {title}
Description: {description}

OVERALL SCORE: {analysis_results['overall_score']:.1f}%
Justification: {justification}

DETAILED ANALYSIS:
- Average Semantic Similarity: {analysis_results['avg_relevance']:.3f}
- Promotional Content Score: {analysis_results['avg_promotional']:.3f}
- Total Segments Analyzed: {analysis_results['total_segments']}

SEGMENT BREAKDOWN:
"""

                for segment in analysis_results['segments']:
                    report += f"\n\nSegment {segment['segment_id']}:\n"
                    report += f"Type: {segment['segment_type']}\n"
                    report += f"Similarity: {segment['similarity_score']:.3f}\n"
                    report += f"Promotional: {segment['promotional_score']:.3f}\n"
                    report += f"Text: {segment['text'][:100]}..."

                st.download_button(
                    label="📥 Download Full Analysis Report",
                    data=report,
                    file_name=f"relevance_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    type="primary",
                    use_container_width=True
                )


if __name__ == "__main__":
    main()