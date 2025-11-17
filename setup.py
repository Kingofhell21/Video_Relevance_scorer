from setuptools import setup, find_packages

setup(
    name="video_relevance_scorer",
    version="1.0.0",
    description="AI-powered Video Content Relevance Evaluator",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.28.0",
        "pandas>=2.0.3",
        "numpy>=1.24.3",
        "plotly>=5.15.0",
        "sentence-transformers>=2.2.2",
        "transformers>=4.33.0",
        "torch>=2.0.1",
        "openai-whisper>=20231117",
        "yt-dlp>=2023.10.13",
        "scikit-learn>=1.3.0",
    ],
    python_requires=">=3.8",
)