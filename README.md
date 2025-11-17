Video Content Relevance Scorer - Project Documentation
Project Overview
The Video Content Relevance Scorer is an AI-powered application designed to analyze and evaluate how well video content aligns with its declared title and description. In today's digital landscape where thousands of videos are uploaded hourly across platforms like YouTube, social media, and e-learning portals, this tool addresses the critical challenge of assessing content relevance and detecting misleading or clickbait content.

Problem Statement
The Challenge
Misleading Content: Many videos use clickbait titles and thumbnails that don't match actual content

Content Verification: Educators, marketers, and recommendation systems need reliable ways to verify content relevance

Quality Assessment: Automated systems struggle to understand semantic relationships between titles and content

Solution
An AI model that:

Takes video URL, title, description, and transcript as input

Evaluates semantic relevance between declared topics and actual content

Detects off-topic, promotional, or filler segments

Generates comprehensive relevance scores with detailed justifications

Core Features
1. Multi-Modal Input System
YouTube Integration: Direct URL processing with automatic transcript extraction

Manual Input: Support for pasting titles, descriptions, and transcripts

Sample Data: Built-in demonstration data for testing

2. AI-Powered Analysis
Semantic Similarity: Uses sentence transformers to compare title/description with content

Promotional Detection: Identifies marketing and sales-oriented content

Segment Classification: Categorizes content as relevant, moderately relevant, irrelevant, or promotional

3. Comprehensive Scoring
Overall Relevance Score: 0-100% based on content alignment

Segment-level Analysis: Individual scoring for different parts of the video

Justification Generation: AI-generated explanations for scores

4. Visual Analytics
Relevance Heatmap: Color-coded timeline showing relevance across segments

Segment Analysis Charts: Bar charts showing distribution of content types

Interactive Dashboard: Real-time visualization of analysis results

Technical Architecture
Technology Stack
Backend & AI
Python 3.8+: Primary programming language

Sentence Transformers: For semantic similarity analysis using 'all-MiniLM-L6-v2'

Transformers: Optional promotional content classification

PyTorch: Deep learning framework for AI models

Web Framework
Streamlit: Interactive web application framework

Plotly: Interactive visualizations and charts

Pandas & NumPy: Data manipulation and analysis

External Services
yt-dlp: YouTube transcript extraction

Optional: OpenAI Whisper (currently disabled due to compatibility issues)

System Components
1. Input Processing Module
text
YouTube URL → yt-dlp → Transcript Extraction
Manual Input → Direct Processing
Sample Data → Pre-loaded Examples
2. Content Analysis Engine
text
Transcript → Segmentation → Semantic Analysis → Classification
              ↓
        Relevance Scoring → Justification Generation
3. Visualization Layer
text
Analysis Results → Heatmaps → Charts → Export Reports
Algorithm Details
Semantic Similarity Calculation
Text Embedding: Convert title, description, and transcript segments to vector representations

Cosine Similarity: Measure angular distance between vectors

Score Normalization: Convert similarity scores to 0-100 scale

Promotional Content Detection
Keyword Matching: Extensive list of marketing and sales terms

Frequency Analysis: Count promotional keyword occurrences

Score Calculation: Normalize based on keyword density

Segment Classification Logic
text
IF promotional_score > 0.3 → "promotional"
ELSE IF similarity_score > 0.6 → "relevant"
ELSE IF similarity_score > 0.3 → "moderately_relevant"
ELSE → "irrelevant"
Installation & Setup
Prerequisites
Python 3.8 or higher

pip package manager

4GB+ RAM recommended for AI models

Step-by-Step Installation
Create Virtual Environment

bash
python -m venv video_relevance_env
source video_relevance_env/bin/activate  # Linux/Mac
video_relevance_env\Scripts\activate    # Windows
Install Dependencies

bash
pip install streamlit pandas numpy plotly scikit-learn
pip install sentence-transformers transformers torch
pip install yt-dlp
Run Application

bash
streamlit run app.py
Dependency Details
Core Dependencies
streamlit==1.28.0: Web application framework

sentence-transformers==2.2.2: Semantic similarity models

transformers==4.33.0: NLP model framework

torch==2.0.1: Machine learning library

Data Processing
pandas==2.0.3: Data manipulation

numpy==1.24.3: Numerical computing

scikit-learn==1.3.0: Machine learning utilities

Visualization
plotly==5.15.0: Interactive charts and heatmaps

External Services
yt-dlp==2023.10.13: YouTube content extraction

Usage Guide
Input Methods
1. YouTube URL Analysis
Paste any YouTube video URL

Automatic transcript extraction

Real-time processing status

2. Manual Input
Enter video title and description

Paste full transcript text

Immediate analysis

3. Sample Data
Pre-loaded demonstration content

Instant results for testing

Educational purposes

Analysis Process
Input Validation: Verify required fields (title and transcript)

Text Segmentation: Split transcript into 200-word segments

Semantic Analysis: Compare each segment with reference text

Classification: Categorize segments based on relevance

Scoring: Calculate overall and segment-level scores

Visualization: Generate charts and heatmaps

Reporting: Create downloadable analysis report

Output Interpretation
Relevance Scores
90-100%: Excellent content alignment

70-89%: Strong relevance with minor deviations

50-69%: Moderate alignment with some off-topic content

30-49%: Weak relevance, significant mismatches

0-29%: Poor alignment, potentially misleading

Segment Types
Relevant: Directly related to declared topic

Moderately Relevant: Some connection to topic

Irrelevant: No meaningful connection

Promotional: Marketing or sales content

Application Scenarios
Educational Use Cases
Course Content Verification: Ensure educational videos match learning objectives

E-learning Quality Control: Maintain content standards in online courses

Curriculum Alignment: Verify video relevance to specific topics

Marketing Applications
Ad Content Validation: Ensure promotional videos deliver promised content

Brand Safety: Detect misleading or off-brand content

Competitor Analysis: Evaluate competitor video content strategies

Platform Integration
Content Recommendation: Improve video recommendation algorithms

Content Moderation: Identify misleading or spam content

Quality Scoring: Automated content quality assessment

Technical Limitations & Future Enhancements
Current Limitations
Audio Transcription: Whisper integration disabled due to compatibility issues

Language Support: Primarily English language focused

Real-time Processing: Limited by model loading times

Context Understanding: Basic semantic analysis without deep context comprehension

Planned Enhancements
Multi-language Support: Expand to other languages

Advanced AI Models: Integration with GPT-4, Claude, or Gemini

Real-time Processing: Optimized for faster analysis

API Integration: RESTful API for platform integration

Batch Processing: Analyze multiple videos simultaneously

Custom Models: Train domain-specific relevance models

Performance Considerations
Processing Times
Model Loading: 10-30 seconds initial load

YouTube Extraction: 5-15 seconds depending on video length

Analysis Time: 2-5 seconds per 1000 words of transcript

Visualization: Near-instant after analysis completion

Resource Requirements
RAM: Minimum 2GB, Recommended 4GB+

Storage: 1-2GB for models and dependencies

CPU: Multi-core processor recommended

Network: Internet required for YouTube processing

Ethical Considerations
Content Privacy
All processing happens locally (no data sent to external servers)

User data is not stored permanently

Transcripts are processed in memory only

Fair Use Compliance
Respects copyright and fair use guidelines

Only processes publicly available content

Provides educational and analytical value

Bias Mitigation
Uses general-purpose language models

Regular updates to address potential biases

Transparent scoring methodology

Business Value Proposition
For Educators
Quality Assurance: Ensure educational content matches objectives

Time Savings: Automated content verification

Improved Learning: Better aligned educational materials

For Marketers
Campaign Effectiveness: Verify ad content delivery

Brand Protection: Detect misleading content early

Competitive Insights: Analyze industry content trends

For Platforms
Content Quality: Improved user experience

Recommendation Accuracy: Better content matching

Moderation Efficiency: Automated quality checks

Conclusion
The Video Content Relevance Scorer represents a significant advancement in automated content analysis technology. By leveraging state-of-the-art AI models and providing comprehensive analytical capabilities, it addresses a critical need in the digital content ecosystem. The application's modular architecture, user-friendly interface, and robust analysis capabilities make it suitable for various professional and educational applications.

As digital content continues to grow exponentially, tools like this become increasingly essential for maintaining content quality, ensuring truth in advertising, and improving user experiences across platforms.
