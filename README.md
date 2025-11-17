# Video Content Relevance Scorer

The **Video Content Relevance Scorer** is an AI-powered application designed to analyze and evaluate how closely a video’s content matches its declared title and description. With the rapid growth of online video platforms, misleading or clickbait content has become increasingly common. This tool enables educators, marketers, and platforms to verify content accuracy and improve user experience through automated, intelligent relevance scoring.

---

## 🚧 Problem Overview

### **The Challenge**

* **Misleading Content**: Video titles and thumbnails often misrepresent content.
* **Content Verification Needs**: Educators, marketers, and recommendation engines require reliable ways to validate video accuracy.
* **Quality Assessment Gaps**: Automated systems struggle to understand deeper semantic alignment.

### **The Solution**

This application uses AI to:

* Process the video’s title, description, and transcript.
* Analyze semantic similarity between declared topics and actual spoken content.
* Detect promotional, irrelevant, or filler segments.
* Produce detailed relevance scores with explanations.

---

## ✨ Core Features

### **1. Multi-Modal Input System**

* **YouTube Integration** with transcript extraction via yt-dlp
* **Manual Input** for title, description, transcript
* **Sample Data** for quick testing and demos

### **2. AI-Powered Analysis**

* Semantic similarity using **Sentence Transformers** (`all-MiniLM-L6-v2`)
* Promotional content identification
* Segment-wise classification: *relevant, moderately relevant, irrelevant, promotional*

### **3. Comprehensive Scoring**

* **Overall relevance score (0-100%)**
* Segment-level scoring
* AI-generated explanations for each segment

### **4. Visual Analytics**

* Relevance heatmap
* Segment classification charts
* Interactive dashboard built with Streamlit & Plotly

---

## 🏗️ Technical Architecture

### **Technology Stack**

**Backend & AI**

* Python 3.8+
* Sentence Transformers
* Transformers
* PyTorch

**Frontend**

* Streamlit for UI
* Plotly for visualizations

**Other Tools**

* Pandas, NumPy
* yt-dlp for YouTube data

---

## 🔧 System Components

1. **Input Processing Module**

   * YouTube URL → Transcript extraction
   * Manual input for custom data
   * Pre-loaded sample content

2. **Content Analysis Engine**

   * Transcript segmentation (200-word blocks)
   * Semantic similarity calculation
   * Promotional content detection
   * Classification and scoring

3. **Visualization Layer**

   * Heatmaps
   * Bar charts
   * Downloadable reports

---

## 🧠 Algorithm Details

### **Semantic Similarity**

* Embeddings generated for title, description, and transcript segments
* Cosine similarity used for comparison
* Normalized scores converted to 0-100 range

### **Promotional Content Detection**

* Keyword-based marketing vocabulary
* Density-based scoring

### **Segment Classification Logic**

```
IF promotional_score > 0.3 → promotional
ELSE IF similarity_score > 0.6 → relevant
ELSE IF similarity_score > 0.3 → moderately relevant
ELSE → irrelevant
```

---

## 📦 Installation & Setup

### **Prerequisites**

* Python 3.8+
* pip
* 4GB+ RAM recommended

### **1. Create Virtual Environment**

```
python -m venv video_relevance_env
source video_relevance_env/bin/activate   # Linux/Mac
video_relevance_env\\Scripts\\activate     # Windows
```

### **2. Install Dependencies**

```
pip install streamlit pandas numpy plotly scikit-learn
pip install sentence-transformers transformers torch
pip install yt-dlp
```

### **3. Run Application**

```
streamlit run app.py
```

---

## 🔍 Usage Guide

### **Input Methods**

#### 1. YouTube URL

* Paste video URL
* Auto transcript extraction

#### 2. Manual Input

* Enter title, description, and paste transcript

#### 3. Sample Data

* Built-in demo datasets

### **Analysis Workflow**

1. Input validation
2. Transcript segmentation
3. Semantic and promotional analysis
4. Classification
5. Scoring and visualization
6. Optional report export

### **Score Interpretation**

| Score   | Meaning                   |
| ------- | ------------------------- |
| 90-100% | Excellent alignment       |
| 70-89%  | Strong relevance          |
| 50-69%  | Moderate alignment        |
| 30-49%  | Weak relevance            |
| 0-29%   | Poor / misleading content |

---

## 🎯 Application Scenarios

### **Education**

* Verify learning content accuracy
* Maintain curriculum consistency

### **Marketing**

* Validate ad content delivery
* Ensure brand safety

### **Platform Use Cases**

* Improve recommendation systems
* Automate content moderation

---

## ⚠️ Limitations & Future Enhancements

### **Current Limitations**

* Whisper-based audio processing disabled
* Primarily supports English
* Limited deep contextual understanding

### **Planned Upgrades**

* Multi-language support
* Advanced models (GPT-4+, Claude, Gemini)
* Real-time processing
* REST API integration
* Batch video processing

---

## 🖥️ Performance Considerations

* Model load time: ~10-30 seconds
* YouTube extraction: 5-15 seconds
* Analysis: 2-5 seconds per 1000 words
* RAM: 2GB minimum, 4GB+ recommended

---

## 🔐 Ethical Considerations

* All processing happens **locally**
* No data uploaded to external servers
* Only public content is analyzed
* Transparent scoring methodology

---

## 💼 Business Value

### For Educators

* Ensures content quality
* Saves verification time

### For Marketers

* Protects brand reputation
* Measures content authenticity

### For Platforms

* Enhances content ranking systems
* Improves user trust

---

## ✅ Conclusion

The **Video Content Relevance Scorer** brings robust, AI-driven content analysis to the rapidly growing video ecosystem. By combining semantic understanding, segment-level insights, and intuitive visualization, it supports educators, marketers, and platforms in ensuring content accuracy, reducing misinformation, and improving overall user experience.

As online video content continues to scale, tools like this play a critical role in maintaining quality and trust across digital platforms.
