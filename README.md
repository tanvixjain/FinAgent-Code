# 🤖 FinAIgent: Multilingual AI Financial Assistant

FinAIgent is a state-of-the-art, RAG-powered financial assistant designed to provide accurate, context-aware, and multilingual financial advice. Specifically tailored for the Indian context, it supports 10+ regional languages and leverages specialized datasets for scam prevention, government schemes, and personal budgeting.

---

## 🚀 Key Features

- **🌐 Multilingual Support**: Seamlessly interact in English, Hindi, Marathi, Tamil, Telugu, and more. 
- **🔎 Hybrid RAG Architecture**: Combines local TF-IDF & KNN-based retrieval with the generative power of **Google Gemini (1.5/2.0 Flash)**.
- **🛡️ Scam & Fraud Protection**: Dedicated modules for identifying common financial scams and providing preventative actions.
- **🏛️ Indian Government Schemes**: Comprehensive database of welfare schemes, eligibility, and application processes.
- **🎨 Premium Glassmorphism UI**: High-end user experience with dark/light modes, animated backgrounds, and responsive design.
- **🚨 Emergency Integration**: Quick access to national cybercrime reporting (Helpline 1930).
- **📊 Research-Grade Visualizations**: Built-in suite to generate PCA/t-SNE vector space plots, accuracy graphs, and system architecture diagrams.
- **⚡ Adaptive Response Levels**: Dynamically adjusts response depth (Low, Medium, High) based on query complexity.

---

## 🏗️ System Architecture

FinAIgent operates on a multi-layered architecture:
1. **Frontend**: Interactive web interface for user queries.
2. **NLU Layer**: Language detection and translation (English processing core).
3. **Retrieval Engine**: TF-IDF Vectorization and KNN (K-Nearest Neighbors) search across curated CSV datasets.
4. **LLM Integration**: Grounded generation via Google Gemini, ensuring responses are both creative and factual.
5. **Post-Processing**: Final translation and formatting to ensure user satisfaction in their native language.

---

## 📂 Project Structure

```text
├── app.py                      # Main Flask API and Core Logic
├── finaigent_visualizations.py  # Scientific plotting and analysis suite
├── evaluate_model_graphs.py    # Performance evaluation scripts
├── templates/                  # Web UI components
├── static/                     # Assets and Styles
├── config.py/config_example.py # Configuration management
├── datasets/                   # Curated CSV files (Scams, Schemes, Budgeting)
└── visualizations_output/      # Generated analysis reports
```

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- Python 3.9+
- Google Gemini API Key

### 2. Install Dependencies
```bash
pip install flask flask-cors pandas scikit-learn google-generativeai googletrans==4.0.0-rc1 langdetect matplotlib seaborn
```

### 3. Configure API Key
Create a `config.py` file in the root directory:
```python
GEMINI_API_KEY = "YOUR_API_KEY_HERE"
```

### 4. Run the Application
```bash
python app.py
```
The server will start on `http://localhost:5000`.

---

## 📈 Research & Performance

FinAIgent is designed with research in mind. You can generate performance metrics and visualizations using:
```bash
python finaigent_visualizations.py
```
This produces:
- **Vector Space Plots**: PCA and t-SNE visualizations of the dataset embeddings.
- **Precision-Recall Curves**: Retrieval performance analysis.
- **Processing Time Breakdown**: Efficiency analysis of each stage (Detection, Retrieval, Generation).

---



