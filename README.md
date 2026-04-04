# Wound Intelligence Dashboard

An AI-based Wound Analysis System using Deep Learning. This application provides automated wound segmentation, technical metrics extraction, and clinical risk assessment through a modern web-based dashboard.

## Features

- **Automated Segmentation**: High-precision wound boundary detection using a PyTorch UNet model.
- **Clinical Metrics**: Automated calculation of Wound Area, Redness Index, and Tissue Compactness.
- **Risk Assessment**: Multi-factor risk scoring (Severity, Infection Risk, Danger Level).
- **AI Explanation**: Local LLM integration (Ollama) providing human-friendly summaries and actionable recommendations.
- **Interactive Dashboard**: Responsive web UI with real-time segmentation overlays and dynamic analytics charts.

## Tech Stack

- **Backend**: Python, FastAPI, PyTorch, OpenCV
- **Frontend**: HTML5, CSS3 (Modern Dark Theme), JavaScript (Vanilla)
- **Charts**: Chart.js
- **LLM**: Ollama (llama3)

## Project Structure

```text
project_root/
├── app.py              # FastAPI Backend Entry
├── core/
│   ├── inference.py    # AI Model Inference Pipeline
│   └── analysis.py     # Metrics & AI Report Logic
├── models/
│   └── best_model.pth  # Model Weights
├── static/             # CSS, JS, Assets
├── templates/          # HTML Templates
├── requirements.txt    # Dependencies
└── README.md           # Documentation
```

## Setup & Installation

1. **Clone the Repository**:
   ```bash
   git clone <your-repo-url>
   cd wound-intelligence
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Model Weights**:
   Ensure the `best_model.pth` file is placed in the `models/` directory. 
   *(Note: If the file is large, download it from [your-source-link] and place it manually).*

4. **Start the AI Server (Optional for Explanations)**:
   Ensure [Ollama](https://ollama.com/) is installed and running with `llama3`.

5. **Run the Application**:
   ```bash
   python app.py
   ```
   Open your browser at **http://127.0.0.1:8001**.

---
*Disclaimer: This tool is for research and demonstration. Always consult a healthcare professional for actual clinical diagnosis.*
