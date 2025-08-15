# Resume Screening & Rewording App

This project is a Streamlit-based web app for screening resumes against job descriptions and automatically rewriting resumes to better match job requirements using generative AI.

## Features

- Upload your resume (TXT or PDF)
- Paste a job description
- Get a match score using semantic similarity
- Automatically rewrite resumes with AI to improve matching
- Uses TinyLlama GGUF model for rewriting and SentenceTransformer for scoring

## Getting Started

### Prerequisites

- Python 3.8+
- Download the quantized TinyLlama model (`tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`) and place it in the project folder.

### Installation

1. Clone this repository:
    ```
    git clone <your-repo-url>
    cd resume-bot-env
    ```

2. Install dependencies:
    ```
    pip install -r requirements.txt
    ```

### Usage

1. Start the app:
    ```
    streamlit run app.py
    ```

2. Upload your resume and paste the job description in the web interface.

3. Click "Analyze & Rewrite" to see your match score and get an improved resume.

## File Structure

- `app.py` — Streamlit frontend
- `screening_op.py` — Resume scoring and rewriting logic
- `requirements.txt` — Python dependencies
- `.gitignore` — Files and folders to exclude from git
- `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` — (Not in git) Model file for rewriting

