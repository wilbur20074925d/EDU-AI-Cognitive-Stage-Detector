# EDU-AI Cognitive Stage Detector (Gemini)

Hybrid rule + Gemini Embeddings to estimate Clarify/Plan/Execute/Verify/Reflect stage percentages for learner messages, with temporal smoothing and weighted aggregation. Streamlit UI + advanced Plotly analytics.

## Local Run
```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
cp .env.example .env  # then put your real key in .env
streamlit run app_gemini.py