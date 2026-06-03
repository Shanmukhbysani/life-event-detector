# life-event-detector

Streamlit prototype for detecting likely financial life events from anonymized transaction clusters.

## Features

- Privacy-first transaction analysis with PII stripped before inference.
- Groq-backed structured JSON output when an API key is available.
- Local heuristic fallback so the app still works without external credentials.
- Custom transaction injection for quick prototyping.

## Run Locally

1. Create a virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Launch the app with `streamlit run app.py`.

## Notes

- The app analyzes merchant strings, dates, and amounts only.
- To use the hosted model path, add a Groq API key in the sidebar.