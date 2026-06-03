# life-event-detector

A Streamlit app for detecting likely financial life events from anonymized transaction clusters.

The app is designed as a lightweight prototype for banking and fintech teams that want to test how merchant patterns, dates, and amounts can be mapped to major life events such as a new baby, relocation, home purchase, job change, or marriage. It supports both a Groq-backed LLM workflow and a local heuristic fallback so it remains usable even when no API key is available.

## What It Does

- Loads a small synthetic transaction dataset for quick prototyping.
- Lets you inspect activity by customer and run a structured analysis.
- Returns a JSON response with the detected event, confidence score, evidence, and a recommended action.
- Accepts new transactions through the UI so you can simulate different customer scenarios.

## Key Features

- Privacy-first analysis with customer identifiers and other direct PII excluded from model input.
- Structured output using Pydantic so the result stays machine-readable.
- Hosted LLM support via Groq when you want higher-quality inference.
- Local heuristic fallback when no API key is configured.
- Simple sandbox workflow for testing merchant patterns and customer journeys.

## Project Structure

- [app.py](app.py) contains the Streamlit UI, transaction data, heuristic logic, and Groq integration.
- [requirements.txt](requirements.txt) lists the Python dependencies.
- [README.md](README.md) documents setup and usage.

## How It Works

1. A customer transaction cluster is selected from the sidebar-driven sandbox.
2. The app strips the data down to date, merchant, and amount before analysis.
3. If a Groq API key is provided, the sanitized transactions are sent to the model with a strict JSON schema.
4. If no API key is provided, the app falls back to a local keyword-based detector.
5. The result is displayed as a readable summary with supporting evidence and a recommended action.

## Requirements

- Python 3.10 or newer.
- A Groq API key if you want to use the hosted model path.
- Internet access for installing Python packages.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies.
3. Start the app.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

On Windows PowerShell, activate the environment with `.\.venv\Scripts\Activate.ps1` before installing packages.

## Using The App

1. Open the app in your browser after launching Streamlit.
2. Choose a customer ID in the Live Sandbox Analytics tab.
3. Optionally paste a Groq API key into the sidebar.
4. Click the analysis button to generate a life-event classification.
5. Use the Add Custom Transaction tab to simulate new spending or income patterns.

## Example Scenarios

- New Baby: recurring diaper, formula, stroller, or nursery-related spending.
- Relocation: moving services, rental-related merchants, or housing transition patterns.
- Home Purchase: escrow, mortgage, realtor, inspection, or home improvement activity.
- Job Change: payroll references, onboarding terms, or new employer-related deposits.
- Marriage: wedding, bridal, engagement, venue, or honeymoon-related purchases.

## Deployment Notes

- The app is suitable for local demos and internal prototypes.
- If you deploy it, keep the Groq key in a secure secret store instead of hardcoding it.
- Only sanitized transaction metadata should be passed to any external model.

## Limitations

- The bundled dataset is synthetic and intentionally small.
- The heuristic fallback is rule-based, so it is less accurate than a tuned model.
- The app does not perform real customer segmentation or production-grade fraud/privacy controls.

## Extending The Project

- Replace the synthetic dataset with a secure internal data source.
- Expand the event taxonomy with more life events and a richer schema.
- Add automated tests for the heuristic analyzer and data sanitation layer.
- Wire the app into a backend service if you want to separate UI from inference logic.

## License

No license has been specified yet.