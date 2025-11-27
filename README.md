# BladeGuard AI (Streamlit)

Streamlit port of the BladeGuard AI app. It uses the same Gemini prompt/model/schema defined in `services/geminiService.ts` to analyze blade images, score health, and draw normalized (0–1000) bounding boxes for detected defects.

## Features
- Upload a blade image (standard or binocular) and run Gemini 3 Pro analysis.
- Annotated preview with bounding boxes and labels in the same coordinate system as the React app.
- Executive summary with health score, timestamp, and concise findings.
- Detailed defect cards: severity, confidence, location, description, and recommended action.

## Prerequisites
- Python 3.10+
- Gemini API key (`GEMINI_API_KEY` or `API_KEY`)

## Setup
1. Create and activate a virtual environment (recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Provide your Gemini key. Either export it:
   ```bash
   export GEMINI_API_KEY=your_key_here
   ```
   or place it in `.env`/`.env.local` (auto-loaded).

## Run
```bash
streamlit run app.py
```

## Notes
- Prompt, model (`gemini-3-pro-preview`), schema, and temperature are unchanged from the TypeScript service.
- React source remains for reference; `app.py` is the primary entry point for the Streamlit app.
