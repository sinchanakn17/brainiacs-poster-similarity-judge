# Brainiacs Poster Similarity Judge

A Python/Streamlit tool for comparing an original challenge image with a participant-generated poster.

## Run

1. Install Python 3.10+.
2. Open a terminal in this folder.
3. Run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

4. Open the local Streamlit URL shown in the terminal.
5. Upload the original image and the participant poster.

## Scoring

- Semantic / CLIP: 50%
- Structure / SSIM: 20%
- Color histogram: 10%
- ORB feature matching: 20%

If CLIP cannot be loaded, the app automatically falls back to the non-AI metrics.

## Important judging note

This is a similarity aid, not an absolute measure of creativity or correctness. For your event, decide the scoring rubric before the competition and apply it consistently to all teams.
