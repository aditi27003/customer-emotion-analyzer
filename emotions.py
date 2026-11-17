"""
emotions.py  -  the AI part.

We use a free, ready-trained AI model from Hugging Face:
  "j-hartmann/emotion-english-distilroberta-base"
It reads a sentence and says which of 7 basic emotions it shows:
  anger, disgust, fear, joy, neutral, sadness, surprise
(These come from psychologist Paul Ekman's basic emotions, plus 'neutral'.)

The first run downloads the model (about 330 MB). After that it works offline.
"""

import pandas as pd

MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

# What each emotion usually means when a CUSTOMER writes it.
EMOTION_INFO = {
    "joy":      ("😊", "Delighted", "Happy customers. Use their words in marketing."),
    "neutral":  ("😐", "Neutral", "Factual comments. Nothing urgent."),
    "surprise": ("😲", "Surprised", "Expectations broken (good or bad). Read these closely."),
    "sadness":  ("😢", "Disappointed", "Let down by the product. Risk of not buying again."),
    "anger":    ("😠", "Angry", "Frustrated customers. Reply fast to stop bad word-of-mouth."),
    "disgust":  ("🤢", "Disgusted", "Strong rejection, often about quality or hygiene."),
    "fear":     ("😨", "Worried", "Concerned about safety, fraud or reliability."),
}
NEGATIVE = ["anger", "disgust", "sadness", "fear"]


def load_model():
    """Load the AI model once. (Imported here so the rest of the app loads fast.)"""
    from transformers import pipeline
    return pipeline("text-classification", model=MODEL_NAME, truncation=True)


def analyze(model, texts: list[str]) -> pd.DataFrame:
    """Run every review through the model. Returns one row per review."""
    texts = [str(t).strip() for t in texts if str(t).strip()]
    predictions = model(texts, batch_size=16)
    rows = []
    for text, p in zip(texts, predictions):
        label = p["label"]
        emoji, meaning, _ = EMOTION_INFO.get(label, ("", label, ""))
        rows.append({
            "Review": text,
            "Emotion": label,
            "Customer mood": f"{emoji} {meaning}",
            "Confidence %": round(p["score"] * 100),
        })
    return pd.DataFrame(rows)


def summary(results: pd.DataFrame) -> dict:
    """Headline numbers for the dashboard."""
    total = len(results)
    share = results["Emotion"].value_counts(normalize=True) * 100
    return {
        "total": total,
        "happy": share.get("joy", 0.0),
        "unhappy": sum(share.get(e, 0.0) for e in NEGATIVE),
        "top": share.idxmax() if total else None,
        "share": share,
    }
