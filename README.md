# 💬 Customer Emotion Analyzer

**🚀 [Try the live app here](https://aditi-customer-emotions.streamlit.app)**. No installation needed, it opens in your browser.

Customers don't just "like" or "dislike" a product. They feel **angry, disappointed, delighted,
worried, surprised**. This app uses a free AI model to read customer reviews, tag each one with
an emotion, and turn the results into business insight.

Built at the intersection of **consumer psychology** and **AI (Natural Language Processing)**.

## What it does
- Analyzes reviews from a sample file, your own CSV, or pasted text
- Tags each review with one of 7 emotions (anger, disgust, fear, joy, neutral, sadness, surprise),
  based on psychologist Paul Ekman's basic emotions
- Shows the mood mix as a chart, the % of delighted vs unhappy customers, and what each mood
  means for the business
- Lets you read reviews filtered by mood (e.g. only angry customers) and download the results

## Why emotions instead of positive/negative?
A "negative" review can mean very different things: an **angry** customer needs a fast reply,
a **disappointed** one may quietly never buy again, and a **worried** one needs reassurance.
Each calls for a different business action.

## Tech used
- **Hugging Face Transformers**: the pre-trained AI model
  [`j-hartmann/emotion-english-distilroberta-base`](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)
- **PyTorch**: runs the model
- **pandas** and **Streamlit**: data handling and the web app

## How to run

```bash
pip install -r requirements.txt
streamlit run app.py
```

On Windows you can also double-click `start_app.bat`. The first run downloads the AI model
(about 330 MB). After that it works offline.

## Files
- `emotions.py`: loads the AI model and analyzes reviews
- `app.py`: the web app
- `sample_reviews.csv`: 30 sample earphone reviews written for this demo

## Limitations
- Works best on English text
- Like any AI model, it can misread sarcasm and mixed feelings
