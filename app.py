import gradio as gr
from transformers import pipeline

# -------------------------
# Load Models
# -------------------------

sentiment_model = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

emotion_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None
)

topic_model = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

# Candidate Categories
categories = [
    "Sports",
    "Education",
    "Technology",
    "Politics",
    "Animals",
    "Health",
    "Entertainment",
    "Business",
    "Travel",
    "Food",
    "Science",
    "Environment",
    "Finance",
    "Crime",
    "Shopping",
    "Gaming",
    "Movies",
    "Music",
    "Books",
    "Fashion"
]


def analyze(text):

    sentiment = sentiment_model(text)[0]

    emotions = emotion_model(text)[0]

    topic = topic_model(text, categories)

    emotion_scores = {
        item["label"]: round(item["score"],4)
        for item in emotions
    }

    return (
        sentiment["label"],
        round(sentiment["score"],4),
        topic["labels"][0],
        emotion_scores
    )

demo = gr.Interface(
    fn=analyze,
    inputs=gr.Textbox(lines=5, label="Enter Text"),
    outputs=[
        gr.Text(label="Sentiment"),
        gr.Number(label="Confidence"),
        gr.Text(label="Category"),
        gr.JSON(label="Emotion Scores")
    ],
    title="AI Sentiment & Emotion Analyzer",
    description="Sentiment + Emotion + Topic Classification"
)

demo.launch()