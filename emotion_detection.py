from textblob import TextBlob
from transformers import pipeline

# Function for emotion detection using TextBlob
def emotion_detection_textblob(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return "Positive Emotion"
    elif polarity < 0:
        return "Negative Emotion"
    else:
        return "Neutral Emotion"

# Function for emotion detection using Hugging Face transformers
def emotion_detection_transformers(text):
    # Load a pre-trained model for emotion detection
    emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
    results = emotion_classifier(text)
    return results[0]["label"]

# Main program
if __name__ == "__main__":
    # Get user input
    text_input = input("Enter a sentence to analyze emotion: ")

    # Detect emotion using TextBlob
    basic_emotion = emotion_detection_textblob(text_input)
    print(f"Emotion (TextBlob): {basic_emotion}")

    # Detect emotion using Hugging Face transformers
    advanced_emotion = emotion_detection_transformers(text_input)
    print(f"Emotion (Transformers): {advanced_emotion}")
