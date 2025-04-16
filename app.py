from flask import Flask, render_template, request
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import random
import os
import logging

# Initialize Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set NLTK data path
nltk.data.path.append(os.path.join(os.getcwd(), 'nltk_data'))
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    logger.info("NLTK data found successfully.")
except LookupError as e:
    logger.error(f"NLTK data not found: {e}")
    logger.warning("Using fallback tokenization.")

# Load the Excel file
df = None
try:
    df = pd.read_excel('data.xlsx')
    logger.info("Excel file loaded successfully.")
    logger.info(f"Columns in data.xlsx: {list(df.columns)}")
except FileNotFoundError:
    logger.error("data.xlsx not found.")
except Exception as e:
    logger.error(f"Failed to load data.xlsx: {e}")

# Preprocessing function
try:
    stop_words = set(stopwords.words('english')) if 'stopwords' in nltk.data.path else set()
    lemmatizer = WordNetLemmatizer() if 'wordnet' in nltk.data.path else None
except Exception as e:
    logger.error(f"Error initializing NLTK resources: {e}")
    stop_words = set()
    lemmatizer = None

def preprocess_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return []
    try:
        tokens = word_tokenize(text.lower()) if 'punkt' in nltk.data.path else text.lower().split()
        tokens = [token for token in tokens if token not in string.punctuation]
        tokens = [token for token in tokens if token not in stop_words]
        tokens = [lemmatizer.lemmatize(token) if lemmatizer else token for token in tokens]
        return tokens
    except Exception as e:
        logger.error(f"Error in preprocess_text: {e}")
        return []

# Calculate similarity between user input and exercise data
def get_best_matches(user_input, top_n=3):
    if df is None:
        logger.warning("No database available.")
        return None
    try:
        user_tokens = preprocess_text(user_input)
        if not user_tokens:
            return None
        similarities = []
        for idx, row in df.iterrows():
            title_tokens = preprocess_text(row.get('Exercise_Name', ''))
            muscle_tokens = preprocess_text(row.get('muscle_gp', ''))
            combined_tokens = title_tokens + muscle_tokens
            common_tokens = set(user_tokens) & set(combined_tokens)
            union_tokens = set(user_tokens) | set(combined_tokens)
            similarity = len(common_tokens) / len(union_tokens) if union_tokens else 0
            similarities.append((similarity, idx))
        similarities.sort(reverse=True)
        top_matches = similarities[:top_n]
        if top_matches and top_matches[0][0] < 0.1:
            return None
        return [df.iloc[idx] for _, idx in top_matches]
    except Exception as e:
        logger.error(f"Error in get_best_matches: {e}")
        return None

# Format exercise details for response
def format_exercise(row):
    try:
        response = f"<strong>{row.get('Exercise_Name', 'Unknown Exercise')}</strong><br>"
        response += f"- <strong>Muscle Group</strong>: {row.get('muscle_gp', 'N/A')}<br>"
        response += f"- <strong>Equipment</strong>: {row.get('Equipment', 'N/A')}<br>"
        response += f"- <strong>Rating</strong>: {row.get('Rating', 'N/A')}<br>"
        response += f"- <strong>Details</strong>: <a href='{row.get('Description_URL', '#')}'>More Details</a>"
        return response
    except Exception as e:
        logger.error(f"Error in format_exercise: {e}")
        return "Error formatting exercise data."

# Chatbot response function
def get_response(user_input):
    if not user_input.strip():
        return "Please enter a query, like 'ab exercises' or 'beginner core'!"
    if df is None:
        return "Sorry, the exercise database is unavailable. Please try again later."
    try:
        matches = get_best_matches(user_input)
        if matches:
            response = "Here are some exercises that match your query:<br><br>"
            for match in matches:
                response += format_exercise(match) + "<br>"
            return response
        else:
            return random.choice([
            "I couldn't find a match. Try specifying a muscle group (e.g., abs), equipment (e.g., kettlebell), or level (e.g., beginner)!",
            "No exercises found. Could you clarify, like 'ab workouts' or 'barbell exercises'?",
                "Hmm, try something like 'core exercises for beginners' or 'kettlebell abs'!"
        ])
    except Exception as e:
        logger.error(f"Error in get_response: {e}")
    return "An error occurred. Please try again with a different query."

# Flask routes
@app.route('/', methods=['GET', 'POST'])
def index():
    response = "Welcome to GymBot! Enter a query above to get started."
    user_input = ""
    try:
        if request.method == 'POST':
            user_input = request.form.get('query', '')
            response = get_response(user_input)
    except Exception as e:
        logger.error(f"Error in index route: {e}")
        response = "An error occurred. Please try again."
    return render_template('index.html', response=response, user_input=user_input)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)