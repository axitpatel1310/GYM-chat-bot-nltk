
from flask import Flask, render_template, request
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import random

# Initialize Flask app
app = Flask(__name__)

# Download required NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the CSV file
try:
    df = pd.read_excel('archive/data.xlsx')
except FileNotFoundError:
    print("Error: gymdata.csv not found. Please ensure the file is in the same directory.")
    exit(1)

# Preprocessing function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return []
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in string.punctuation]
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens

# Calculate similarity between user input and exercise data
def get_best_matches(user_input, top_n=3):
    user_tokens = preprocess_text(user_input)
    if not user_tokens:
        return None
    
    similarities = []
    for idx, row in df.iterrows():
        title_tokens = preprocess_text(row['Exercise_Name'])
        muscle_tokens = preprocess_text(row['muscle_gp'])
        combined_tokens = title_tokens + muscle_tokens
        
        # Calculate Jaccard similarity
        common_tokens = set(user_tokens) & set(combined_tokens)
        union_tokens = set(user_tokens) | set(combined_tokens)
        similarity = len(common_tokens) / len(union_tokens) if union_tokens else 0
        
        similarities.append((similarity, idx))
    
    # Sort by similarity and get top matches
    similarities.sort(reverse=True)
    top_matches = similarities[:top_n]
    if top_matches[0][0] < 0.1:  # Threshold for relevance
        return None
    
    return [df.iloc[idx] for _, idx in top_matches]

# Format exercise details for response
def format_exercise(row):
    response = f"<strong>{row['Exercise_Name']}</strong><br>"
    response += f"- <strong>Muscle Group</strong>: {row['muscle_gp']}<br>"
    response += f"- <strong>Equipment</strong>: {row['Equipment']}<br>"
    response += f"- <strong>Rating</strong>: {row['Rating']}<br>"
    response += f"- <strong>Details</strong>: <a href='{row['Description_URL']}'>More Details</a>"
    return response

# Chatbot response function
def get_response(user_input):
    if not user_input.strip():
        return "Please enter a query, like 'ab exercises' or 'beginner core'!"
    
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

# Flask routes
@app.route('/', methods=['GET', 'POST'])
def index():
    response = ""
    user_input = ""
    if request.method == 'POST':
        user_input = request.form.get('query', '')
        response = get_response(user_input)
    return render_template('index.html', response=response, user_input=user_input)

if __name__ == '__main__':
    app.run(debug=True)
