import nltk
import pandas as pd
import re
import os
from flask import Flask, render_template, request, jsonify

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS to allow frontend-backend communication

# Download necessary NLTK packages (run once)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize NLTK's lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()

# Custom stopwords - keeping key technical terms while removing general words
custom_stopwords = {
    "python", "java", "developer", "machine", "data", "engineer", "sql", "aws", 
    "docker", "kubernetes", "deep", "learning", "ai", "model", "cloud", "flask", 
    "django", "javascript", "neural", "training", "regression", "clustering", 
    "rest", "api", "project", "experience", "automation", "implementation", 
    "testing", "performance", "ci/cd", "full stack", "react", "node.js"
}
stop_words = set(stopwords.words("english")) - custom_stopwords

# Text preprocessing function
def preprocess_text(text):
    text = re.sub(r"[^\w\s]", "", text.lower())  # Remove punctuation, lowercase text
    words = word_tokenize(text)  # Tokenize the text
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# Load resume dataset and preprocess it
def load_and_process_resumes(csv_file_path):
    if not os.path.isfile(csv_file_path):
        return None

    # Load CSV file and handle missing values
    df = pd.read_csv(csv_file_path, delimiter="\t", encoding='utf-8')
    df.fillna({"Education": "Missing", "Projects": "Missing", "Experience": "0"}, inplace=True)
    df["processed_text"] = df["Projects"].apply(preprocess_text)  # Apply preprocessing
    return df

# Function to rank resumes based on job description
def rank_resumes(job_description, df):
    job_description_processed = preprocess_text(job_description)  # Preprocess job description

    # TF-IDF vectorization for resumes and job description
    vectorizer = TfidfVectorizer()
    resume_vectors = vectorizer.fit_transform(df["processed_text"])
    job_vector = vectorizer.transform([job_description_processed])  # Preprocess before vectorizing

    # Calculate cosine similarities and add scores to DataFrame
    cosine_similarities = cosine_similarity(job_vector, resume_vectors).flatten()
    df["resume_score(%)"] = (cosine_similarities * 100).round(2)
    top_resumes = df.sort_values(by="resume_score(%)", ascending=False).head(5)
    return top_resumes

# API route to rank resumes based on job description (JSON request)
@app.route('/rank', methods=['GET', 'POST'])
def rank():
    csv_file_path = os.path.join(os.getcwd(), "datasets", "ResumeDataSetFinal.csv")
    df = load_and_process_resumes(csv_file_path)

    if df is None:
        return render_template('error.html', error_message="Resume dataset not found!")

    if request.method == 'POST':
        job_description = request.get_json().get('job_description', '')
        top_resumes = rank_resumes(job_description, df)
        result = top_resumes[["Name", "resume_score(%)"]].to_dict(orient='records')
        return jsonify(result)

    return render_template('form.html')

if __name__ == '__main__':
    app.run(debug=True)
