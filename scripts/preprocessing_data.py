import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the dataset
data = pd.read_csv('C:/Users/AA/Desktop/Text_Sentiment_Analysis/dataset/IMDB Dataset.csv')

# Preprocessing function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Apply preprocessing
data['processed_text'] = data['Review'].apply(preprocess_text)

# Feature Engineering using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['processed_text'])

# Convert sentiment to binary (1 for positive, 0 for negative)
y = data['Sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

print(X.shape, y.shape)  # Output the shape of the feature matrix and target vector
