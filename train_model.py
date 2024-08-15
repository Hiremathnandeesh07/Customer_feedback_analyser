import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import joblib
import nltk
from nltk.corpus import movie_reviews
from sklearn.model_selection import train_test_split

# Download the movie_reviews dataset
nltk.download('movie_reviews')

# Load dataset
def load_data():
    # Load data from nltk's movie_reviews corpus
    documents = [(list(movie_reviews.words(fileid)), category)
                 for category in movie_reviews.categories()
                 for fileid in movie_reviews.fileids(category)]

    # Convert to DataFrame
    df = pd.DataFrame(documents, columns=['text', 'label'])
    df['text'] = df['text'].apply(lambda x: " ".join(x))
    return df

# Train the model
def train_model(df):
    # Create a pipeline that combines a CountVectorizer with a LogisticRegression classifier
    pipeline = make_pipeline(
        CountVectorizer(),
        LogisticRegression()
    )

    # Train the model
    pipeline.fit(df['text'], df['label'])

    # Save the model and vectorizer
    joblib.dump(pipeline, 'pretrained_model.pkl')

# Load and prepare data
df = load_data()

# Train and save the model
train_model(df)

print("Model training complete and saved to 'pretrained_model.pkl'.")
