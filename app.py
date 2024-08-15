from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from gensim import corpora
from gensim.models import LdaModel
import io
import base64
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)

# Load the pre-trained model and vectorizer (done once when the app starts)
pipeline = joblib.load('pretrained_model.pkl')
model = pipeline.named_steps['logisticregression']
vectorizer = pipeline.named_steps['countvectorizer']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Step 1: Load the uploaded CSV file
        feedback_file = request.files['file']
        df = pd.read_csv(feedback_file)

        # Step 2: Vectorize the uploaded data using the pretrained vectorizer
        X = vectorizer.transform(df['Feedback'])

        # Step 3: Predict Sentiments using the pretrained model
        df['Predicted_Sentiment'] = model.predict(X)

        # Save the processed data for further use (e.g., visualization)
        df.to_csv('uploaded_data.csv', index=False)

        return redirect(url_for('options'))

@app.route('/options')
def options():
    return render_template('options.html')

@app.route('/visualize')
def visualize():
    df = pd.read_csv('uploaded_data.csv')

    # Print out the sentiment counts for debugging
    print(df['Predicted_Sentiment'].value_counts())

    # Plot Sentiment Distribution
    sentiment_counts = df['Predicted_Sentiment'].value_counts()
    plt.figure(figsize=(10, 6))
    sentiment_counts.plot(kind='bar', color=['#4CAF50', '#FFC107', '#F44336'])
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig('static/sentiment_distribution.png')
    plt.close()

    return render_template('visualize.html', image='sentiment_distribution.png')

@app.route('/wordcloud')
def wordcloud():
    df = pd.read_csv('uploaded_data.csv')

    # Generate Word Cloud for all feedback text
    text = " ".join(feedback for feedback in df['Feedback'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig('static/wordcloud.png')
    plt.close()

    return render_template('wordcloud.html', image='wordcloud.png')


@app.route('/topics')
def topics() :
    df = pd.read_csv('uploaded_data.csv')

    # Prepare data for topic modeling
    stop_words = set(stopwords.words('english'))
    texts = df['Feedback'].apply(lambda x : [word for word in x.lower().split() if word not in stop_words])

    # Create dictionary and corpus for topic modeling
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Train LDA model
    lda_model = LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)

    # Generate word clouds for each topic and prepare descriptions
    topic_data = []
    topics = lda_model.show_topics(num_words=10, formatted=False)

    for i, topic in enumerate(topics) :
        word_probabilities = dict(topic[1])

        # Create a word cloud image
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(
            word_probabilities)
        img = io.BytesIO()
        wordcloud.to_image().save(img, format='PNG')
        img.seek(0)
        img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

        # Create a description based on the top words
        top_words = [word for word, _ in topic[1]]
        description = f"Topic {i + 1}: {' '.join(top_words)}"

        topic_data.append({
            'image' : img_base64,
            'description' : description
        })

    return render_template('topics.html', topic_data=topic_data)


if __name__ == '__main__':
    app.run(debug=True)
