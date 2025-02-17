import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import nltk
import re
import json
import swifter 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

nltk.download('stopwords')
from nltk.corpus import stopwords

print("Step 1: Loading dataset...")
yelp_data_path = r"yelp_dataset\yelp_academic_dataset_review.json"  

reviews = []
with open(yelp_data_path, "r", encoding="utf-8") as file:
    for i, line in enumerate(file):
        reviews.append(json.loads(line))
        if i >= 50000:  
            break

df = pd.DataFrame(reviews)
print(f"Loaded {len(df)} reviews.")

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  
    text = re.sub(r'\s+', ' ', text).strip()  
    text = ' '.join([word for word in text.split() if word not in stop_words]) 
    return text

print("Step 2: Cleaning text data (using swifter)...")
df['clean_text'] = df['text'].swifter.apply(clean_text)
print("Text cleaning completed!")

df['sentiment'] = df['stars'].apply(lambda x: 1 if x > 3 else 0)

print("Step 3: Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['sentiment'], test_size=0.2, random_state=42)

print("Step 4: Tokenizing text data...")
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_length = 100
X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post')

y_train = np.array(y_train)
y_test = np.array(y_test)

print("Data preprocessing completed!")


print("Step 5: Building LSTM model...")
model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=max_length),
    LSTM(128, return_sequences=True),
    Dropout(0.5),
    LSTM(64),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

print("Step 6: Training LSTM model...")
history = model.fit(X_train_pad, y_train, epochs=5, batch_size=64, validation_data=(X_test_pad, y_test))


print("Step 7: Evaluating model performance...")
loss, accuracy = model.evaluate(X_test_pad, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

def predict_sentiment(text):
    text_cleaned = clean_text(text)
    text_seq = tokenizer.texts_to_sequences([text_cleaned])
    text_pad = pad_sequences(text_seq, maxlen=max_length, padding='post')
    prediction = model.predict(text_pad)[0][0]
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    return sentiment

print("Testing model on sample reviews...")
test_review_1 = "The food was absolutely amazing! Loved it."
print(f"Review: {test_review_1}\nSentiment: {predict_sentiment(test_review_1)}")

test_review_2 = "The service was terrible, I waited 40 minutes for my order."
print(f"Review: {test_review_2}\nSentiment: {predict_sentiment(test_review_2)}")
