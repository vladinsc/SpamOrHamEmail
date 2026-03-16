import pickle
import re
def clean_text(text):
    """Removes the massive URLs and cleans up the newlines."""
    # Replaces the entire URL with JUST the domain name
    text = re.sub(r'https?://(?:www\.)?([a-zA-Z0-9.-]+)\S*', r' \1 ', text)
    # 2. Remove the [tel:...] tags
    text = re.sub(r'\[tel:\d+\]', '', text)
    # 3. Replace newline characters with spaces
    text = text.replace('\n', ' ')

    #remove stop words
    text = text.lower()
    stop_words = [
        'si', 'de', 'la', 'in', 'cu', 'o', 'un', 'din', 'pe', 'pentru',
        'ca', 'mai', 'ce', 'sunt', 'a', 'au', 'al', 'ale', 'lor', 'lui',
        'este', 'sa', 'nu', 'se', 'care', 'prin', 'spre', 'dar', 'iar'
    ]
    text = ' '.join([word for word in text.split() if word not in stop_words])
    # 4. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = tf.keras.models.load_model('ro_spam_model.keras')
with open('ro_tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
MAX_LENGTH = 200

def predict_spam(text):
    clean_body = clean_text(text)
    sequence = tokenizer.texts_to_sequences([clean_body])
    padded = pad_sequences(sequence, maxlen=MAX_LENGTH, padding='post', truncating='post')
    prediction = model.predict(padded, verbose=0)[0][0]
    return prediction

