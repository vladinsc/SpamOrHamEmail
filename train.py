from datasets import load_dataset
import pandas as pd
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils.class_weight import compute_class_weight
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

dataset = load_dataset("readerbench/ro-business-emails")


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
def extract_specific_annotation(choices_list, target_name):
    """Digs into the dictionary and extracts the 1/0 value for a specific target."""
    for choice in choices_list:
        if choice['name'] == target_name:
            return 1 if choice['value'] == 'True' else 0
    return 0
def clean_data(df: pd.DataFrame)-> pd.DataFrame:
    df['text'] = df['data'].apply(lambda x: x['body'])
    df['clean_text'] = df['text'].apply(clean_text)
    # 1. Extract the original Spam label
    df['label_spam'] = df['annotation'].apply(
        lambda x: extract_specific_annotation(x['choices'], 'Is SPAM')
    )

    # 2. Extract the "Automatically Generated" label
    df['label_auto'] = df['annotation'].apply(
        lambda x: extract_specific_annotation(x['choices'], 'Is Automatically Generated')
    )

    # 3. Extract the "Needs Action" label
    df['label_action'] = df['annotation'].apply(
        lambda x: extract_specific_annotation(x['choices'], 'Needs Action from User')
    )

    df_final = df[['clean_text', 'label_spam', 'label_auto', 'label_action']]
    return df_final



#Cleaning the data
df_train = clean_data(dataset['train'].to_pandas())
df_test = clean_data(dataset['test'].to_pandas())
df_val = clean_data(dataset['val'].to_pandas())

tokenizer = Tokenizer(num_words = 5000, oov_token = '<OOV>')
x_train = df_train['clean_text']
y_train = df_train['label_spam']
unique_classes = np.unique(y_train)
weights = compute_class_weight(
    class_weight='balanced',
    classes=unique_classes,
    y=y_train
)
class_weights_dict = dict(zip(unique_classes, weights))
x_validate = df_val['clean_text']
y_validate = df_val['label_spam']

x_test = df_test['clean_text']
y_test = df_test['label_spam']

tokenizer.fit_on_texts(x_train)

train_sequences = tokenizer.texts_to_sequences(x_train)
validate_sequences = tokenizer.texts_to_sequences(x_validate)
test_sequences = tokenizer.texts_to_sequences(x_test)

max_len = 200  # Maximum sequence length
train_sequences = pad_sequences(train_sequences, maxlen=max_len, padding='post', truncating='post')
test_sequences = pad_sequences(test_sequences, maxlen=max_len, padding='post', truncating='post')
validate_sequences = pad_sequences(validate_sequences, maxlen=max_len,padding='post', truncating='post')

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(max_len,)),
    tf.keras.layers.Embedding(input_dim=5000, output_dim=32),
    tf.keras.layers.SpatialDropout1D(0.3), # Drops 0.3 ranndom per batch
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32,recurrent_dropout=0.2)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0001)
print("\nStarting the training process...")
history = model.fit(
    train_sequences,
    y_train,
    validation_data=(validate_sequences, y_validate),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weights_dict
)

test_loss, test_accuracy = model.evaluate(test_sequences, y_test)
print('Test Loss :',test_loss)
print('Test Accuracy :',test_accuracy)

raw_predictions = model.predict(test_sequences)

# 2. Convert probabilities into hard 1s and 0s
# If the model is more than 50% sure, we label it Spam
y_pred = (raw_predictions > 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred)
# tn = True Negatives, fp = False Positives, fn = False Negatives, tp = True Positives
tn, fp, fn, tp = cm.ravel()


total_normal = tn + fp
total_spam = fn + tp
misclassified_normal_rate = (fp / total_normal) * 100
misclassified_spam_rate = (fn / total_spam) * 100
print("\n--- ERROR RATES ---")
print(f"Misclassified Normal Emails (Sent to Spam): {misclassified_normal_rate:.2f}% ({fp} out of {total_normal})")
print(f"Misclassified Spam Emails (Slipped Through): {misclassified_spam_rate:.2f}% ({fn} out of {total_spam})")

print("\n--- DETAILED REPORT ---")
print(classification_report(y_test, y_pred, target_names=["Normal (0)", "Spam (1)"]))
plt.figure(figsize=(8, 6))
# fmt='d' ensures it prints whole numbers, not scientific notation
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Normal", "Spam"],
            yticklabels=["Normal", "Spam"])

plt.ylabel('Actual Truth')
plt.xlabel('Model Prediction')
plt.title('Spam Classifier Confusion Matrix')
plt.show()

import pickle

print("\nExporting the final production model...")

# 1. Save the Neural Network Model
model.save('ro_spam_model.keras')

# 2. Save the Tokenizer
with open('ro_tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("SUCCESS! Model and Tokenizer saved to your project folder.")
