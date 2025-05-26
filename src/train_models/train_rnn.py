import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import TextVectorization

from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam

# Read dataset
DATASET_FOLDER = 'dataset/nusaX'

train_df = pd.read_csv(f'{DATASET_FOLDER}/train.csv')
valid_df = pd.read_csv(f'{DATASET_FOLDER}/valid.csv')
test_df = pd.read_csv(f'{DATASET_FOLDER}/test.csv')

# Encode labels
label_encoder = LabelEncoder()
train_df['label'] = label_encoder.fit_transform(train_df['label'])
valid_df['label'] = label_encoder.transform(valid_df['label'])
test_df['label'] = label_encoder.transform(test_df['label'])

# Split dataset
X_train, y_train = train_df['text'].values, train_df['label'].values
X_valid, y_valid = valid_df['text'].values, valid_df['label'].values
X_test, y_test   = test_df['text'].values,  test_df['label'].values

# Vectorization
vocab_size = 10000
sequence_length = 100

vectorizer = TextVectorization(
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length
)

vectorizer.adapt(X_train)

X_train_vec = vectorizer(X_train)
X_valid_vec = vectorizer(X_valid)
X_test_vec = vectorizer(X_test)

# Build RNN model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128, mask_zero=True),
    SimpleRNN(64, return_sequences=True),
    SimpleRNN(64),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(),
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    X_train_vec,
    y_train,
    validation_data=(X_valid_vec, y_valid),
    epochs=10,
    batch_size=32
)

# # Save model
# model.save_weights("nusax_rnn.weights.h5")

weights = model.get_weights()
model.summary()
for i, w in enumerate(weights):
    print(f"Weight {i}: shape {w.shape}")
