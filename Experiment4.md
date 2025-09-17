#samplecode:
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences, to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# --------------------------
# Sample corpus
# --------------------------
data = "Deep learning is amazing. Deep learning builds intelligent systems."

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])

# Generate sequences
words = data.split()
sequences = []
for i in range(1, len(words)):
    seq = words[:i+1]
    sequences.append(' '.join(seq))

# Integer encoding
encoded = tokenizer.texts_to_sequences(sequences)

# Padding
max_len = max([len(x) for x in encoded])
padded = pad_sequences(encoded, maxlen=max_len)

# Split into predictors (X) and labels (y)
X = padded[:, :-1]  # all words except last
y = padded[:, -1]   # last word
y = to_categorical(y, num_classes=len(tokenizer.word_index) + 1)

# --------------------------
# Model
# --------------------------
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1,
              output_dim=10,
              input_length=max_len - 1),
    SimpleRNN(50),
    Dense(len(tokenizer.word_index) + 1, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=200, verbose=0)

print("✅ Model training complete")

# --------------------------
# Prediction Function
# --------------------------
def predict_next_word(model, tokenizer, text, max_len):
    # Encode text
    encoded = tokenizer.texts_to_sequences([text])[0]
    # Pad to match input length
    encoded = pad_sequences([encoded], maxlen=max_len-1, padding='pre')
    # Predict
    y_pred = model.predict(encoded, verbose=0)
    predicted_word_index = np.argmax(y_pred)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# --------------------------
# Test Predictions
# --------------------------
tests = ["Deep learning", "Deep learning is", "Deep learning builds intelligent"]

for t in tests:
    next_word = predict_next_word(model, tokenizer, t, max_len)
    print(f"{t} -> {next_word}")
    #sample output:
    ![Screenshot_2025-09-17-07-49-09-997_com android chrome](https://github.com/user-attachments/assets/b90bbea8-5e41-4f46-90db-65a7e0a72105)
