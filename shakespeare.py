import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop

# Load Shakespeare text data
filepath = tf.keras.utils.get_file("shakespeare.txt",
                                   "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt")
text = open(filepath, "rb").read().decode(encoding='utf-8').lower()

# Use a slice of the text
text = text[300000:800000]

# Create a character set
characters = sorted(set(text))

# Create mappings from characters to indices and vice versa
char_to_index = dict((c, i) for i, c in enumerate(characters))
index_to_char = dict((i, c) for i, c in enumerate(characters))

SEQ_NO = 40  # Length of sequences for training
STEP_SIZE = 3  # Step size to create overlapping sequences

model = tf.keras.models.load_model('textgenerator.model.keras')


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probs = np.random.multinomial(1, preds, 1)
    return np.argmax(probs)


def generate_text(length, temperature):
    start_index = random.randint(0, len(text) - SEQ_NO - 1)
    generated = ''
    sentence = text[start_index: start_index + SEQ_NO]
    generated += sentence
    for i in range(length):
        x = np.zeros((1, SEQ_NO, len(characters)))

        # Rename loop variable from `characters` to `char`
        for t, char in enumerate(sentence):
            x[0, t, char_to_index[char]] = 1

        predictions = model.predict(x, verbose=0)[0]
        next_index = sample(predictions, temperature)
        next_letter = index_to_char[next_index]

        generated += next_letter
        sentence = sentence[1:] + next_letter
    return generated


print(generate_text(300, 0.2))

print(generate_text(300, 0.6))



