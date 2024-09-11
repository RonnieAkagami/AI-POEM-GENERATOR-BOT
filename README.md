

# AI POEM GENERATOR BOT

This project demonstrates text generation using an LSTM model trained on a portion of Shakespeare's works. The model generates text character by character, based on the input seed, allowing for creative and random sequences of Shakespearean-style writing.

## Project Overview

- **Model:** Long Short-Term Memory (LSTM) neural network
- **Dataset:** A portion of Shakespeare's text (~500,000 characters)
- **Objective:** Generate new text sequences that resemble Shakespearean writing
- **Framework:** TensorFlow with Keras

## How It Works

1. The dataset is processed into sequences of 40 characters, with each sequence predicting the next character.
2. An LSTM model is trained on the data.
3. After training, the model is used to generate new text by predicting one character at a time, given a seed sequence.
4. A temperature parameter is used to control the randomness of predictions.

## Files

- `text_generator.py`: The main script to load the model and generate text.
- `textgenerator.model.keras`: Pre-trained LSTM model used for generating text.
- `shakespeare.txt`: Dataset of Shakespeare's text (downloaded automatically by the script).
- `README.md`: This file.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/shakespeare-text-generator.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the text generation script:
   ```bash
   python text_generator.py
   ```
4. The script will output a generated sequence of text.

## Model Usage

You can customize the length of the generated text and the randomness through the `generate_text()` function:

```python
print(generate_text(300, 0.2))
```

- **Length**: Controls how many characters are generated.
- **Temperature**: Controls the randomness. Lower values (e.g., 0.2) result in more predictable text, while higher values (e.g., 1.0) produce more random sequences.

## Dependencies

- Python 3.x
- TensorFlow 2.x
- NumPy

You can install the dependencies via `requirements.txt`:
```bash
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

Feel free to modify the `README.md` as per your specific requirements!
