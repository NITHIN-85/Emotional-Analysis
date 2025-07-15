🤓 Kannada Text Emotion Analysis

📖 Overview

Emotional-Analysis is a machine learning project focused on detecting and classifying emotions in Kannada language text. It uses deep learning models (BERT + LSTM) and custom preprocessing techniques to analyze sentences and predict emotional states such as Joy, Neutral, Fear, Anger, Sadness, Disgust, and Surprise. The project is designed to handle inputs with mixed emotions and provides detailed output on the detected emotional content.

🚀 Features

Emotion Detection Model : Utilizes BERT embeddings with an LSTM classifier to predict emotions from Kannada text.

Custom Preprocessing : Cleans and tokenizes input sentences, splitting on delimiters like ".", "ಆದರೆ", "ಮತ್ತು" for fine-grained analysis.

Weighted Emotion Mapping : Assigns weights to specific emotion keywords in Kannada for more accurate classification.

Mixed Emotion Handling : Capable of detecting and reporting multiple emotions in a single sentence.

Dataset Handling : Includes code for loading, preprocessing, encoding, and splitting datasets for training and testing.

Evaluation Metrics : Calculates accuracy using labeled test data and saves predictions to Excel for further analysis.

🛠️ Tech Stack

- Python 3.x
- PyTorch
- Transformers (HuggingFace)
- Pandas
- scikit-learn

📊 Dataset Information

Prepare your dataset in CSV/Excel format with the following columns:

- `Sentence`: The Kannada text to analyze.
- `Emotion`: The labeled emotion for supervised training.

Example:

| Sentence                                       | Emotion  |
|------------------------------------------------|----------|
| ಜನರು ಇತರರ ಕಡೆಗೆ ಸಹಾನುಭೂತಿಯನ್ನು ಪ್ರದರ್ಶಿಸುವುದನ್ನು ... | Joy      |
| ನಗರವು ಭಾರತದಲ್ಲಿನ ಪ್ರಮುಖ ಸಾರಿಗೆ ಕೇಂದ್ರಗಳಲ್ಲಿ ಒಂದಾಗಿದೆ | Neutral  |

Training the Model

Run `BERT_LSTM.ipynb` to:

- Load and preprocess the data.
- Encode sentences and emotions.
- Split into train/validation sets.
- Train the BERT + LSTM model.
- Evaluate accuracy on a test set.

 Predicting Emotions

Use the function in `test.ipynb`:

```python
sentence = "ರಾತ್ರಿ ಹೊರಗೆ ಹೋಗುವುದಕ್ಕೆ ನನಗೆ ಬಹಳ Fear. ಆದರೆ ನನ್ನ ಸ್ನೇಹಿತನ ನಿರ್ಲಕ್ಷ್ಯದಿಂದ ನನಗೆ ಕೋಪ ಬಂದಿದೆ."
predicted_label = predict_emotion(sentence)
print(f"Predicted Emotion Label: {predicted_label}")
```
Custom Keyword Mapping

The project uses a dictionary mapping Kannada emotion words to model labels and weights for fine-tuned predictions.

Example mapping:

python
{
    'ಉಲ್ಲಾಸ': {'label': 'joy', 'weight': 2.0},
    'ಭಯ': {'label': 'fear', 'weight': 2.0},
    'ಕ್ರೋಧ': {'label': 'anger', 'weight': 2.0},
    'ದುಃಖ': {'label': 'sadness', 'weight': 2.0},
    'ಆಶ್ಚರ್ಯ': {'label': 'surprise', 'weight': 2.0},
    # ... more mappings
}

✅ How It Works
Data Preprocessing : Clean text, remove stopwords, tokenize, and vectorize using TF-IDF.

Model Training : Train SVM classifier on labeled emotion dataset.

Prediction & Evaluation : Predict emotions for new text and evaluate using confusion matrix, accuracy score.

Usage

- Train the model using your own dataset or the provided example.
- Test new sentences for emotion prediction.
- Analyze accuracy and results using the saved Excel output.

Acknowledgments
- Uses HuggingFace Transformers for BERT
- Developed for Kannada language NLP tasks

For more details and examples, refer to the Jupyter notebooks in the repository.
