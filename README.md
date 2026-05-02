📊 Sentiment & Aspect-Based Analysis of Bank Customer Reviews

A deep learning-based Natural Language Processing (NLP) project for analyzing customer reviews of banks using IndoBERT and Bidirectional LSTM (Bi-LSTM). This project focuses on extracting sentiment and service aspects from Indonesian-language reviews and predicting customer ratings.

🚀 Overview

Customer reviews contain valuable insights into service quality, but analyzing them manually is inefficient. This project provides an automated solution to:

Classify sentiment (positive, negative, neutral)
Identify service aspects mentioned in reviews
Predict customer ratings (1–5)
Analyze worst-performing service aspects over time

The system is designed specifically for Indonesian text, leveraging transformer-based embeddings and deep learning models.

🧠 Methodology
1. Data Collection
Bank customer reviews collected from Google Maps
Includes both labeled and unlabeled datasets
2. Preprocessing
Case folding
Tokenization
Stopword removal (Sastrawi)
Text normalization
3. Feature Extraction
IndoBERT used to generate contextual embeddings
4. Modeling
🔹 Aspect Classification
Model: Bidirectional LSTM (Bi-LSTM)
Multi-label classification
Classes: Positive, Negative, Not Mentioned (per aspect)
🔹 Rating Prediction
Model: MLP (Multi-Layer Perceptron)
Input: Output from aspect classification
Output: Rating (1–5)
🏷️ Service Aspects

The model analyzes the following aspects:

Availability (Ketersediaan)
Service (Pelayanan)
Place (Tempat)
Comfort (Kenyamanan)
Queue (Antrian)
Information (Informasi)
Security (Keamanan)
Complaint Handling (Pengaduan)
Accessibility (Aksesibilitas)
⚙️ Techniques & Optimization
✅ IndoBERT for contextual feature extraction
✅ Bi-LSTM for sequence modeling
✅ SMOTE / ADASYN for handling class imbalance
✅ Stratified K-Fold Cross Validation
✅ Early Stopping for efficient training
✅ Threshold tuning for multi-label classification
📈 Outputs

The system generates:

📄 Labeled dataset (sentiment & aspects)
⭐ Predicted ratings per review
📊 Worst service aspects analysis per bank per year
📉 Evaluation metrics (accuracy, precision, recall, F1-score)
🛠️ Tech Stack
Python
- TensorFlow / Keras
- HuggingFace Transformers
- Pandas, NumPy
- Scikit-learn
- Sastrawi (Indonesian NLP preprocessing)

📊 Results
The model demonstrates strong capability in handling Indonesian NLP tasks and multi-label classification. Performance is evaluated using:
Accuracy
Precision
Recall
F1-Score

💡 Key Contributions
End-to-end NLP pipeline for Indonesian bank review analysis
Combination of transformer embeddings + Bi-LSTM
Multi-aspect sentiment classification
Integration of aspect results into rating prediction
Temporal analysis of service quality

📌 Future Improvements
Improve model generalization across different domains
Explore transformer-based fine-tuning (full IndoBERT training)
Deploy as a web-based dashboard
Real-time review analysis system.
