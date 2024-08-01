"# LLMs-Generated-Text-Detection" 

---

## Project Overview

This project aims to distinguish between human-written text and AI-generated text using machine learning techniques. We utilized logistic regression, random forest, and the BERT model to achieve high accuracy in text classification.

## Datasets

1. **LLM-Detect AI Generated Text (DAIGT):**
   - Contains over 28,000 essays written by students and AI.
   - Attributes: "text" and "generated".
   - Target class: 1 represents LLM-generated text, 0 represents human-generated text.
   - Size: 62 MB in CSV format.
   - Balanced classes.

2. **Augmented Data for LLM - Detect AI Generated Text:**
   - Contains over 86,000 essays gathered from multiple sources on Kaggle.
   - Attributes: "text" and "label".
   - Target class: 1 represents LLM-generated text, 0 represents human-generated text.
   - Size: 329 MB in CSV format.
   - Balanced classes.

## Data Preprocessing

### Noise Removal

- Eliminated duplicate rows and handled null values to improve data quality and model performance.

### Class Balancing

- Ensured balanced representation of both classes by identifying and addressing class imbalances.

### Stop Words Removal

- Removed superfluous terms using NLTK's English-based dictionary to optimize data analysis.

### Feature Extraction

- Applied TF-IDF to convert text into numerical features for model training.
- Used `train_test_split` to divide the dataset into training (80%) and testing (20%) sets, ensuring class distribution.

## Model Training

### Logistic Regression

- Transformed continuous value outputs into categorical values using a sigmoid function.
- Created a pipeline including TF-IDF vectorization and logistic regression for text classification.
- Visualized the pipeline to illustrate the training process.

### Random Forest

- Combined predictions of multiple decision trees to improve classification accuracy.
- Used TfidfVectorizer for text vectorization, focusing on the top 1000 features based on TF-IDF scores.

### BERT (Bidirectional Encoder Representations from Transformers)

- Utilized BERT's ability to grasp contextual nuances in language.
- Preprocessed text data using a pre-trained BERT tokenizer from TensorFlow Hub.
- Trained a BERT model with dense layers and dropout for regularization.
- Compiled the model with the Adam optimizer and binary cross-entropy loss.

## Evaluation

- Measured performance using accuracy, precision, recall, F1-score, and confusion matrix.
- Achieved 99% accuracy with both logistic regression and random forest models.
- BERT model demonstrated high accuracy in distinguishing LLM-generated content.

## Results

- Comprehensive testing showed perfect performance metrics for both classes (0 and 1), including precision, recall, and F1-score, all at 1.00.
- Logistic regression and random forest models achieved 99% accuracy on the evaluation dataset.

---
