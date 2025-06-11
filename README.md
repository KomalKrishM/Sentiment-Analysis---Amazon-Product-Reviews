# Sentiment Analysis — Amazon Product Reviews

This repository performs sentiment analysis on Amazon product reviews using pre-trained BERT embeddings and an XGBoost classifier. The workflow includes data preprocessing, embedding generation, visualization, classification, and evaluation. All code and implementation were developed step-by-step with the assistance of ChatGPT.

---

## 📂 Dataset

- **Source**: [Kaggle - Amazon Product Reviews](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews)
- **Contents**: The dataset includes text reviews along with sentiment labels (0 for negative, 1 for positive).

---

## 🧩 Project Structure

### 1. **Data Preprocessing**
- **Script**: `seperate_labels.py`
- **Function**: 
  - Parses the dataset.
  - Extracts review texts and their sentiment labels.
  - Stores them in two separate lists.
  - Combines them into a Pandas DataFrame.

---

### 2. **Text Embeddings**
- **Function**: `generate_embeddings`
- **Model**: Pre-trained **BERT** from HuggingFace Transformers.
- **Output**: High-dimensional embeddings are generated for each review and saved in a `.pt` PyTorch file.

---

### 3. **Visualization**
- **Script**: `view_embeddings.py`
- **Techniques**: 
  - t-SNE
  - UMAP
- **Purpose**: To reduce high-dimensional BERT embeddings into 2D space for visual analysis and clustering.

---

### 4. **Classification**
- **Classifier**: XGBoost
- **Input**: BERT-generated embeddings
- **Task**: Binary classification to predict sentiment (positive/negative)

---

### 5. **Evaluation**
- **Metrics**:
  - Accuracy: `0.8911`
  - Precision: `0.8912`
  - Recall: `0.8911`
  - F1 Score: `0.8911`
  - ROC AUC: `0.9574`

```text
               precision    recall  f1-score   support

           0       0.88      0.90      0.89    200000
           1       0.90      0.88      0.89    200000

    accuracy                           0.89    400000
   macro avg       0.89      0.89      0.89    400000
weighted avg       0.89      0.89      0.89    400000
