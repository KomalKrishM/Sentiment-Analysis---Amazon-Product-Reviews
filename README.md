# Sentiment Analysis — Amazon Product Reviews

This repository performs sentiment analysis on Amazon product reviews using pre-trained 110M parameter BERT embeddings and an XGBoost classifier, and also fine-tuned the 66M parameter BERT model with additional classification layer.  The fine-tuned model improves the accuracy by `7%`. The code was executed on M3 MAC 36GB Memory. All code and implementation were developed step-by-step by giving instructions/prompts to the ChatGPT basic version.

---

## Evaluating Model Response without finetuning

- **Metrics**:
  - Accuracy: `89.11`
  - Precision: `0.8912`
  - Recall: `0.8911`
  - F1 Score: `0.8911`
  - ROC AUC: `0.9574`

***original text*** = "Good product at **reasonable price:** Though the output of this power supply is lower than the Apple supply, \
it seems to work fine--and it is very reasonably priced. Too bad that Amazon stopped carrying the supply."

***original_label*** = 1
***classified_label*** = 0

***modified text*** = "Good product at **a great price:** Though the output of this power supply is lower than the Apple supply, \
it seems to work fine--and it is very reasonably priced. Too bad that Amazon stopped carrying the supply."

***classified_label*** = 1

***Observation:***
The model incorrectly classified the original review as negative (0), but correctly classified the modified version (with "a great price") as positive (1).

***Insight:***
This suggests the BERT-based embeddings did not fully capture the positive sentiment of the phrase **reasonable price**, highlighting a limitation in subtle semantic understanding.

***Fine-tuned Model:***
The fine-tuned model classifies the few test samples with `100%` accuracy.

## Evaluating Model Response with finetuning

- **Metrics**:
  - Accuracy: `96.44`
  - Precision: `0.9644`
  - Recall: `0.9644`
  - F1 Score: `0.9644`

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

### 5. **Fine-tuning**

- **Script**: `fine_tuning_BERT.py`, `Test_fine_tuned_BERT.py`

