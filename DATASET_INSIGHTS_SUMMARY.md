# MahaSent Dataset - Comprehensive Analysis & Insights

## 📊 Overview
This is a **Marathi Sentiment Analysis Dataset** containing social media comments/text classified into three sentiment categories. The dataset appears to be focused on political and social discourse in Maharashtra, India.

---

## 📁 Dataset Structure

### Files & Split Distribution
| Dataset | Samples | Percentage |
|---------|---------|------------|
| **Train** | 48,114 | 79.1% |
| **Test** | 6,750 | 11.1% |
| **Val** | 6,000 | 9.8% |
| **Total** | 60,864 | 100% |

The dataset follows a standard **79-11-10 train-test-validation split**, which is ideal for machine learning model training.

---

## 🏷️ Label Distribution

### Sentiment Classes
The dataset has **perfectly balanced classes** across all splits:

- **Label -1**: Negative Sentiment (33.33%)
- **Label 0**: Neutral Sentiment (33.33%)
- **Label 1**: Positive Sentiment (33.33%)

### Distribution by Split
| Split | Negative (-1) | Neutral (0) | Positive (1) |
|-------|--------------|-------------|--------------|
| Train | 16,038 | 16,038 | 16,038 |
| Test | 2,250 | 2,250 | 2,250 |
| Val | 2,000 | 2,000 | 2,000 |

**✅ Key Insight**: Perfect class balance means no need for class weighting or resampling techniques during model training.

---

## 📏 Text Characteristics

### Length Statistics (in characters)

| Dataset | Mean | Median | Min | Max | Std Dev |
|---------|------|--------|-----|-----|---------|
| Train | 90.4 | 64 | 1 | 467 | 74.8 |
| Test | 100.8 | 70 | 3 | 449 | 81.4 |
| Val | 89.4 | 64 | 1 | 349 | 73.2 |

**📌 Observations**:
- **Short to medium-length texts**: Most texts are social media comments (average ~90 characters)
- **High variance**: Standard deviation of ~75 suggests diverse text lengths
- **Right-skewed distribution**: Mean > Median indicates some longer outlier texts
- **Test set slightly longer**: Test data has higher average length (100.8 vs 90.4)

---

## 🔍 Content Analysis

### Language
- **Primary Language**: Marathi (मराठी)
- **Script**: Devanagari
- **Domain**: Political and social commentary, appears to be from social media platforms

### Sample Texts by Sentiment

#### Negative Sentiment (Label -1)
Examples suggest criticism, complaints, and negative political commentary:
- *"होता होता राहीलेला निवडणूक मारो मर्ज़ीभई"* (About elections)
- *"खरा लखोबा तर हा बोबडाच आहे"* (Criticism)
- References to political issues, corruption, governance problems

#### Neutral Sentiment (Label 0)
Factual statements, questions, or balanced observations:
- *"तारक मेहता'तील 'बबीताजी' कितवी शिकल्या माहितेय का?"* (Question about TV show)
- *"खरंय पण जनतेतला हा राग मतांमध्ये करायला पाहिजे"* (Neutral observation)

#### Positive Sentiment (Label 1)
Praise, appreciation, and positive expressions:
- *"अतिशय उत्कृष्ट चित्रपट आहे हा"* (Movie praise)
- *"म्हणून आता करताव होय"* (Affirmation)
- *"तुमच्या बाबाचा तुमच्यावर विश्वास तरी आहे"* (Trust/appreciation)

---

## 🎯 Use Cases & Applications

This dataset is suitable for:

1. **Sentiment Analysis Models**
   - Training Marathi language sentiment classifiers
   - Multi-class classification (3 classes)
   - Social media sentiment analysis

2. **Natural Language Processing Research**
   - Low-resource language NLP (Marathi)
   - Regional language sentiment analysis
   - Political discourse analysis

3. **Model Types**
   - Traditional ML: Naive Bayes, SVM, Logistic Regression
   - Deep Learning: LSTM, GRU, Transformers
   - Pre-trained models: mBERT, XLM-RoBERTa, IndicBERT

---

## ✅ Data Quality Assessment

### Strengths
- ✅ **No missing values**: Complete dataset with no NaNs
- ✅ **Perfect class balance**: Equal representation of all sentiment classes
- ✅ **Good size**: 60K+ samples is substantial for NLP tasks
- ✅ **Standard splits**: Pre-split into train/test/val
- ✅ **Real-world data**: Appears to be authentic social media content

### Potential Considerations
- ⚠️ **Very short texts**: Some samples as short as 1-3 characters
- ⚠️ **Domain-specific**: Heavy focus on politics/social issues
- ⚠️ **Colloquial language**: Social media style with slang/informal speech
- ⚠️ **Code-mixing**: May contain English words mixed with Marathi

---

## 🔧 Recommended Preprocessing Steps

1. **Text Cleaning**
   - Remove special characters and URLs
   - Handle emojis appropriately
   - Normalize whitespace

2. **Tokenization**
   - Use Marathi-specific tokenizers
   - Consider subword tokenization (BPE/WordPiece)

3. **Handling Short Texts**
   - Filter or flag extremely short texts (< 5 chars)
   - Consider context augmentation

4. **Encoding**
   - Use multilingual models (mBERT, XLM-R)
   - Or train custom Marathi embeddings

---

## 📈 Model Training Recommendations

### Baseline Approaches
- **TF-IDF + Logistic Regression**: Quick baseline
- **Word2Vec/FastText + Traditional ML**: Good starting point

### Advanced Approaches
- **Multilingual BERT (mBERT)**: Fine-tune on this dataset
- **XLM-RoBERTa**: Strong multilingual performance
- **IndicBERT**: Specifically designed for Indian languages
- **MuRIL**: Google's Multilingual Representations for Indian Languages

### Evaluation Metrics
- Accuracy (balanced dataset, so valid metric)
- F1-Score (macro/micro)
- Confusion Matrix analysis
- Per-class precision and recall

---

## 🎓 Project Context

This appears to be a **Marathi Sentiment Analysis (Maha SA)** project focusing on:
- Political and social discourse in Maharashtra
- Regional language NLP
- Social media sentiment monitoring
- Possibly election-related or governance sentiment tracking

---

## 📊 Quick Statistics Summary

```
Total Samples:     60,864
Languages:         Marathi (Devanagari script)
Classes:           3 (Negative, Neutral, Positive)
Class Balance:     Perfect (33.33% each)
Avg Text Length:   ~90 characters
Domain:            Political/Social Media
Missing Values:    0
Data Quality:      High
```

---

## 🚀 Next Steps for Your Project

1. **Exploratory Data Analysis (EDA)**
   - Word frequency analysis
   - Common n-grams per sentiment
   - Topic modeling

2. **Model Selection & Training**
   - Start with baseline models
   - Progress to transformer-based models
   - Compare performance

3. **Evaluation & Validation**
   - Cross-validation
   - Error analysis
   - Confusion matrix interpretation

4. **Deployment Considerations**
   - Model size vs accuracy tradeoff
   - Inference speed requirements
   - Real-time vs batch processing

---

**Generated on**: November 27, 2025
**Dataset**: MahaSent (Marathi Sentiment Analysis)
