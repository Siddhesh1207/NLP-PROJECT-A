# Employee Satisfaction Survey Analysis

## Course: NLP (Semester 6) - Pillai College of Engineering

## Project Overview:
This project is part of the Natural Language Processing (NLP) course for Semester 6 students at Pillai College of Engineering. The project focuses on Employee Sentiment Classification, where we apply various Machine Learning (ML), Deep Learning (DL), and Transformer-based Language Models to categorize whether an employees's reviews are postive, negative or neutral. This implementation involves exploring techniques like text preprocessing, feature extraction, model training, and evaluating the models for their effectiveness in classifying employee sentiments.

You can learn more about the college by visiting the official website of [Pillai College of Engineering](https://www.pillai.edu.in/).

## Acknowledgements:
We would like to express our sincere gratitude to the following individuals:

**Theory Faculty:**  
- Dhiraj Amin  
- Sharvari Govilkar  

**Lab Faculty:**  
- Dhiraj Amin  
- Neha Ashok  
- Shubhangi Chavan  

Their guidance and support have been invaluable throughout this project.

---

## Project Title:
Employee Satisfaction Survey Analysis using Natural Language Processing

## Project Abstract:
The Employee Satisfaction Analysis project aims to categorize employee's reviews into different sentiments such as positive, negative and neutral. This task involves applying Machine Learning, Deep Learning, and Language Models to accurately identify employee's sentiments from text input. The project explores different algorithms, including traditional machine learning techniques, deep learning models, and state-of-the-art pre-trained language models. The goal is to evaluate the performance of each approach and select the best-performing model for employee sentiment analysis.

---

## Algorithms Used:

### Machine Learning Algorithms:
- Logistic Regression  
- Support Vector Machine (SVM)  
- Random Forest Classifier  

### Deep Learning Algorithms:
- Convolutional Neural Networks (CNN)  
- Long Short-Term Memory (LSTM)  
- Bidirectional LSTM (BiLSTM)  
- Combined CNN-BiLSTM  

### Language Models:
- BERT (Bidirectional Encoder Representations from Transformers)  
- RoBERTa (Robustly Optimized BERT Pre-training Approach)  

---

## Comparative Analysis:
The comparative analysis of different models highlights their effectiveness in classifying employee sentiments. The following tables summarize the performance metrics:

### Task 1: Machine Learning Models
| No. | Model Name            | Feature      | Accuracy |
|-----|-----------------------|--------------|-----------|
| 0   | Random Forest         | NLP Features | 0.5333    |  
| 1   | SVM                   | NLP Features | 0.5519    | 
| 2   | Logistic Regression   | NLP Features | 0.5407    |  
| 3   | Random Forest         | BoW          | 1.00      |   
| 4   | SVM                   | BoW          | 1.00      |   
| 5   | Logistic Regression   | BoW          | 1.00      |
| 6   | Random Forest         | TFIDF        | 1.00      | 
| 7   | SVM                   | TFIDF        | 1.00      |
| 8   | Logistic Regression   | TFIDF        | 1.00      |
| 9   | Random Forest         | FastText     | 0.9648    |
| 10  | SVM                   | FastText     | 0.8833    |
| 11  | Logistic Regression   | FastText     | 0.9111    |
| 12  | Random Forest         | Combined     | 0.9944    |
| 13  | SVM                   | Combined     | 1.00      |
| 14  | Logistic Regression   | Combined     | 0.9926    |

### Task 2: Deep Learning Models
| No. | Model Name   | Feature  | Precision | Recall | F1 Score | Accuracy |
|-----|-------------|-----------|-----------|--------|----------|----------|
| 1   | CNN         | BoW       | 0.99      | 0.99   | 0.99     | 0.9944   |
| 2   | LSTM        | BoW       | 0.27      | 0.33   | 0.17     | 0.3222   |
| 3   | BiLSTM      | BoW       | 0.56      | 0.51   | 0.50     | 0.5139   |
| 4   | CNN-BiLSTM  | BoW       | 0.56      | 0.51   | 0.50     | 0.5139   |
| 5   | CNN         | TF-IDF    | 0.11      | 0.33   | 0.16     | 0.3222   |
| 6   | LSTM        | TF-IDF    | 0.11      | 0.33   | 0.16     | 0.3222   |
| 7   | BiLSTM      | TF-IDF    | 0.11      | 0.33   | 0.16     | 0.3278   |
| 8   | CNN-BiLSTM  | TF-IDF    | 0.11      | 0.33   | 0.16     | 0.3278   |
| 9   | CNN         | FastText  | 0.69      | 0.68   | 0.67     | 0.6778   |
| 10  | LSTM        | FastText  | 0.11      | 0.33   | 0.16     | 0.3222   |
| 11  | BiLSTM      | FastText  | 0.32      | 0.41   | 0.31     | 0.4083   |
| 12  | CNN-BiLSTM  | FastText  | 0.61      | 0.53   | 0.49     | 0.5222   |

### Task 3: Transformer Models
| No. | Model Name | Precision | Recall | F1 Score | Accuracy | MCC   |
|-----|------------|-----------|--------|----------|----------|-------|
| 1   | BERT       | 1.0       | 1.0    | 1.0      | 1.0      | 0.994 |
| 2   | RoBERTa    | 1.0       | 1.0    | 1.0      | 1.0      | 1.000 |

---

## Conclusion:
This Employee Analysis project demonstrates the potential of Machine Learning, Deep Learning, and Language Models for text classification tasks, particularly for categorizing user intents. The comparative analysis reveals that:  
- **BoW and TF-IDF based SVM** achieve high performance (~97% accuracy) among traditional ML models.  
- **Deep Learning models (CNN with BoW)** perform comparably to ML models, but others (LSTM, BiLSTM) struggle, likely due to insufficient data or tuning.  
- **Transformer models (RoBERTa, BERT)** outperform all others, with RoBERTa performing a bit better than BERT having **100% MCC (Matthews correlation coefficient)**, highlighting it superiority in capturing contextual semantics.  
