# Census-Data-Analysis-for-Income-Classification

The primary
 objective is to develop an accurate predictive model capable of
 categorizing individuals into two income classes: > 50K and
 ≤50K. This classification task is essential for understanding
 income distribution patterns and their implications for various
 social and economic sectors.

 Through meticulous
 data preprocessing, feature selection, and employing various
 classification algorithms, including Gradient Boosted Decision Trees (GBDT), Support Vector
 Machines (SVM), Naive Bayes, Multi-Layer Perceptron (MLP) and Ensemble Methods, we construct and evaluate
 predictive models. 

# TABLE I: SVM: Classification Report

 |        | Precision | Recall | F1-score | Support |
|--------|-----------|--------|----------|---------|
| Class 0 | 0.85      | 0.93   | 0.89     | 4485    |
| Class 1 | 0.71      | 0.51   | 0.59     | 1543    |
| Macro Avg | 0.78    | 0.72   | 0.74     | 6028    |
| Weighted Avg | 0.81 | 0.82   | 0.81     | 6028    |

Training accuracy: 0.8297, Testing accuracy: 0.8215

# TABLE II: ANN: Classification Report
|             | Precision | Recall | F1-score | Support |
|-------------|-----------|--------|----------|---------|
| Class 0     | 0.86      | 0.91   | 0.89     | 4485    |
| Class 1     | 0.69      | 0.59   | 0.63     | 1543    |
| Macro Avg   | 0.78      | 0.75   | 0.76     | 6028    |
| Weighted Avg| 0.82      | 0.83   | 0.82     | 6028    |

Training accuracy: 0.8336, Testing accuracy: 0.8254

# TABLE III: GBDT: Classification Report
|             | Precision | Recall | F1-score | Support |
|-------------|-----------|--------|----------|---------|
| Class 0     | 0.86      | 0.92   | 0.89     | 4485    |
| Class 1     | 0.72      | 0.56   | 0.63     | 1543    |
| Macro Avg   | 0.79      | 0.74   | 0.76     | 6028    |
| Weighted Avg| 0.82      | 0.83   | 0.82     | 6028    |

Training accuracy: 0.8486, Testing accuracy: 0.8301

# TABLE IV: NB: Classification Report
|             | Precision | Recall | F1-score | Support |
|-------------|-----------|--------|----------|---------|
| Class 0     | 0.89      | 0.77   | 0.82     | 4485    |
| Class 1     | 0.52      | 0.73   | 0.61     | 1543    |
| Macro Avg   | 0.71      | 0.75   | 0.72     | 6028    |
| Weighted Avg| 0.80      | 0.76   | 0.77     | 6028    |

Training accuracy: 0.7535, Testing accuracy: 0.7574

# TABLE V: ENSEMBLE: Classification Report
|             | Precision | Recall | F1-score | Support |
|-------------|-----------|--------|----------|---------|
| Class 0     | 0.87      | 0.91   | 0.89     | 4485    |
| Class 1     | 0.70      | 0.60   | 0.65     | 1543    |
| Macro Avg   | 0.78      | 0.76   | 0.77     | 6028    |
| Weighted Avg| 0.83      | 0.83   | 0.83     | 6028    |

Weighted Ensemble Model Accuracy: 0.8312





