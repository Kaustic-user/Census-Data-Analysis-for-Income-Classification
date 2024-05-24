import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                'hours-per-week', 'native-country','target']

df = pd.read_csv('adult.data', header=None,names=column_names)
df.to_csv('output_file.csv', index=False,)

"""# Handling NULL and Duplicate Values"""

# Replace '?' with NaN values
df.replace(' ?', np.nan, inplace=True)

#calculating % nan values
df.isnull().sum()*100/df.shape[0]

print(df.shape)

#dropping null values as less than 6%
df.dropna(inplace=True)
print(df.shape)

#dropping duplicates
df = df.drop_duplicates()
print(df.shape)

"""# Encoding of Categorial Features"""

#ordinal encoding bcoz education attributes are in order
print(df['education'].unique())
education_order = [
    ' Preschool', ' 1st-4th', ' 5th-6th', ' 7th-8th', ' 9th', ' 10th', ' 11th',
    ' 12th', ' HS-grad', ' Some-college', ' Assoc-acdm', ' Assoc-voc',
    ' Bachelors', ' Masters', ' Prof-school', ' Doctorate'
]

education_mapping = {level: index for index, level in enumerate(education_order)}
print(education_mapping)

df['education'] = df['education'].map(education_mapping)

#binarization
print(df['sex'].unique())
sex_mapping = {' Female': 0, ' Male': 1}
print(sex_mapping)
df['sex'] = df['sex'].map(sex_mapping)

#label encoding (0,1)
print(df['target'].unique())
target_mapping = {' <=50K': 0, ' >50K': 1}
print(target_mapping)

df['target'] = df['target'].map(target_mapping)

#one hot-encoding bcoz all are independent, and have equal weightage
print(df['race'].unique())
# Perform one-hot encoding for the 'race' column
df = pd.get_dummies(df, columns=['race'], drop_first=False)

#calculating the frequency
frequency = df['native-country'].value_counts()
print(frequency)

#count encoding
print(df['native-country'].unique())
count_encoding = df['native-country'].value_counts()

# Map each value to its corresponding count
df['native-country'] = df['native-country'].map(count_encoding)

#ordinal encoding
print(df['marital-status'].unique())
marital_status_order = [
    ' Never-married', ' Separated', ' Divorced', ' Widowed',
    ' Married-spouse-absent', ' Married-civ-spouse', ' Married-AF-spouse'
]

# Create a dictionary mapping each marital status to its ordinal value
ordinal_encoding = {status: index for index, status in enumerate(marital_status_order)}

# Perform ordinal encoding on the 'marital-status' column
df['marital-status'] = df['marital-status'].map(ordinal_encoding)

#count encoding
print(df['occupation'].unique())
count_encoding = df['occupation'].value_counts()

# Map each occupation to its corresponding count
df['occupation'] = df['occupation'].map(count_encoding)

#one hot encoding
relationship_status_order = [
    ' Not-in-family', ' Husband', ' Not-in-family', ' Husband', ' Wife'
]
unique_relationship = df['relationship'].unique()

# Perform one-hot encoding for the 'race' column
df = pd.get_dummies(df, columns=['relationship'], drop_first=False)

#calculate frequency
frequency = df['workclass'].value_counts()
print(frequency)

#count encoding
print(df['workclass'].unique())
count_encoding = df['workclass'].value_counts()

# Map each value to its corresponding count
df['workclass'] = df['workclass'].map(count_encoding)

#Shift the target column to the end
# Pop the 'target' column
target = df.pop('target')

# Insert the 'target' column at the end
df['target'] = target

# Print the names of all columns
print("Column names:")
print(df.columns.tolist())

frequency = df['target'].value_counts()
print(frequency)

"""# Normalization"""

# Min-max Normalization bcoz we have done one-hot-encoding for the attributes so to maintain 0-1.
from sklearn.preprocessing import MinMaxScaler
print(df)
X = df.drop(columns=['target'])

# Extract target variable (y)
y = df['target']

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the data
X = scaler.fit_transform(X)

print("Normalized Data:")
print(X)

print(df.shape)

import seaborn as sns
import matplotlib.pyplot as plt
# Calculate the correlation matrix of the original features
original_corr = df.drop('target', axis=1).corr()

# Create a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(original_corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap of Original Features')
plt.show()

plt.savefig('heatmap.png')

"""# Principal Component Analysis"""

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Perform PCA
pca = PCA()
pca.fit(X)

# Calculate the percentage of variance retained for each principal component
percentage_variance_retained = pca.explained_variance_ratio_ * 100

# Scree plot (bar plot)
plt.figure(figsize=(10, 6))
plt.bar(range(1, 24), percentage_variance_retained, align='center')
plt.xlabel('Principal Component')
plt.ylabel('Percentage of Variance Retained')
plt.title('Scree Plot')
plt.show()

plt.savefig('screeplot.png')

# Print percentage of variance retained per principal component
print('Percentage of variance retained per principal component:')
for i, variance_retained in enumerate(percentage_variance_retained):
    print(f'PC{i+1}: {variance_retained:.2f}%')

desired_variance_percentage=90
# Calculate cumulative explained variance ratios
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_) * 100

# Find the number of principal components needed to explain the desired percentage of variance
num_components_needed = np.argmax(cumulative_variance_ratio >= desired_variance_percentage) + 1

print(f'Number of principal components needed to explain {desired_variance_percentage}% of variance:', num_components_needed)

X = pca.transform(X)[:, :num_components_needed]

print("Shape of X:", X.shape)

# Concatenate principal components with the target variable
df = pd.DataFrame(np.concatenate((X, y.values.reshape(-1, 1)), axis=1),
                  columns=[f'PC{i}' for i in range(X.shape[1])]+['target'])

"""# Train-Test Split"""

from sklearn.model_selection import train_test_split

X = df.drop('target', axis=1)
y = df['target']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
df

import seaborn as sns

# Calculate the correlation matrix of the original features
original_corr = df.drop('target', axis=1).corr()

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(original_corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap of Original Features')
plt.show()
plt.savefig('heatmap2.png')

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve

import seaborn as sns

# Pair plot of the original features with different colors for different classes
sns.pairplot(df, hue='target')
plt.suptitle('Pair Plot of Original Features with Different Colors for Different Classes', y=1.02)
plt.show()
plt.savefig('pairplot.png')

"""# Suppot Vector Machines (SVM)"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 10],                        #represents the regularization parameter in Support Vector Machines (SVMs). It controls the trade-off between achieving a low training error and minimizing the norm of the weights.
    'gamma': [0.001, 0.01, 0.1, 1],           #The gamma parameter can be seen as the inverse of the radius of influence of samples selected by the model as support vectors.
    'kernel': ['rbf', 'poly', 'sigmoid']
}

# Create the SVM classifier
svm = SVC(probability=True)

# Perform GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(svm, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Print the best hyperparameters
print("Best hyperparameters:", grid_search.best_params_)

# Get the best SVM model
best_svm = grid_search.best_estimator_

import matplotlib.pyplot as plt
# Get training accuracy
train_accuracy = best_svm.score(X_train_scaled, y_train)
print("Training Accuracy:", train_accuracy)

# Evaluate the model on the test set
test_accuracy = best_svm.score(X_test_scaled, y_test)
print("Test Accuracy:", test_accuracy)

from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np

train_sizes, train_scores, test_scores = learning_curve(best_svm, X_train_scaled, y_train, cv=5)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, color='blue', marker='o', label='Training Accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', label='Validation Accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.title('Learning Curve')
plt.show()
plt.savefig('lcsvm.png')

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
y_pred = best_svm.predict(X_test_scaled)

# Compute precision-recall pairs
precision, recall, _ = precision_recall_curve(y_test, y_pred)

# Plot precision-recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label='Precision-Recall Curve', color='blue')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('prsvm.png')

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Compute false positive rate and true positive rate
fpr, tpr, _ = roc_curve(y_test, y_pred)

# Compute ROC AUC
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
plt.savefig('rocsvm.png')

from sklearn.metrics import classification_report


print("Classification Report:")
print(classification_report(y_test, y_pred))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
plt.savefig('cmsvm.png')

"""# Artificial Neural Networks"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the ANN classifier
ann_clf = MLPClassifier(max_iter=1000)  # You can customize other parameters here

# Define the hyperparameters to tune
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'activation': ['logistic', 'relu'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01],
}
# Perform grid search to find the best combination of hyperparameters
grid_search = GridSearchCV(estimator=ann_clf, param_grid=param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_

# Create the best MLPClassifier model using the best hyperparameters
best_ann_clf = MLPClassifier(max_iter=1000, **best_params)

# Train the model with the best hyperparameters
best_ann_clf = MLPClassifier(max_iter=1000, **best_params)  # Set the maximum number of iterations
# best_ann_clf.fit(X_train_scaled, y_train)

# Fit the best ANN classifier
best_ann_clf.fit(X_train_scaled, y_train)

# Get training accuracy
train_accuracy = best_ann_clf.score(X_train_scaled, y_train)
print("Training Accuracy:", train_accuracy)

# Evaluate the model on the test set
test_accuracy = best_ann_clf.score(X_test_scaled, y_test)
print("Test Accuracy:", test_accuracy)

from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np

train_sizes, train_scores, test_scores = learning_curve(best_ann_clf, X_train_scaled, y_train, cv=5)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, color='blue', marker='o', label='Training Accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', label='Validation Accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.title('Learning Curve')
plt.show()
plt.savefig('lcann.png')

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

# Compute precision-recall pairs
precision, recall, _ = precision_recall_curve(y_test, best_ann_clf.predict_proba(X_test_scaled)[:, 1])

# Plot precision-recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label='Precision-Recall Curve', color='blue')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('prann.png')

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Compute false positive rate and true positive rate
fpr, tpr, _ = roc_curve(y_test, best_ann_clf.predict_proba(X_test_scaled)[:, 1])

# Compute ROC AUC
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
plt.savefig('rocann.png')

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

y_pred = best_ann_clf.predict(X_test_scaled)
# Compute confusion matrix
# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
plt.savefig('cmann.png')
# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

"""# Gradient Boosted Decision Tree (GBDT)"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the gradient boosted classifier
gb_clf = GradientBoostingClassifier()

# Define the hyperparameters to tune
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.5],
    'max_depth': [3, 4, 5]
}

# Perform grid search to find the best combination of hyperparameters
grid_search = GridSearchCV(estimator=gb_clf, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_

# Train the model with the best hyperparameters
best_gb_clf = GradientBoostingClassifier(**best_params)
# best_gb_clf.fit(X_train_scaled, y_train)

best_gb_clf.fit(X_train_scaled, y_train)

# Get training accuracy
train_accuracy = accuracy_score(y_train, best_gb_clf.predict(X_train_scaled))
print("Training Accuracy:", train_accuracy)

# Evaluate the model on the test set
test_accuracy = accuracy_score(y_test, best_gb_clf.predict(X_test_scaled))
print("Test Accuracy:", test_accuracy)

import numpy as np
import matplotlib.pyplot as plt

train_errors = []
test_errors = []

estimators = [50, 100, 150]  # Assuming you want to plot for these values
for estimator in estimators:
    gb_clf = GradientBoostingClassifier(n_estimators=estimator, **{key: value for key, value in best_params.items() if key != 'n_estimators'})
    gb_clf.fit(X_train_scaled, y_train)
    train_errors.append(1 - gb_clf.score(X_train_scaled, y_train))
    test_errors.append(1 - gb_clf.score(X_test_scaled, y_test))

plt.figure(figsize=(10, 6))
plt.plot(estimators, train_errors, label='Training Error')
plt.plot(estimators, test_errors, label='Testing Error')
plt.xlabel('Number of Estimators')
plt.ylabel('Error')
plt.title('Learning Curve')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('lcdt.png')

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

# Compute precision-recall pairs
precision, recall, _ = precision_recall_curve(y_test, best_gb_clf.predict_proba(X_test_scaled)[:, 1])

# Plot precision-recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label='Precision-Recall Curve', color='blue')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('prgbdt.png')

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Compute false positive rate and true positive rate
fpr, tpr, _ = roc_curve(y_test, best_gb_clf.predict_proba(X_test_scaled)[:, 1])

# Compute ROC AUC
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
plt.savefig('rocgbdt.png')

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt


# Assuming best_gb_clf is your trained GradientBoostingClassifier
# Make sure you have already defined and trained it
y_pred = best_gb_clf.predict(X_test_scaled)
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
plt.savefig('cmgbdt.png')
# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

"""# Naive Bayes

var_smoothing is a smoothing parameter added to the variances for calculation stability. Smoothing is necessary because if a feature in the test set has a value not present in the training set, then the model assigns it zero probability and, therefore, cannot make predictions. To prevent this, a small value is added to the variance to smooth the calculation.

This parameter determines the number of folds in cross-validation. In this case, cv=3 means that the data will be split into 3 folds

This parameter controls the verbosity of the output during the fitting process. When verbose=1

Setting verbose=0 will suppress these progress messages, while setting verbose>1 may increase the level of detail in the output.
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the Naive Bayes classifier
nb_clf = GaussianNB()

# Define the hyperparameters to tune (Note: Naive Bayes doesn't have many hyperparameters to tune)
param_grid = {
    'var_smoothing': [1e-9, 1e-8, 1e-7]  # Smoothing parameter
}


# Perform grid search to find the best combination of hyperparameters
grid_search = GridSearchCV(estimator=nb_clf, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_

# Train the model with the best hyperparameters
best_nb_clf = GaussianNB(**best_params)
best_nb_clf.fit(X_train_scaled, y_train)

# Make predictions
y_pred = best_nb_clf.predict(X_test_scaled)

# Get training accuracy
train_accuracy = best_nb_clf.score(X_train_scaled, y_train)
print("Training Accuracy:", train_accuracy)

# Evaluate the model on the test set
test_accuracy = best_nb_clf.score(X_test_scaled, y_test)
print("Test Accuracy:", test_accuracy)

# Define the training set sizes
import matplotlib.pyplot as plt
train_sizes = np.linspace(0.1, 1.0, 10)  # 10 different training set sizes from 10% to 100%

# Compute learning curve
train_sizes, train_scores, test_scores = learning_curve(
    nb_clf, X, y, train_sizes=train_sizes, cv=5)

# Compute mean and standard deviation of training and test scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Training Error', color='blue')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, label='Validation Error', color='orange')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15, color='orange')
plt.xlabel('Training Set Size')
plt.ylabel('Error')
plt.title('Learning Curve for Naive Bayes')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('lcnb.png')

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve


# Compute precision-recall pairs
precision, recall, _ = precision_recall_curve(y_test, y_pred)

# Plot precision-recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label='Precision-Recall Curve', color='blue')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Naive Bayes')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('prnb.png')

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc



# Compute false positive rate and true positive rate
fpr, tpr, _ = roc_curve(y_test, y_pred)

# Compute ROC AUC
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Naive Bayes')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
plt.savefig('rocnb.png')

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.tight_layout()

# Save the plot as an image file
plt.savefig('confusion_matrix.png')

# Show the plot
plt.show()
plt.savefig('cmnb.png')
# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

"""# Multimodel"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# Get the best estimators from each grid search
# best_svm_clf = svm_grid_search.best_estimator_
# best_gbdt_clf = gbdt_grid_search.best_estimator_
# best_nb_clf = nb_grid_search.best_estimator_
# best_ann_clf = ann_grid_search.best_estimator_

# Fit the best classifiers on the full training data
best_svm.fit(X_train_scaled, y_train)
best_gb_clf.fit(X_train_scaled, y_train)
best_nb_clf.fit(X_train_scaled, y_train)
best_ann_clf.fit(X_train_scaled, y_train)

# Predict probabilities on the test set
svm_proba = best_svm.predict_proba(X_test_scaled)[:, 1]
gbdt_proba = best_gb_clf.predict_proba(X_test_scaled)[:, 1]
nb_proba = best_nb_clf.predict_proba(X_test_scaled)[:, 1]
ann_proba = best_ann_clf.predict_proba(X_test_scaled)[:, 1]

# Concatenate the probabilities
all_proba = np.column_stack((svm_proba, gbdt_proba, nb_proba, ann_proba))

# Define a logistic regression meta-learner
meta_learner = LogisticRegression()

# Fit the meta-learner on the probabilities
meta_learner.fit(all_proba, y_test)

# Make predictions on the probabilities
weighted_proba = meta_learner.predict_proba(all_proba)[:, 1]

# Convert probabilities to binary predictions
weighted_pred = np.where(weighted_proba >= 0.5, 1, 0)

# Evaluate the weighted ensemble model
accuracy = accuracy_score(y_test, weighted_pred)
print("Weighted Ensemble Model Accuracy:",accuracy)

# Define the training set sizes
import matplotlib.pyplot as plt
train_sizes = np.linspace(0.1, 1.0, 10)  # 10 different training set sizes from 10% to 100%

# Initialize empty arrays to store cross-validated accuracy scores
train_scores = []
test_scores = []

# Perform manual cross-validation and calculate accuracy scores for each training set size
for train_size in train_sizes:
    # Determine the number of samples for the current training size
    n_train_samples = int(train_size * X_train.shape[0])
    X_train_subset = X_train[:n_train_samples]
    y_train_subset = y_train[:n_train_samples]

    # Fit the meta-learner on the subset of the training data
    meta_learner.fit(X_train_subset, y_train_subset)

    # Predict probabilities on the subset of the training data
    weighted_proba_subset = meta_learner.predict_proba(X_train_subset)[:, 1]

    # Convert probabilities to binary predictions
    weighted_pred_subset = np.where(weighted_proba_subset >= 0.5, 1, 0)

    # Calculate accuracy scores
    train_accuracy = accuracy_score(y_train_subset, weighted_pred_subset)
    test_accuracy = accuracy_score(y_test, meta_learner.predict(X_test))

    # Append accuracy scores to the lists
    train_scores.append(train_accuracy)
    test_scores.append(test_accuracy)

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores, label='Training Accuracy', color='blue')
plt.plot(train_sizes, test_scores, label='Test Accuracy', color='orange')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title('Learning Curve for Weighted Ensemble Model')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('lcmm.png')

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

# Compute precision and recall
precision, recall, _ = precision_recall_curve(y_test, weighted_proba)

# Plot precision-recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True)
plt.show()
plt.savefig('prmm.png')

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Compute ROC curve
fpr, tpr, _ = roc_curve(y_test, weighted_proba)

# Compute area under the curve (AUC)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
plt.savefig('rocmm.png')

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
# Make predictions
weighted_pred = np.where(weighted_proba >= 0.5, 1, 0)
# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, weighted_pred)

# Plot confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
plt.savefig('cmmm.png')
# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, weighted_pred))

