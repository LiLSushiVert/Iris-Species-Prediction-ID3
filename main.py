import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import plot_tree

df = pd.read_csv('D:/PyThon/UDAI/iris/iris.csv')

df.info()
df.head()

species_counts = df['species'].value_counts()
print(species_counts)

print(df.shape)

species_features = df.groupby('species').agg(['mean', 'std'])
print(species_features)

for feature in df.columns[:-1]:
    plt.figure(figsize=(8, 6))
    sns.kdeplot(data=df, x=feature, hue='species', fill=True)
    plt.title(f'Distribution of {feature} by Species')
    plt.show()

sns.pairplot(df, hue='species', diag_kind='kde')
plt.show()

features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
for feature in features:
    print(f"\nThống kê {feature} theo loài hoa:")
    print(df.groupby('species')[feature].describe())

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
for i, feature in enumerate(features):
    sns.boxplot(data=df, x='species', y=feature, ax=axes[i], hue='species', palette='Set2')
    axes[i].set_title(f'Distribution of {feature} by Species')
    axes[i].set_xlabel('Species')
    axes[i].set_ylabel(feature)
plt.tight_layout()
plt.suptitle('Box Plot of Features by Species', y=1.02, fontsize=16)
plt.savefig('boxplot_features_by_species.png')
plt.show()

correlation_matrix = df.select_dtypes(include=np.number).corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", mask=mask)
plt.title('Correlation Matrix Heatmap')
plt.show()

# Histograms with KDE
colors = ['red', 'green', 'blue', 'purple']
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
for i, feature in enumerate(features):
    sns.histplot(df[feature], ax=axes[i], color=colors[i], kde=True)
    axes[i].set_title(f'Distribution of {feature}')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Frequency')
plt.tight_layout()
plt.show()

X = df.drop('species', axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

id3_classifier = DecisionTreeClassifier(criterion='entropy', random_state=42)
id3_classifier.fit(X_train, y_train)

y_val_pred = id3_classifier.predict(X_val)
accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {accuracy}")
print("Classification Report on Validation Data:")
print(classification_report(y_val, y_val_pred))

y_test_pred = id3_classifier.predict(X_test)
accuracy_test = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {accuracy_test}")
print("Classification Report on Test Data:")
print(classification_report(y_test, y_test_pred))

results_df = pd.DataFrame({
    'STT': range(1, len(y_test) + 1),
    'Danh sách': y_test.index,
    'Actual': y_test.values,
    'Predicted': y_test_pred
})
print("Results (Test Data):")
print(results_df)


importances = id3_classifier.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('Feature Importance in ID3 Classifier')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.savefig('feature_importance.png')
plt.show()

plt.figure(figsize=(20, 10))
plot_tree(id3_classifier, feature_names=feature_names, class_names=y.unique(), filled=True)
plt.title('Decision Tree Visualization')
plt.savefig('decision_tree.png')
plt.show()

cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=y.unique(), yticklabels=y.unique())
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()