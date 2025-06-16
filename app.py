import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import os

@st.cache_data
def load_data():
    df = pd.read_csv('D:/PyThon/UDAI/iris/iris.csv')
    return df

@st.cache_resource
def train_model(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    id3_classifier = DecisionTreeClassifier(criterion='entropy', random_state=42)
    id3_classifier.fit(X_train_scaled, y_train)
    return scaler, id3_classifier

@st.cache_data
def create_pairplot(_df):
    fig = sns.pairplot(_df, hue='species', diag_kind='kde')
    return fig

@st.cache_data
def create_heatmap(_df):
    correlation_matrix = _df.select_dtypes(include=np.number).corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", mask=mask, ax=ax)
    ax.set_title('Correlation Matrix Heatmap')
    return fig

def create_comparison_table(y_test, y_pred):
    comparison_df = pd.DataFrame({
        'Wine ID': range(1, len(y_test) + 1),
        'Actual Species': y_test,
        'Predicted Species': y_pred,
        'Match': [
            "Match" if actual == predicted else "Mismatch"
            for actual, predicted in zip(y_test, y_pred)
        ]
    })
    return comparison_df

df = load_data()
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

X = df.drop('species', axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

scaler, id3_classifier = train_model(X_train, y_train)

st.title("Iris Species Prediction App")
st.write("This app predicts the Iris species based on sepal and petal measurements using a Decision Tree Classifier.")

st.sidebar.header("Input Features")
def create_sliders():
    inputs = {}
    for feature in features:
        min_val = float(df[feature].min())
        max_val = float(df[feature].max())
        mean_val = float(df[feature].mean())
        inputs[feature] = st.sidebar.slider(
            label=feature.replace('_', ' ').title(),
            min_value=min_val,
            max_value=max_val,
            value=round(mean_val, 1),
            step=0.1
        )
    return inputs

user_inputs = create_sliders()
input_df = pd.DataFrame([user_inputs], columns=features)

input_scaled = scaler.transform(input_df)
prediction = id3_classifier.predict(input_scaled)[0]
probabilities = id3_classifier.predict_proba(input_scaled)[0]
class_names = id3_classifier.classes_

species_images = {
    'setosa': 'image/setosa.jpg',
    'versicolor': 'image/versicolor.jpg',
    'virginica': 'image/virginica.jpg'
}

st.subheader("Prediction")
st.write(f"Predicted Species: *{prediction}*")

image_path = species_images.get(prediction)
if image_path:
    st.image(image_path, caption=f'{prediction} Flower', use_column_width=True)

st.write("Prediction Probabilities:")
prob_df = pd.DataFrame({
    'Species': class_names,
    'Confidence': [f"{prob:.2f}" for prob in probabilities]
})
st.table(prob_df)

st.subheader("Model Performance")
y_val_pred = id3_classifier.predict(scaler.transform(X_val))
y_test_pred = id3_classifier.predict(scaler.transform(X_test))
val_accuracy = id3_classifier.score(scaler.transform(X_val), y_val)
test_accuracy = id3_classifier.score(scaler.transform(X_test), y_test)
st.write(f"Validation Accuracy: {val_accuracy:.2f}")
st.write(f"Test Accuracy: {test_accuracy:.2f}")
st.write("Classification Report (Test Set):")
st.text(classification_report(y_test, y_test_pred))

st.subheader("Detailed Comparison: Actual vs Predicted")
comparison_df = create_comparison_table(y_test, y_test_pred)
st.write(comparison_df.style.applymap(
    lambda x: 'background-color: #5994bc' if x == 'Exact Match' else 'background-color: lightgreen' if x == 'Match' else 'background-color: lightcoral', subset=['Match']
).set_table_styles(
    [{'selector': 'th', 'props': [('font-size', '110%'), ('text-align', 'center')]}]
))

st.subheader("Data Visualizations")
st.write("Pairplot of Features by Species")
pairplot_fig = create_pairplot(df)
st.pyplot(pairplot_fig)

st.write("Correlation Matrix Heatmap")
heatmap_fig = create_heatmap(df)
st.pyplot(heatmap_fig)

st.write("Box Plots of Features by Species")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
for i, feature in enumerate(features):
    sns.boxplot(data=df, x='species', y=feature, ax=axes[i], palette='Set2')
    axes[i].set_title(f'Distribution of {feature} by Species')
    axes[i].set_xlabel('Species')
    axes[i].set_ylabel(feature)
plt.tight_layout()
plt.suptitle('Box Plot of Features by Species', y=1.02, fontsize=16)
st.pyplot(fig)

st.write("Feature Importance in ID3 Classifier")
importances = id3_classifier.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

fig1, ax1 = plt.subplots(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis', ax=ax1)
ax1.set_title('Feature Importance in ID3 Classifier')
ax1.set_xlabel('Importance')
ax1.set_ylabel('Feature')
st.pyplot(fig1)

st.write("Decision Tree Visualization (ID3 Classifier)")
from sklearn.tree import plot_tree

fig2, ax2 = plt.subplots(figsize=(20, 10))
plot_tree(id3_classifier, feature_names=feature_names, class_names=y.unique(), filled=True, ax=ax2)
st.pyplot(fig2)
