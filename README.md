# Iris Species Classification with Decision Tree (ID3)

This project builds a simple interactive web application using **Streamlit** to classify **Iris flower species** based on four morphological features. The machine learning model used is the **Decision Tree Classifier** with the **entropy criterion** (ID3 algorithm).

---

##  Dataset Description

- **Dataset**: `iris.csv` with **150 samples**
- Each sample includes 4 features:
  - Sepal length and width
  - Petal length and width
- **Target label**: Flower species (`setosa`, `versicolor`, `virginica`)

---

##  Model & Workflow

- Model: `DecisionTreeClassifier(criterion='entropy')`
- Preprocessing steps:
  - Data split: **64% training**, the rest for validation and testing
  - Feature scaling with **StandardScaler**
- Train and evaluate the model using standard metrics

---

##  User Interface (Streamlit App)

The app is built using **Streamlit** with the following features:

- Input form to enter flower features
- Real-time prediction of flower species
- Display of illustrative images
- Data visualization (scatter plots, histograms, etc.)

---

##  Libraries Used

- `pandas`, `numpy`
- `scikit-learn`
- `matplotlib`, `seaborn`
- `streamlit`

---

##  Project Objectives

- Help students and beginners understand how **Decision Trees** work
- Provide a visual, interactive way to explore classification algorithms

---

## How to Run the App

pip install -r requirements.txt
streamlit run app.py

## Result:
 ![image](https://github.com/user-attachments/assets/8ff7bd84-0c99-4806-a82b-fb60e48ebb8a)

![image](https://github.com/user-attachments/assets/f1e517b0-0134-44d2-83b0-9cfd5eb24a73)

![image](https://github.com/user-attachments/assets/b21d3c40-8194-48a6-a344-85b288557c5a)



