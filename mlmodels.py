import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

data = pd.read_csv("data\clean_data.csv")

categorical = data[["Ethnicity", "Gender", "Age", "Industry", "Education", "Cost of Living"]]
numerical = data[["Annual Income", "Annual Expenses"]]


label_encoders = {}
for column in categorical.columns:
    label_encoders[column] = LabelEncoder()
    categorical[column] = label_encoders[column].fit_transform(categorical[column])

scaler = StandardScaler()
numerical_scaled = scaler.fit_transform(numerical)

X = pd.concat([pd.DataFrame(numerical_scaled, columns=numerical.columns)])
y = data["Relationship Status"].values


def plot_confusion_matrix(y_valid, predictions):
    cm = confusion_matrix(y_valid, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_valid), yticklabels=np.unique(y_valid))
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Labels', fontsize=14)
    plt.ylabel('True Labels', fontsize=14)
    plt.show()

def bayesian_model():
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)

    model = GaussianNB()
    model.fit(X_train, y_train)
    predictions = model.predict(X_valid)
    print(model.score(X_train,y_train))
    print(model.score(X_valid,y_valid))
    plot_confusion_matrix(y_valid,predictions)

def knn_model():
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train,y_train)

    predictions = model.predict(X_valid)
    print(model.score(X_train,y_train))
    print(model.score(X_valid,y_valid))
    plot_confusion_matrix(y_valid,predictions)


def main():
    #bayesian_model()
    knn_model()

main()  