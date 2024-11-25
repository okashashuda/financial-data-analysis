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
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier

# reddit_data = pd.read_csv("data\clean_reddit.csv")
# kaggle_data = pd.read_csv("data\clean_kaggle.csv")
reddit_data = pd.read_csv("data\clean_reddit.csv")
kaggle_data = pd.read_csv("data\clean_kaggle.csv")


# Plots a confusion matrix to compare correctly predicted values.
def plot_confusion_matrix(y_valid, predictions):
    cm = confusion_matrix(y_valid, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_valid), yticklabels=np.unique(y_valid))
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Labels', fontsize=14)
    plt.ylabel('True Labels', fontsize=14)
    plt.show()

# Bayesian Model
def bayesian_model(X,y):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)

    model = GaussianNB()
    model.fit(X_train, y_train)
    predictions = model.predict(X_valid)
    print(model.score(X_train, y_train))
    print(model.score(X_valid, y_valid))
    plot_confusion_matrix(y_valid, predictions)

#KNN Model
def knn_model(X,y):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train,y_train)

    predictions = model.predict(X_valid)
    print(model.score(X_train, y_train))
    print(model.score(X_valid, y_valid))
    plot_confusion_matrix(y_valid, predictions)

# Random forest model
def randomforest_model(X,y):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train,y_train)

    predictions = model.predict(X_valid)
    print(model.score(X_train, y_train))
    print(model.score(X_valid, y_valid))
    plot_confusion_matrix(y_valid, predictions)

#SVM Model
def svm_model(X,y):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)

    model = SVC(kernel="linear", C=2.0)
    model.fit(X_train, y_train)

    predictions = model.predict(X_valid)
    print(model.score(X_train, y_train))
    print(model.score(X_valid, y_valid))
    plot_confusion_matrix(y_valid, predictions)

# Neural Network Model
def neural_network_model(X,y):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)

    model = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500)
    model.fit(X_train, y_train)

    predictions = model.predict(X_valid)
    print(model.score(X_train, y_train))
    print(model.score(X_valid, y_valid))
    plot_confusion_matrix(y_valid, predictions)

# PCA transformed data model
def pca_model(X,y):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)
    
    pca = PCA(n_components=1)
    X_train_pca = pca.fit_transform(X_train)
    X_valid_pca = pca.transform(X_valid)
    
    # Train a classification model (e.g., GaussianNB) on PCA-transformed data
    model = RandomForestClassifier(n_estimators=150)
    model.fit(X_train_pca, y_train)

    predictions = model.predict(X_valid_pca)

    print(model.score(X_train_pca, y_train))
    print(model.score(X_valid_pca, y_valid))
    plot_confusion_matrix(y_valid, predictions)    
    
def main():
    #Reddit Data organization and encoding the categorical data
    reddit_categorical = reddit_data[["Age", "Industry", "Relationship Status"]]
    reddit_numerical = reddit_data[["Annual Income", "Annual Expenses"]]


    reddit_label_encoders = {}
    for column in reddit_categorical.columns:
        reddit_label_encoders[column] = LabelEncoder()
        reddit_categorical[column] = reddit_label_encoders[column].fit_transform(reddit_categorical[column])

    scaler = StandardScaler()
    reddit_numerical_scaled = scaler.fit_transform(reddit_numerical)

    Xred = pd.concat([pd.DataFrame(reddit_numerical_scaled, columns=reddit_numerical.columns)])
    yred = reddit_data["Education"].values

    kaggle_categorical = kaggle_data[["Relationship Status","Education"]]
    kaggle_numerical = kaggle_data[["Age", "Annual Income"]]

    kaggle_label_encoders = {}
    for column in kaggle_categorical.columns:
        kaggle_label_encoders[column] = LabelEncoder()
        kaggle_categorical[column] = kaggle_label_encoders[column].fit_transform(kaggle_categorical[column])
    
    kaggle_numerical_scaled = scaler.fit_transform(kaggle_numerical)
    Xkag = pd.concat([pd.DataFrame(kaggle_numerical_scaled, columns=kaggle_numerical.columns)])
    ykag = kaggle_data["Industry"].values

    #Running models on reddit data
    # bayesian_model(Xred,yred)
    # knn_model(Xred,yred)
    # randomforest_model(Xred,yred)
    # svm_model(Xred,yred)
    # neural_network_model(Xred,yred)
    # pca_model(Xred,yred)

    #Running models on kaggle data
    bayesian_model(Xkag,ykag)
    knn_model(Xkag,ykag)
    randomforest_model(Xkag,ykag)
    svm_model(Xkag,ykag)
    neural_network_model(Xkag,ykag)
    pca_model(Xkag,ykag)

main()  