import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from scipy import stats
import statsmodels.api as sm

# OKASHA
# reddit_data = pd.read_csv("cmpt353dataproject/data/clean_reddit.csv")

# STEFAN
reddit_data = pd.read_csv("data\clean_reddit.csv")



# # Plots a confusion matrix to compare correctly predicted values.
# def plot_confusion_matrix(y_valid, predictions):
#     cm = confusion_matrix(y_valid, predictions)
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_valid), yticklabels=np.unique(y_valid))
#     plt.title('Confusion Matrix', fontsize=16)
#     plt.xlabel('Predicted Labels', fontsize=14)
#     plt.ylabel('True Labels', fontsize=14)
#     plt.show()

def plot_predicted_vs_original(X_valid,y_valid,predictions):
    # Plot Annual Expenses vs. Annual Income with regression line
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(X_valid['Annual Expenses'], y_valid, color='blue', label='Original Data')
    plt.scatter(X_valid['Annual Expenses'], predictions, color='green', label='Predicted Data', alpha=0.5)
    plt.xlabel('Annual Expenses')
    plt.ylabel('Annual Income')
    plt.title('Annual Expenses vs. Annual Income')
    plt.legend()

    # Plot Total Asset Value vs. Annual Income with regression line
    plt.subplot(1, 2, 2)
    plt.scatter(X_valid['Total Asset Value'], y_valid, color='blue', label='Original Data')
    plt.scatter(X_valid['Total Asset Value'], predictions, color='green', label='Predicted Data', alpha=0.5)
    plt.xlabel('Total Asset Value')
    plt.ylabel('Annual Income')
    plt.title('Total Asset Value vs. Annual Income')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Bayesian Model
def bayesian_model(X,y):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)

    model = GaussianNB()
    model.fit(X_train, y_train)
    predictions = model.predict(X_valid)
    print(model.score(X_train, y_train))
    print(model.score(X_valid, y_valid))

    plot_predicted_vs_original(X_valid,y_valid,predictions)

#KNN Model
def knn_model(X,y):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)

    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(X_train,y_train)

    predictions = model.predict(X_valid)
    print(model.score(X_train, y_train))
    print(model.score(X_valid, y_valid))

    plot_predicted_vs_original(X_valid,y_valid,predictions)

# Random forest model
def randomforest_model(X,y):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train,y_train)

    predictions = model.predict(X_valid)
    print(model.score(X_train, y_train))
    print(model.score(X_valid, y_valid))

    plot_predicted_vs_original(X_valid,y_valid,predictions)

    
def linear_reg_model_ML(X,y):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)

    model = LinearRegression(fit_intercept=True)
    model.fit(X_train,y_train)

    predictions = model.predict(X_valid)
    print("Training Score: ",model.score(X_train, y_train))
    print("Validation Score: ",model.score(X_valid, y_valid))

    plot_predicted_vs_original(X_valid,y_valid,predictions)

def linear_reg_stats(X,y):
    X_train, X_valid, y_train, y_valid = train_test_split(X['Annual Expenses'], y)

    model = stats.linregress(X_train,y_train)

    # Plot Annual Expenses vs. Annual Income with regression line
    plt.plot(X_valid, y_valid, 'b.')
    plt.plot(X_valid, X_valid*model.slope + model.intercept, 'r-', linewidth=3)
    plt.xlabel('Annual Expenses')
    plt.ylabel('Annual Income')
    plt.title('Annual Expenses vs. Annual Income')
    plt.legend(['Original Data', 'Regression Line'])

    plt.tight_layout()
    plt.show()    




def main():

    Xred = reddit_data[['Annual Expenses', 'Total Asset Value']]
    yred = reddit_data["Annual Income"].values

    #Running models on reddit data
    # linear_reg_model_ML(Xred,yred)
    # bayesian_model(Xred,yred)
    # knn_model(Xred,yred)
    # randomforest_model(Xred,yred)
    linear_reg_stats(Xred,yred)

main()  