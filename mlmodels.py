import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from skimage.color import rgb2lab
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("cmpt353dataproject/data/clean_data.csv")

categorical = data[["Ethnicity", "Gender", "Age", "Relationship Status", "Industry", "Education", "Cost of Living"]]
numerical = data[["Annual Income", "Annual Expenses"]]


label_encoders = {}
for column in categorical.columns:
    label_encoders[column] = LabelEncoder()
    categorical[column] = label_encoders[column].fit_transform(categorical[column])

scaler = StandardScaler()
numerical_scaled = scaler.fit_transform(numerical)

X = pd.concat([pd.DataFrame(numerical_scaled, columns=numerical.columns)])
y = data["Relationship Status"].values

def bayesian_model():
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)

    model = GaussianNB()
    model.fit(X_train, y_train)

    model.predict(X_valid, y_valid)
    print(model.score(X_valid, y_valid))


def main():
    bayesian_model()

    