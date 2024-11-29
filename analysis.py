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
reddit_data2023 = pd.read_csv("data\clean_reddit2023.csv")
reddit_data2022 = pd.read_csv("data\clean_reddit2022.csv")
reddit_data2021 = pd.read_csv("data\clean_reddit2021.csv")
reddit_data2020 = pd.read_csv("data\clean_reddit2020.csv")
reddit_data2018 = pd.read_csv("data\clean_reddit2018.csv")
reddit_data2017 = pd.read_csv("data\clean_reddit2017.csv")
reddit_data2016 = pd.read_csv("data\clean_reddit2016.csv")

datasets = [
    reddit_data2023, reddit_data2022, reddit_data2021, reddit_data2020,
    reddit_data2018, reddit_data2017, reddit_data2016
]

years = [2023, 2022, 2021, 2020, 2018, 2017, 2016]
for df, year in zip(datasets, years):
    df['Year'] = year

combined_data = pd.concat(datasets, ignore_index=True)

summary = combined_data.groupby('Year')[['Annual Income', 'Annual Expenses']].mean().reset_index()

# Split data into pre-pandemic and pandemic-era
pre_pandemic = combined_data[combined_data['Year'] < 2020]
pandemic = combined_data[combined_data['Year'] >= 2020]

def saving_rate_graph():
    combined_data['Savings Rate'] = (combined_data['Annual Income'] - combined_data['Annual Expenses']) / combined_data['Annual Income']
    savings_summary = combined_data.groupby('Year')['Savings Rate'].mean().reset_index()
    sns.lineplot(x='Year', y='Savings Rate', data=savings_summary)

def density_plot_income():
    sns.kdeplot(data=pre_pandemic, x='Annual Income', label='Pre-Pandemic', shade=True)
    sns.kdeplot(data=pandemic, x='Annual Income', label='Pandemic Era', shade=True)
    plt.legend()

def p_tests():
    # Perform t-tests for income and expenses
    income_ttest = stats.ttest_ind(pre_pandemic['Annual Income'], pandemic['Annual Income'], nan_policy='omit')
    expenses_ttest = stats.ttest_ind(pre_pandemic['Annual Expenses'], pandemic['Annual Expenses'], nan_policy='omit')


    print("Ttest Results:")
    print(f"Annual Income: U-statistic = {income_ttest.statistic}, p-value = {income_ttest.pvalue}")
    print(f"Annual Expenses: U-statistic = {expenses_ttest.statistic}, p-value = {expenses_ttest.pvalue}")

    # Mann-Whitney U Test for Annual Income and Annual Expenses
    income_mwutest = stats.mannwhitneyu(pre_pandemic['Annual Income'], pandemic['Annual Income'], alternative='two-sided')
    expenses_mwutest = stats.mannwhitneyu(pre_pandemic['Annual Expenses'], pandemic['Annual Expenses'], alternative='two-sided')

    print("Mann-Whitney U Test Results:")
    print(f"Annual Income: U-statistic = {income_mwutest.statistic}, p-value = {income_mwutest.pvalue}")
    print(f"Annual Expenses: U-statistic = {expenses_mwutest.statistic}, p-value = {expenses_mwutest.pvalue}")

def income_expense_trend_plot():
    # Plot trends
    plt.figure(figsize=(10, 6))
    plt.plot(summary['Year'], summary['Annual Income'], label='Annual Income', marker='o')
    plt.plot(summary['Year'], summary['Annual Expenses'], label='Annual Expenses', marker='o')
    plt.axvline(2020, color='red', linestyle='--', label='Pandemic (2020)')
    plt.xlabel('Year')
    plt.ylabel('Amount')
    plt.title('Trends in Annual Income and Expenses')
    plt.legend()
    plt.grid()
    plt.show()

#Function which produces the CAGR stats for annual income and expenses, compares which one is growing at a faster rate.
def cagr_stats():
    #Process for calculating CAGR, Compound annual growth rate. https://dabure.com/blog/calculating-cagr-in-pandas From this link process derived.
    # Extract the starting and ending years for both metrics
    start_year = summary['Year'].min()
    end_year = summary['Year'].max()

    # Calculate CAGR for Income
    start_income = summary.loc[summary['Year'] == start_year, 'Annual Income'].values[0]
    end_income = summary.loc[summary['Year'] == end_year, 'Annual Income'].values[0]
    cagr_income = ((end_income / start_income) ** (1 / (end_year - start_year))) - 1

    # Calculate CAGR for Expenses
    start_expenses = summary.loc[summary['Year'] == start_year, 'Annual Expenses'].values[0]
    end_expenses = summary.loc[summary['Year'] == end_year, 'Annual Expenses'].values[0]
    cagr_expenses = ((end_expenses / start_expenses) ** (1 / (end_year - start_year))) - 1

    # Print Results
    print(f"CAGR for Annual Income: {cagr_income:.2%}")
    print(f"CAGR for Annual Expenses: {cagr_expenses:.2%}")

    # Compare growth rates
    if cagr_income > cagr_expenses:
        print("Annual Income is growing faster.")
    elif cagr_expenses > cagr_income:
        print("Annual Expenses are growing faster.")
    else:
        print("Both are growing at the same rate.")

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

    p_tests()
    income_expense_trend_plot()
    cagr_stats()
    saving_rate_graph()
    # Xred = reddit_data[['Annual Expenses', 'Total Asset Value']]
    # yred = reddit_data["Annual Income"].values

    # #Running models on reddit data
    # # linear_reg_model_ML(Xred,yred)
    # # bayesian_model(Xred,yred)
    # # knn_model(Xred,yred)
    # # randomforest_model(Xred,yred)
    # linear_reg_stats(Xred,yred)

main()  