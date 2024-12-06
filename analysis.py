import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy import stats
import statsmodels.api as sm

# OKASHA
reddit_data2023 = pd.read_csv("data/clean_reddit2023.csv")
reddit_data2022 = pd.read_csv("data/clean_reddit2022.csv")
reddit_data2021 = pd.read_csv("data/clean_reddit2021.csv")
reddit_data2020 = pd.read_csv("data/clean_reddit2020.csv")
reddit_data2018 = pd.read_csv("data/clean_reddit2018.csv")
reddit_data2017 = pd.read_csv("data/clean_reddit2017.csv")
reddit_data2016 = pd.read_csv("data/clean_reddit2016.csv")

reddit_data2023indexp = pd.read_csv("data/clean_reddit2023indexp.csv")
reddit_data2022indexp = pd.read_csv("data/clean_reddit2022indexp.csv")
reddit_data2021indexp = pd.read_csv("data/clean_reddit2021indexp.csv")
reddit_data2020indexp = pd.read_csv("data/clean_reddit2020indexp.csv")
reddit_data2018indexp = pd.read_csv("data/clean_reddit2018indexp.csv")

# STEFAN
# reddit_data2023 = pd.read_csv("data\clean_reddit2023.csv")
# reddit_data2022 = pd.read_csv("data\clean_reddit2022.csv")
# reddit_data2021 = pd.read_csv("data\clean_reddit2021.csv")
# reddit_data2020 = pd.read_csv("data\clean_reddit2020.csv")
# reddit_data2018 = pd.read_csv("data\clean_reddit2018.csv")
# reddit_data2017 = pd.read_csv("data\clean_reddit2017.csv")
# reddit_data2016 = pd.read_csv("data\clean_reddit2016.csv")

# 2017 and 16 not included as they are lacking many of the expenses columns.
# reddit_data2023indexp = pd.read_csv("data\clean_reddit2023indexp.csv")
# reddit_data2022indexp = pd.read_csv("data\clean_reddit2022indexp.csv")
# reddit_data2021indexp = pd.read_csv("data\clean_reddit2021indexp.csv")
# reddit_data2020indexp = pd.read_csv("data\clean_reddit2020indexp.csv")
# reddit_data2018indexp = pd.read_csv("data\clean_reddit2018indexp.csv")

# PRELIMINARY STATISTICS
# the datasets we are working on for annual income and annual expenses
datasets = [
    reddit_data2023, reddit_data2022, reddit_data2021, reddit_data2020,
    reddit_data2018, reddit_data2017, reddit_data2016
]

# specifies the years we are working on and adds a year column to each database
years = [2023, 2022, 2021, 2020, 2018, 2017, 2016]
for df, year in zip(datasets, years):
    df['Year'] = year

# combines the datasets and groups by years and means the annual income and expenses
combined_data = pd.concat(datasets, ignore_index=True)
summary = combined_data.groupby('Year')[['Annual Income', 'Annual Expenses']].mean().reset_index()


# lists the expense columns we want, and all the individual expense datasets
expense_columns = ['Annual Income', 'Annual Expenses', 'Housing', 'Utilities', 'Transportation', 'Necessities', 'Luxuries', 'Children', 'Debt Repayment', 'Charity', 'Healthcare', 'Taxes', 'Education', 'Other']
datasetsindexp = [
    reddit_data2023indexp, reddit_data2022indexp, reddit_data2021indexp, reddit_data2020indexp,
    reddit_data2018indexp
]

# specifies years that we have avaliable and adds a column called 'Year' so we can associate expenses with a year
yearsindexp = [2023, 2022, 2021, 2020, 2018]
for df, year in zip(datasetsindexp,yearsindexp):
    df['Year'] = year

# concatenates the datasets together into one dataset, then groups by year and finds the mean of each years expense values
combined_expdata = pd.concat(datasetsindexp,ignore_index=True)
aggregated_expenses = combined_expdata.groupby('Year')[expense_columns].mean()


# PLOTS AND VISUALS
# plots that show information about expense trends over the years and the percentage change of expense spending over the years
def ind_exp_plots():
    # calculates expense growth rate
    expense_growth = aggregated_expenses.pct_change() * 100
    # plot trends for all expense columns
    aggregated_expenses.plot(kind='line', figsize=(10, 6))
    plt.title('Expense Trends Over Years')
    plt.ylabel('Average Expense Value')
    plt.xlabel('Year')
    plt.grid()
    plt.show()

    # plot percentage changes
    expense_growth.plot(kind='bar', figsize=(10, 6), rot=45)
    plt.title('Year-over-Year Growth/Decrease in Expenses')
    plt.ylabel('Percentage Change')
    plt.xlabel('Year')
    plt.grid()
    plt.show()



# split data into pre- and post-pandemic periods
pre_pandemic = combined_data[combined_data['Year'] < 2020]
pandemic = combined_data[combined_data['Year'] >= 2020]

# graph about savings rate
def saving_rate_graph():
    combined_data['Savings Rate'] = (combined_data['Annual Income'] - combined_data['Annual Expenses']) / combined_data['Annual Income']
    savings_summary = combined_data.groupby('Year')['Savings Rate'].mean().reset_index()
    sns.lineplot(x='Year', y='Savings Rate', data=savings_summary)

# density plot on annual income
def density_plot_income():
    sns.kdeplot(data=pre_pandemic, x='Annual Income', label='Pre-Pandemic', fill=True)
    sns.kdeplot(data=pandemic, x='Annual Income', label='Pandemic Era', fill=True)
    plt.legend()

def p_tests():
    # perform t-tests for income and expenses
    income_ttest = stats.ttest_ind(pre_pandemic['Annual Income'], pandemic['Annual Income'], nan_policy='omit')
    expenses_ttest = stats.ttest_ind(pre_pandemic['Annual Expenses'], pandemic['Annual Expenses'], nan_policy='omit')


    print("T-test Results:")
    print(f"Annual Income: t-statistic = {income_ttest.statistic}, p-value = {income_ttest.pvalue}")
    print(f"Annual Expenses: t-statistic = {expenses_ttest.statistic}, p-value = {expenses_ttest.pvalue}")

    # Mann-Whitney U Test for Annual Income and Annual Expenses
    income_mwutest = stats.mannwhitneyu(pre_pandemic['Annual Income'], pandemic['Annual Income'], alternative='two-sided')
    expenses_mwutest = stats.mannwhitneyu(pre_pandemic['Annual Expenses'], pandemic['Annual Expenses'], alternative='two-sided')

    print("Mann-Whitney U-Test Results:")
    print(f"Annual Income: U-statistic = {income_mwutest.statistic}, p-value = {income_mwutest.pvalue}")
    print(f"Annual Expenses: U-statistic = {expenses_mwutest.statistic}, p-value = {expenses_mwutest.pvalue}")

def income_expense_trend_plot():
    # plot trends
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

# function which produces the CAGR stats for annual income and expenses, compares which one is growing at a faster rate
def cagr_stats():
    # process for calculating CAGR, Compound Annual Growth Rate. 
    # adapted from: https://dabure.com/blog/calculating-cagr-in-pandas
    
    # extract the starting and ending years for both metrics
    start_year = summary['Year'].min()
    end_year = summary['Year'].max()

    # calculate CAGR for income
    start_income = summary.loc[summary['Year'] == start_year, 'Annual Income'].values[0]
    end_income = summary.loc[summary['Year'] == end_year, 'Annual Income'].values[0]
    cagr_income = ((end_income / start_income) ** (1 / (end_year - start_year))) - 1

    # calculate CAGR for expenses
    start_expenses = summary.loc[summary['Year'] == start_year, 'Annual Expenses'].values[0]
    end_expenses = summary.loc[summary['Year'] == end_year, 'Annual Expenses'].values[0]
    cagr_expenses = ((end_expenses / start_expenses) ** (1 / (end_year - start_year))) - 1

    # print results
    print(f"CAGR for Annual Income: {cagr_income:.2%}")
    print(f"CAGR for Annual Expenses: {cagr_expenses:.2%}")

    # compare growth rates
    if cagr_income > cagr_expenses:
        print("Annual Income is growing faster.")
    elif cagr_expenses > cagr_income:
        print("Annual Expenses are growing faster.")
    else:
        print("Both are growing at the same rate.")

def validate_2023_prediction():
    # filter data for training (2020-2022) and testing (2023)
    training_data = combined_data[(combined_data['Year'] >= 2017) & (combined_data['Year'] <= 2022)]
    test_data = combined_data[combined_data['Year'] == 2023]

    # features (X) and target (y)
    X_train = training_data[['Annual Expenses']]
    y_train = training_data['Annual Income']
    X_test = test_data[['Annual Expenses']]
    y_test = test_data['Annual Income']

    # train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # print model scores
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Training Score: {train_score:.4f}")
    print(f"Testing Score: {test_score:.4f}")

    # predict for 2023
    predictions_2023 = model.predict(X_test)

    # compare predictions with actual values
    test_data = test_data.copy()
    test_data['Predicted Annual Income'] = predictions_2023
    test_data['Difference'] = test_data['Predicted Annual Income'] - test_data['Annual Income']

    # print results
    print("2023 Predictions vs Actuals:")
    print(test_data[['Annual Expenses', 'Annual Income', 'Predicted Annual Income', 'Difference']])

    # plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(test_data['Annual Expenses'], test_data['Annual Income'], color='blue', label='Actual')
    plt.scatter(test_data['Annual Expenses'], test_data['Predicted Annual Income'], color='red', label='Predicted', alpha=0.7)
    plt.plot(test_data['Annual Expenses'], test_data['Predicted Annual Income'], color='green', label='Regression Line')
    plt.xlabel('Annual Expenses')
    plt.ylabel('Annual Income')
    plt.title('Actual vs Predicted Annual Income for 2023')
    plt.legend()
    plt.grid()
    plt.show()

def main():
    p_tests()
    density_plot_income()
    saving_rate_graph()
    income_expense_trend_plot()
    cagr_stats()
    saving_rate_graph()
    validate_2023_prediction()

main()  