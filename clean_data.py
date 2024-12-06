import pandas as pd


# IMPORT DATA
reddit_data2023 = pd.read_csv("data/reddit_financedata2023.csv")
reddit_data2022 = pd.read_csv("data/reddit_financedata2022.csv")
reddit_data2021 = pd.read_csv("data/reddit_financedata2021.csv")
reddit_data2020 = pd.read_csv("data/reddit_financedata2020.csv")
reddit_data2018 = pd.read_csv("data/reddit_financedata2018.csv")
reddit_data2017 = pd.read_csv("data/reddit_financedata2017.csv")
reddit_data2016 = pd.read_csv("data/reddit_financedata2016.csv")


# CLEAN REDDIT DATA
# keep only columns that are relevant to our use case
# we want to have expense columns that are common to all years. we will try to follow 2023s columns of expenses.
# clarification on transporation = car + any other stuff like buses. necessities is clothing, grocery, personal care, household supplies. luxuries is restaurant, entertainment, travel, pets, gifts
expense_columns2023analysis = ['Housing', 'Utilities', 'Transportation', 'Necessities', 'Luxuries', 'Children', 'Debt Repayment', 'Charity', 'Healthcare', 'Taxes', 'Education', 'Other']
reddit_data2023indexp = reddit_data2023[['Annual Income', 'Annual Expenses'] + expense_columns2023analysis]
reddit_data2023 = reddit_data2023[["Annual Expenses", "Annual Income"]]

expense_columns2022analysis = ['Housing', 'Utilities', 'Transportation', 'Necessities', 'Luxuries', 'Children', 'Debt Repayment', 'Charity', 'Healthcare', 'Taxes', 'Education', 'Other']
reddit_data2022indexp = reddit_data2022[['Annual Income', 'Annual Expenses'] + expense_columns2022analysis]
reddit_data2022 = reddit_data2022[["Annual Expenses", "Annual Income"]]


expense_columns2021analysis = ['Housing', 'Utilities', 'Transportation', 'Necessities', 'Luxuries', 'Children', 'Debt Repayment', 'Charity', 'Healthcare', 'Taxes', 'Education', 'Other']
reddit_data2021indexp = reddit_data2021[['Annual Income', 'Annual Expenses'] + expense_columns2021analysis]
reddit_data2021 = reddit_data2021[["Annual Expenses", "Annual Income"]]

expense_columns2020 = ['Housing', 'Utilities', 'Transportation', 'Necessities', 'Luxuries', 'Children', 'Debt Repayment', 'Charity', 'Healthcare', 'Taxes', 'Education', 'Other']
reddit_data2020['Annual Expenses'] = reddit_data2020[expense_columns2020].sum(axis=1)
expense_columns2020analysis = ['Housing', 'Utilities', 'Transportation', 'Necessities', 'Luxuries', 'Children', 'Debt Repayment', 'Charity', 'Healthcare', 'Taxes', 'Education', 'Other']
reddit_data2020indexp = reddit_data2020[['Annual Income', 'Annual Expenses'] + expense_columns2020analysis]
reddit_data2020 = reddit_data2020[["Annual Expenses", "Annual Income"]]

expense_columns2018 = ['Housing', 'Utilities', 'Childcare', 'Transportation', 'Groceries', 'Dining', 'Personal Care', 'Clothing', 'Pets', 'Gifts', 'Entertainment', 'Debt Repayment', 'Vacation', 'Charity', 'Healthcare', 'Taxes', 'Education', 'Other']
reddit_data2018['Annual Expenses'] = reddit_data2018[expense_columns2018].sum(axis=1)
reddit_data2018['Children'] = reddit_data2018[['Childcare']].sum(axis=1)
reddit_data2018['Necessities'] = reddit_data2018[['Groceries', 'Personal Care', 'Clothing']].sum(axis=1)
reddit_data2018['Luxuries'] = reddit_data2018[['Dining', 'Pets', 'Gifts', 'Entertainment', 'Vacation']].sum(axis=1)
expense_columns2018analysis = ['Housing', 'Utilities', 'Transportation', 'Necessities', 'Luxuries', 'Children', 'Debt Repayment', 'Charity', 'Healthcare', 'Taxes', 'Education', 'Other']
reddit_data2018indexp = reddit_data2018[['Annual Income', 'Annual Expenses'] + expense_columns2018analysis]
reddit_data2018 = reddit_data2018[["Annual Expenses", "Annual Income"]]

expense_columns2017 = ['Housing', 'Utilities', 'Childcare', 'Transportation', 'Groceries', 'Dining', 'Personal Care', 'Clothing', 'Car Payment', 'Entertainment', 'Debt Repayment', 'Vacation', 'Auto Fuel & Maintenance', 'Insurance', 'Open-Ended Response']
reddit_data2017['Annual Expenses'] = reddit_data2017[expense_columns2017].sum(axis=1)
reddit_data2017 = reddit_data2017[["Annual Expenses", "Annual Income"]]

expense_columns2016 = ['Housing', 'Utilities', 'Childcare', 'Transportation', 'Groceries', 'Dining', 'Personal Care', 'Clothing', 'Car Payment', 'Entertainment', 'Debt Repayment', 'Vacation', 'Auto Fuel & Maintenance', 'Insurance', 'Open-Ended Response']
reddit_data2016['Annual Expenses'] = reddit_data2016[expense_columns2016].sum(axis=1)
reddit_data2016 = reddit_data2016[["Annual Expenses", "Annual Income"]]

# correlation Calculation
correlations = reddit_data2023.corr()
print(correlations["Annual Income"].sort_values(ascending=False))

# keep only columns where values are NOT zero
reddit_data2023 = reddit_data2023[(reddit_data2023 != 0).all(axis=1)]
reddit_data2023 = reddit_data2023.dropna()

# only not zero and NA for now with individual expenses columns
reddit_data2023indexp = reddit_data2023indexp[(reddit_data2023indexp[['Annual Income', 'Annual Expenses']] != 0).all(axis=1)]
reddit_data2023indexp = reddit_data2023indexp.dropna()

reddit_data2022 = reddit_data2022[(reddit_data2022 != 0).all(axis=1)]
reddit_data2022 = reddit_data2022.dropna()

# only not zero and NA for now with individual expenses columns
reddit_data2022indexp = reddit_data2022indexp[(reddit_data2022indexp[['Annual Income', 'Annual Expenses']] != 0).all(axis=1)]
reddit_data2022indexp = reddit_data2022indexp.dropna()

reddit_data2021 = reddit_data2021[(reddit_data2021 != 0).all(axis=1)]
reddit_data2021 = reddit_data2021.dropna()

# only not zero and NA for now with individual expenses columns
reddit_data2021indexp = reddit_data2021indexp[(reddit_data2021indexp[['Annual Income', 'Annual Expenses']] != 0).all(axis=1)]
reddit_data2021indexp = reddit_data2021indexp.dropna()

reddit_data2020 = reddit_data2020[(reddit_data2020 != 0).all(axis=1)]
reddit_data2020 = reddit_data2020.dropna()

# only not zero and NA for now with individual expenses columns
reddit_data2020indexp = reddit_data2020indexp[(reddit_data2020indexp[['Annual Income', 'Annual Expenses']] != 0).all(axis=1)]
reddit_data2020indexp = reddit_data2020indexp.dropna()

reddit_data2018 = reddit_data2018[(reddit_data2018 != 0).all(axis=1)]
reddit_data2018 = reddit_data2018.dropna()

# only not zero and NA for now with individual expenses columns
reddit_data2018indexp = reddit_data2018indexp[(reddit_data2018indexp[['Annual Income', 'Annual Expenses']] != 0).all(axis=1)]
reddit_data2018indexp = reddit_data2018indexp.dropna()

reddit_data2017 = reddit_data2017[(reddit_data2017 != 0).all(axis=1)]
reddit_data2017 = reddit_data2017.dropna()

reddit_data2016 = reddit_data2016[(reddit_data2016 != 0).all(axis=1)]
reddit_data2016 = reddit_data2016.dropna()

# convert int to float, for consistency
reddit_data2023["Annual Expenses"] = reddit_data2023["Annual Expenses"].astype("float")
reddit_data2023["Annual Income"] = reddit_data2023["Annual Income"].astype("float")

reddit_data2022["Annual Expenses"] = reddit_data2022["Annual Expenses"].astype("float")
reddit_data2022["Annual Income"] = reddit_data2022["Annual Income"].astype("float")

reddit_data2021["Annual Expenses"] = reddit_data2021["Annual Expenses"].astype("float")
reddit_data2021["Annual Income"] = reddit_data2021["Annual Income"].astype("float")

reddit_data2020["Annual Expenses"] = reddit_data2020["Annual Expenses"].astype("float")
reddit_data2020["Annual Income"] = reddit_data2020["Annual Income"].astype("float")

reddit_data2018["Annual Expenses"] = reddit_data2018["Annual Expenses"].astype("float")
reddit_data2018["Annual Income"] = reddit_data2018["Annual Income"].astype("float")

reddit_data2017["Annual Expenses"] = reddit_data2017["Annual Expenses"].astype("float")
reddit_data2017["Annual Income"] = reddit_data2017["Annual Income"].astype("float")

reddit_data2016["Annual Expenses"] = reddit_data2016["Annual Expenses"].astype("float")
reddit_data2016["Annual Income"] = reddit_data2016["Annual Income"].astype("float")

reddit_data2023indexp = reddit_data2023indexp.astype(float)
reddit_data2022indexp = reddit_data2022indexp.astype(float)
reddit_data2021indexp = reddit_data2021indexp.astype(float)
reddit_data2020indexp = reddit_data2020indexp.astype(float)
reddit_data2018indexp = reddit_data2018indexp.astype(float)

# FILTER OUTLIERS
reddit_data2023 = reddit_data2023[(reddit_data2023["Annual Income"] >= 20000) & 
                                  (reddit_data2023["Annual Income"] <= 1000000)]

reddit_data2022 = reddit_data2022[(reddit_data2022["Annual Income"] >= 20000) & 
                                  (reddit_data2022["Annual Income"] <= 1000000)]

reddit_data2021 = reddit_data2021[(reddit_data2021["Annual Income"] >= 20000) & 
                                  (reddit_data2021["Annual Income"] <= 1000000)]

reddit_data2020 = reddit_data2020[(reddit_data2020["Annual Income"] >= 20000) & 
                                  (reddit_data2020["Annual Income"] <= 1000000)]

reddit_data2018 = reddit_data2018[(reddit_data2018["Annual Income"] >= 20000) & 
                                  (reddit_data2018["Annual Income"] <= 1000000)]

reddit_data2017 = reddit_data2017[(reddit_data2017["Annual Income"] >= 20000) & 
                                  (reddit_data2017["Annual Income"] <= 1000000)]

reddit_data2016 = reddit_data2016[(reddit_data2016["Annual Income"] >= 20000) & 
                                  (reddit_data2016["Annual Income"] <= 1000000)]


reddit_data2023indexp = reddit_data2023indexp[(reddit_data2023indexp["Annual Income"] >= 20000) & 
                                              (reddit_data2023indexp["Annual Income"] <= 1000000)]

reddit_data2022indexp = reddit_data2022indexp[(reddit_data2022indexp["Annual Income"] >= 20000) & 
                                              (reddit_data2022indexp["Annual Income"] <= 1000000)]

reddit_data2021indexp = reddit_data2021indexp[(reddit_data2021indexp["Annual Income"] >= 20000) & 
                                              (reddit_data2021indexp["Annual Income"] <= 1000000)]

reddit_data2020indexp = reddit_data2020indexp[(reddit_data2020indexp["Annual Income"] >= 20000) & 
                                              (reddit_data2020indexp["Annual Income"] <= 1000000)]

reddit_data2018indexp = reddit_data2018indexp[(reddit_data2018indexp["Annual Income"] >= 20000) & 
                                              (reddit_data2018indexp["Annual Income"] <= 1000000)]

# EXPORT DATA TO CSV FILES
# puts the cleaned data in a new .csv files
reddit_data2023.to_csv("data/clean_reddit2023.csv", index=True)
reddit_data2022.to_csv("data/clean_reddit2022.csv", index=True)
reddit_data2021.to_csv("data/clean_reddit2021.csv", index=True)
reddit_data2020.to_csv("data/clean_reddit2020.csv", index=True)
reddit_data2018.to_csv("data/clean_reddit2018.csv", index=True)
reddit_data2017.to_csv("data/clean_reddit2017.csv", index=True)
reddit_data2016.to_csv("data/clean_reddit2016.csv", index=True)


reddit_data2023indexp.to_csv("data/clean_reddit2023indexp.csv", index=True)
reddit_data2022indexp.to_csv("data/clean_reddit2022indexp.csv", index=True)
reddit_data2021indexp.to_csv("data/clean_reddit2021indexp.csv", index=True)
reddit_data2020indexp.to_csv("data/clean_reddit2020indexp.csv", index=True)
reddit_data2018indexp.to_csv("data/clean_reddit2018indexp.csv", index=True)