import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


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
#We want to have expense columns that are common to all years. We will try to follow 2023s columns of expenses.
#Clarification on transporation = car + any other stuff like buses. Necessities is clothing, grocery, personal care, household supplies. Luxiries is restaurant, entertainment, travel, pets, gifts
expense_columns2023analysis = ['Housing','Utilities','Transportation','Necessities','Luxuries','Children','Debt Repayment','Charity','Healthcare','Taxes','Education','Other']
reddit_data2023indexp = reddit_data2023[['Annual Income','Annual Expenses']+expense_columns2023analysis]
reddit_data2023 = reddit_data2023[["Annual Expenses", "Annual Income"]]

expense_columns2022analysis = ['Housing','Utilities','Transportation','Necessities','Luxuries','Children','Debt Repayment','Charity','Healthcare','Taxes','Education','Other']
reddit_data2022indexp = reddit_data2022[['Annual Income','Annual Expenses']+expense_columns2022analysis]
reddit_data2022 = reddit_data2022[["Annual Expenses", "Annual Income"]]


expense_columns2021analysis = ['Housing','Utilities','Transportation','Necessities','Luxuries','Children','Debt Repayment','Charity','Healthcare','Taxes','Education','Other']
reddit_data2021indexp = reddit_data2021[['Annual Income','Annual Expenses']+expense_columns2021analysis]
reddit_data2021 = reddit_data2021[["Annual Expenses", "Annual Income"]]

expense_columns2020 = ['Housing','Utilities','Transportation','Necessities','Luxuries','Children','Debt Repayment','Savings','Charity','Healthcare','Taxes','Education','Other']
reddit_data2020['Annual Expenses'] = reddit_data2020[expense_columns2020].sum(axis=1)
expense_columns2020analysis = ['Housing','Utilities','Transportation','Necessities','Luxuries','Children','Debt Repayment','Charity','Taxes','Education','Other']
reddit_data2020indexp = reddit_data2020[['Annual Income','Annual Expenses']+expense_columns2020analysis]
reddit_data2020 = reddit_data2020[["Annual Expenses", "Annual Income"]]

expense_columns2018 = ['Housing','Utilities','Childcare','Transportation','Groceries','Dining','Personal Care','Clothing','Pets','Gifts','Entertainment','Debt Repayment','Vacation','Charity','Healthcare','Taxes','Education','Other']
reddit_data2018['Annual Expenses'] = reddit_data2018[expense_columns2018].sum(axis=1)
reddit_data2018['Children'] = reddit_data2018[['Childcare']].sum(axis=1)
reddit_data2018['Necessities'] = reddit_data2018[['Groceries','Personal Care','Clothing']].sum(axis=1)
reddit_data2018['Luxuries'] = reddit_data2018[['Dining','Pets','Gifts','Entertainment','Vacation']].sum(axis=1)
expense_columns2018analysis = ['Housing','Utilities','Transportation','Necessities','Luxuries','Children','Debt Repayment','Charity','Healthcare','Taxes','Education','Other']
reddit_data2018indexp = reddit_data2018[['Annual Income','Annual Expenses']+expense_columns2018analysis]
reddit_data2018 = reddit_data2018[["Annual Expenses", "Annual Income"]]

expense_columns2017 = ['Housing','Utilities','Childcare','Transportation','Groceries','Dining','Personal Care','Clothing','Car Payment','Entertainment','Debt Repayment','Vacation','Auto Fuel & Maintenance','Insurance','Open-Ended Response']
reddit_data2017['Annual Expenses'] = reddit_data2017[expense_columns2017].sum(axis=1)
# reddit_data2017['Children'] = reddit_data2017[['Childcare']].sum(axis=1)
# reddit_data2017['Housing'] = reddit_data2017[['Housing','Insurance']].sum(axis=1)
# reddit_data2017['Transportation'] = reddit_data2017[['Transportation','Car Payment','Auto Fuel & Maintenance']].sum(axis=1)
# reddit_data2017['Necessities'] = reddit_data2017[['Groceries','Personal Care','Clothing']].sum(axis=1)
# reddit_data2017['Luxuries'] = reddit_data2017[['Dining','Entertainment','Vacation']].sum(axis=1)
# reddit_data2017['Other'] = reddit_data2017[['Open-Ended Response']].sum(axis=1)
# expense_columns2017analysis = ['Housing','Utilities','Transportation','Necessities','Luxuries','Children','Debt Repayment','Charity','Healthcare','Taxes','Education','Other']
# reddit_data2017indexp = reddit_data2017[['Annual Income','Annual Expenses']+expense_columns2017analysis]
reddit_data2017 = reddit_data2017[["Annual Expenses", "Annual Income"]]

expense_columns2016 = ['Housing','Utilities','Childcare','Transportation','Groceries','Dining','Personal Care','Clothing','Car Payment','Entertainment','Debt Repayment','Vacation','Auto Fuel & Maintenance','Insurance','Open-Ended Response']
reddit_data2016['Annual Expenses'] = reddit_data2016[expense_columns2016].sum(axis=1)
# reddit_data2016['Children'] = reddit_data2016[['Childcare']].sum(axis=1)
# reddit_data2016['Housing'] = reddit_data2016[['Housing','Insurance']].sum(axis=1)
# reddit_data2016['Transportation'] = reddit_data2016[['Transportation','Car Payment','Auto Fuel & Maintenance']].sum(axis=1)
# reddit_data2016['Necessities'] = reddit_data2016[['Groceries','Personal Care','Clothing']].sum(axis=1)
# reddit_data2016['Luxuries'] = reddit_data2016[['Dining','Entertainment','Vacation']].sum(axis=1)
# reddit_data2016['Other'] = reddit_data2016[['Open-Ended Response']].sum(axis=1)
# expense_columns2016analysis = ['Housing','Utilities','Transportation','Necessities','Luxuries','Children','Debt Repayment','Charity','Healthcare','Taxes','Education','Other']
# reddit_data2016indexp = reddit_data2016[['Annual Income','Annual Expenses']+expense_columns2016analysis]
reddit_data2016 = reddit_data2016[["Annual Expenses", "Annual Income"]]

#Correlation Calculation
correlations = reddit_data2023.corr()
print(correlations["Annual Income"].sort_values(ascending=False))

# keep only columns where values are NOT zero
reddit_data2023 = reddit_data2023[(reddit_data2023 != 0).all(axis=1)]
reddit_data2023 = reddit_data2023.dropna()

# Only not zero and NA for now with individual expenses columns
reddit_data2023indexp = reddit_data2023indexp[(reddit_data2023indexp[['Annual Income','Annual Expenses']] != 0).all(axis=1)]
reddit_data2023indexp = reddit_data2023indexp.dropna()

reddit_data2022 = reddit_data2022[(reddit_data2022 != 0).all(axis=1)]
reddit_data2022 = reddit_data2022.dropna()

# Only not zero and NA for now with individual expenses columns
reddit_data2022indexp = reddit_data2022indexp[(reddit_data2022indexp[['Annual Income','Annual Expenses']] != 0).all(axis=1)]
reddit_data2022indexp = reddit_data2022indexp.dropna()

reddit_data2021 = reddit_data2021[(reddit_data2021 != 0).all(axis=1)]
reddit_data2021 = reddit_data2021.dropna()

# Only not zero and NA for now with individual expenses columns
reddit_data2021indexp = reddit_data2021indexp[(reddit_data2021indexp[['Annual Income','Annual Expenses']] != 0).all(axis=1)]
reddit_data2021indexp = reddit_data2021indexp.dropna()

reddit_data2020 = reddit_data2020[(reddit_data2020 != 0).all(axis=1)]
reddit_data2020 = reddit_data2020.dropna()

# Only not zero and NA for now with individual expenses columns
reddit_data2020indexp = reddit_data2020indexp[(reddit_data2020indexp[['Annual Income','Annual Expenses']] != 0).all(axis=1)]
reddit_data2020indexp = reddit_data2020indexp.dropna()

reddit_data2018 = reddit_data2018[(reddit_data2018 != 0).all(axis=1)]
reddit_data2018 = reddit_data2018.dropna()

# Only not zero and NA for now with individual expenses columns
reddit_data2018indexp = reddit_data2018indexp[(reddit_data2018indexp[['Annual Income','Annual Expenses']] != 0).all(axis=1)]
reddit_data2018indexp = reddit_data2018indexp.dropna()

reddit_data2017 = reddit_data2017[(reddit_data2017 != 0).all(axis=1)]
reddit_data2017 = reddit_data2017.dropna()

# Only not zero and NA for now with individual expenses columns
# reddit_data2017indexp = reddit_data2017indexp[(reddit_data2017indexp != 0).all(axis=1)]
# reddit_data2017indexp = reddit_data2017indexp.dropna()

reddit_data2016 = reddit_data2016[(reddit_data2016 != 0).all(axis=1)]
reddit_data2016 = reddit_data2016.dropna()

# Only not zero and NA for now with individual expenses columns
# reddit_data2016indexp = reddit_data2016indexp[(reddit_data2016indexp != 0).all(axis=1)]
# reddit_data2016indexp = reddit_data2016indexp.dropna()

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
# reddit_data2017indexp = reddit_data2017indexp.astype(float)
# reddit_data2016indexp = reddit_data2016indexp.astype(float)

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

# reddit_data2017indexp = reddit_data2017indexp[(reddit_data2017indexp["Annual Income"] >= 20000) & 
#                           (reddit_data2017indexp["Annual Income"] <= 1000000)]

# reddit_data2016indexp = reddit_data2016indexp[(reddit_data2016indexp["Annual Income"] >= 20000) & 
#                           (reddit_data2016indexp["Annual Income"] <= 1000000)]
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
# reddit_data2017indexp.to_csv("data/clean_reddit2017indexp.csv", index=True)
# reddit_data2016indexp.to_csv("data/clean_reddit2016indexp.csv", index=True)
# PRELIMINARY GRAPHS (to get an idea of the data)
def extract_midpoint(cost_range):
    if(cost_range =='<20'):
        return 20
    lower, upper = map(int, cost_range.split('-'))
    return (lower + upper) / 2

def prelim_graphs(reddit_data):
    # gender distribution pie chart
    gender_counts = reddit_data['Gender'].value_counts()
    plt.pie(gender_counts,labels=gender_counts.index)
    plt.title("Gender Distribution")
    plt.axis('equal')
    plt.show()

    # education levels countplot
    sns.countplot(x='Education',reddit_data=reddit_data)
    plt.title("Education Levels")
    plt.show()

    # age countplot
    sns.countplot(x='Age',reddit_data=reddit_data)
    plt.title("Age")
    plt.show()
    
    # industry of work by gender
    gender_industry = pd.crosstab(reddit_data['Industry'],reddit_data['Gender'])
    gender_industry.plot(kind='bar', stacked=True, colormap='Set2')
    plt.title("Gender Distribution by Industry")
    plt.ylabel('Count')
    plt.show()

    # education by gender countplot
    sns.countplot(x='Education', hue='Gender', reddit_data=reddit_data)
    plt.title('Education by Gender')
    plt.xticks(rotation=45)
    plt.show()

    # cost of living by industry worked in
    g = sns.FacetGrid(reddit_data, col='Industry', col_wrap=4, height=3)
    g.map(sns.histplot, 'Cost of Living')
    g.set_axis_labels('Cost of Living', 'Count')
    plt.show()

    # heat map of age and cost of living
    reddit_data['Age Middle'] = reddit_data['Age'].apply(extract_midpoint)
    reddit_data['COL Middle'] = reddit_data['Cost of Living'].apply(extract_midpoint)
    corr = reddit_data[['Age Middle', 'COL Middle']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.show()
