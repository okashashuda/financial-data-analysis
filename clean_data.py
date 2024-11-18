import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from skimage.color import rgb2lab
from sklearn.ensemble import RandomForestClassifier


# IMPORT DATA
data = pd.read_csv("cmpt353dataproject/data/financedata.csv")


# CLEAN DATA
# keep only columns that are relevant to our use case
data = data[["What is your race / ethnicity? Select all that apply.",
             "What is your gender?",
             "What is your age? ",
             "What is your relationship status? ",
             "Which of the following best describes the industry in which you currently or most recently worked? ",
             "What is the highest level of education you have completed?",
             "Using this cost of living index, looking at the third column of the table / first column of numbers (with the title Cost of Living Index) what is the cost of living in your location? If your location is not listed, use the closest approximation in your opinion. ",
             "Annual Expenses",
             "Annual Income"]]

# rename columns for ease of use
data = data.rename(columns={"What is your race / ethnicity? Select all that apply.": "Ethnicity",
                            "What is your gender?": "Gender",
                            "What is your age? ": "Age",
                            "What is your relationship status? ": "Relationship Status",
                            "Which of the following best describes the industry in which you currently or most recently worked? ": "Industry",
                            "What is the highest level of education you have completed?": "Education",
                            "Using this cost of living index, looking at the third column of the table / first column of numbers (with the title Cost of Living Index) what is the cost of living in your location? If your location is not listed, use the closest approximation in your opinion. ": "Cost of Living"})

# keep only columns where 'annual expenses' or 'annual income' are NOT zero
data = data[(data != 0).all(axis=1)]
data = data.dropna()

# convert int to float, for consistency
data["Annual Income"] = data["Annual Income"].astype("float")

# orders both of the ordered categorical columns, to provide brevity in graphs
data['Age'] = pd.Categorical(data['Age'],
                             categories=['<20','21-25','26-30','31-35','36-40','41-45','46-50','51-55','56-60','61-65','66-70'],
                             ordered=True)

relationship = ['Married', 'In a relationship, but not married', 'Divorced - In a relationship, but not married', 'Divorced - Remarried', 'Widowed - In a relationship, but not married', 'Widowed - Remarried']
single = ['Single, never married', 'Divorced', 'Widowed']

# function to categorize relationship status
def categorize_status(status):
    if status in relationship:
        return "relationship"
    elif status in single:
        return "single"
    return status

data["Relationship Status"] = data["Relationship Status"].apply(categorize_status)

data['Cost of Living'] = pd.Categorical(data['Cost of Living'],
                                        categories=['1-20','21-30','31-40','41-50','51-60','61-70','71-80','81-90','91-100','101-110','111-120','121-130','131-140','141-150'],
                                        ordered=True)

# puts the cleaned data in a new csv
data.to_csv("cmpt353dataproject/data/clean_data.csv", index=True)

def extract_midpoint(cost_range):
    if(cost_range =='<20'):
        return 20
    lower, upper = map(int, cost_range.split('-'))
    return (lower + upper) / 2

def prelim_graphs():
    # gender distribution pie chart
    gender_counts = data['Gender'].value_counts()
    plt.pie(gender_counts,labels=gender_counts.index)
    plt.title("Gender Distribution")
    plt.axis('equal')
    plt.show()

    # education levels countplot
    sns.countplot(x='Education',data=data)
    plt.title("Education Levels")
    plt.show()

    # age countplot
    sns.countplot(x='Age',data=data)
    plt.title("Age")
    plt.show()
    
    # industry of work by gender
    gender_industry = pd.crosstab(data['Industry'],data['Gender'])
    gender_industry.plot(kind='bar', stacked=True, colormap='Set2')
    plt.title("Gender Distribution by Industry")
    plt.ylabel('Count')
    plt.show()

    # education by gender countplot
    sns.countplot(x='Education', hue='Gender', data=data)
    plt.title('Education by Gender')
    plt.xticks(rotation=45)
    plt.show()

    # cost of living by industry worked in
    g = sns.FacetGrid(data, col='Industry', col_wrap=4, height=3)
    g.map(sns.histplot, 'Cost of Living')
    g.set_axis_labels('Cost of Living', 'Count')
    plt.show()

    # heat map of age and cost of living
    data['Age Middle'] = data['Age'].apply(extract_midpoint)
    data['COL Middle'] = data['Cost of Living'].apply(extract_midpoint)
    corr = data[['Age Middle', 'COL Middle']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.show()
