import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import sklearn


# IMPORT DATA
data = pd.read_csv("data/financedata.csv")


# CLEAN DATA
# keep only columns that are relevant to our use case
data = data[["What is your race / ethnicity? Select all that apply.",
             "What is your gender?",
             "What is your age? ",
             "Which of the following best describes the industry in which you currently or most recently worked? ",
             "What is the highest level of education you have completed?",
             "Using this cost of living index, looking at the third column of the table / first column of numbers (with the title Cost of Living Index) what is the cost of living in your location? If your location is not listed, use the closest approximation in your opinion. ",
             "Annual Expenses",
             "Annual Income"]]

# rename columns for ease of use
data = data.rename(columns={"What is your race / ethnicity? Select all that apply.": "Ethnicity",
                            "What is your gender?": "Gender",
                            "What is your age? ": "Age",
                            "Which of the following best describes the industry in which you currently or most recently worked? ": "Industry",
                            "What is the highest level of education you have completed?": "Education",
                            "Using this cost of living index, looking at the third column of the table / first column of numbers (with the title Cost of Living Index) what is the cost of living in your location? If your location is not listed, use the closest approximation in your opinion. ": "Cost of Living"})

# keep only columns where 'annual expenses' or 'annual income' are NOT zero
data = data[(data != 0).all(axis=1)]
data = data.dropna()
# convert int to float, for consistency
data["Annual Income"] = data["Annual Income"].astype("float")
#Orders both of the ordered categorical columns, to provide brevity in graphs.
data['Age'] = pd.Categorical(data['Age'],categories=['<20','21-25','26-30','31-35','36-40','41-45','46-50','51-55','56-60','61-65','66-70'],ordered=True)
data['Cost of Living'] = pd.Categorical(data['Cost of Living'],categories=['1-20','21-30','31-40','41-50','51-60','61-70','71-80','81-90','91-100','101-110','111-120','121-130','131-140','141-150'],ordered=True)
#Puts the cleaned data in a new csv.
data.to_csv("data/clean_data.csv", index=True)


# gender_counts = data['Gender'].value_counts()
# plt.pie(gender_counts,labels=gender_counts.index)
# plt.title("Gender Distribution")
# plt.axis('equal')
# plt.show()

# sns.countplot(x='Education',data=data)
# plt.title("Education Levels")
# plt.show()


sns.countplot(x='Age',data=data)
plt.title("Age")
plt.show()

# gender_industry = pd.crosstab(data['Industry'],data['Gender'])
# gender_industry.plot(kind='bar', stacked=True, colormap='Set2')
# plt.title("Gender Distribution by Industry")
# plt.ylabel('Count')
# plt.show()

# sns.countplot(x='Education', hue='Gender', data=data)
# plt.title('Education by Gender')
# plt.xticks(rotation=45)
# plt.show()


# g = sns.FacetGrid(data, col='Industry', col_wrap=4, height=3)
# g.map(sns.histplot, 'Cost of Living')
# g.set_axis_labels('Cost of Living', 'Count')
# plt.show()


def extract_midpoint(cost_range):
    if(cost_range =='<20'):
        return 20
    lower, upper = map(int, cost_range.split('-'))
    return (lower + upper) / 2

data['Age Middle'] = data['Age'].apply(extract_midpoint)
data['COL Middle'] = data['Cost of Living'].apply(extract_midpoint)
corr = data[['Age Middle', 'COL Middle']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()