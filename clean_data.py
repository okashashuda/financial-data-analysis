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

# convert int to float, for consistency
data["Annual Income"] = data["Annual Income"].astype("float")

data.to_csv("data/clean_data.csv", index=True)

