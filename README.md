# Financial Data Analysis and Visualization Project
This repository contains the final project for CMPT 353 by Stefan Pricope and Okasha Huda.

Our project analyzes financial survey data collected from Reddit over multiple years (2016–2023) to uncover trends in income, expenses, and savings rates. It further explores the impact of the COVID-19 pandemic on individual financial behaviors by comparing pre-pandemic and post-pandemic datasets.

## Key Features
1. Data Cleaning and Preprocessing:
- Consolidated data from multiple annual datasets, ensuring consistent columns across years
- Filtered out missing values, zero entries, and outliers for robust analysis
- Standardized financial categories to maintain compatibility between datasets with varying formats

2. Financial Analysis:
- Segmented data into pre-pandemic (2016–2019) and pandemic (2020–2023) periods to evaluate financial trends
- Performed statistical tests (t-tests and Mann-Whitney U tests) to identify significant changes in income and expense distributions
- Calculated and visualized Compound Annual Growth Rates (CAGR) for income and expenses

3. Visualization:
- Created dynamic plots to illustrate trends in income, expenses, and savings rates across years
- Plotted year-over-year growth rates for individual expense categories (e.g., housing, healthcare, education)

## Setup
### Prerequisites
Ensure the following tools and libraries are installed:
- Git
- Python (3.8+)
- Python Libraries:
    - `pandas`
    - `matplotlib`
    - `seaborn`
    - `scipy`
    - `statsmodels`
    - `sklearn`

### Data
The cleaned datasets are included in the `data/` folder. In case of any issues, download the data from the following zip folder:
[Reddit Financial Survey Data](https://drive.google.com/file/d/1xKDLVbc12-XuYQASu1qNxYMV6h2RZmyH/view)

## How to Run
1. Data Cleaning

Run `clean_data.py` to clean and preprocess the raw datasets:
```
python clean_data.py
```
This script:
- Cleans datasets for consistency
- Exports cleaned data to the `data/` folder
- Generates exploratory graphs, including correlations and category distributions

2. Analysis and Visualization

Run `analysis.py` to perform financial trend analysis and visualization:
```
python analysis.py
```
This script:
- Segments the data into pre-pandemic and pandemic periods
- Performs statistical analysis (t-tests and Mann-Whitney U-tests) on income and expense changes
- Generates plots for income trends, savings rates, and expense growth rates
- Validates a predictive model for annual income based on expenses

## Authors

* **Stefan Pricope** - [GitHub](https://github.com/scp10sfu)
* **Okasha Huda** - [GitHub](https://github.com/okashashuda) - [Personal Website](http://okashahuda.com/)

## Acknowledgments
- Data Source: [Reddit Financial Survey](https://www.reddit.com/r/financialindependence/comments/1cl177n/the_official_2023_survey_results_are_here/)
- README Template: [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2#file-readme-template-md)
