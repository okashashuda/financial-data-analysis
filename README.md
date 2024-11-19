# Predicting Relationship status from various data from Reddit Survey.

This is a repository storing the final project for CMPT 353 by Stefan Pricope and Okasha Huda.

Our project takes reddit financial survey data and cleans up in order to analyze how accurately we can predict the relationship status of the submissions by various factors in our data. We wanted to try predicting a slightly different factor than just predicting income. 

The main factors we used are:
categorical = ["Ethnicity", "Gender", "Age", "Industry", "Education", "Cost of Living"]
numerical = ["Annual Income", "Annual Expenses"]

## Setup

These are necessary tools to run our program:
- Git
- Python (3.8+) and these libraries:
    - pandas
    - numpy
    - matplotlib
    - seaborn
    - sklearn
- In case of data errors, download CSV from here: https://docs.google.com/spreadsheets/d/1RtUK46jQ1C5MwVacKZiBxQFlEttFt9WT/edit?usp=sharing&ouid=100454923916141027946&rtpof=true&sd=true

## Running our program

Our data is all included in the data folder already, therefore there is no need to download any data from external sources. In case errors occur please download the data from the link in setup.

 Otherwise we have 2 files which are ran seperately with each of their purposes being slightly different:
 - clean_data.py is used to clean our data and show some graphs giving information on the data. 
 - mlmodels.py is used to run various machine learning models on the data and also provides graphs showing the accuracy of the models with a visual.

 Both programs can simply just be ran without any extra requirements. clean_data.py must be ran before mlmodels.py atleast once, as it generates the clean datafile.


## Authors

* **Stefan Pricope** - [Github](https://github.com/scp10sfu)
* **Okasha Huda** - [GitHub](https://github.com/okashashuda) - [Personal Website](http://okashahuda.com/)

## Acknowledgments

* https://gist.github.com/PurpleBooth/109311bb0361f32d87a2#file-readme-template-md : Credit for README template.
* Data from reddit financial survey 2024

