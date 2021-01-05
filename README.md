# Disaster Response Pipeline Project
Repository for Udacity's Data Scientist Nanodegree - Project 2

- - - -
![alt text](https://github.com/Andrea-Schulz/datascience/blob/master/icons/notamused1.png?raw=true)
- - - -

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

- - - -
## Installation <a name="installation"></a>

* Python 3.6.x
* Jupyter Notebook - as a useful tool to improve readability of the Jupyter Notebooks I recommend the [Unofficial Jupyter Extensions](https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/index.html)

## Project Motivation<a name="motivation"></a>

...becoming a Data Scientist, duh. The Data Scientist Nanodegree course contains 4 projects, with the questions of interest differing for each one:

The second project is concerned with ETL and Machine Learning Pipelines, with the main goal to process categorize relevant communication (text, social media etc.) to be used by disaster response professionals. Apart from the pipeline scripts, this project includes a web app with data visualizations, where an emergency worker can input a new message and get classification results in several categories.

Feel free to check out the Jupyter Notebooks as well, which I used to practice and prepare the final implementation.

## File Descriptions <a name="files"></a>

The project contains 3 main components:
* **ETL Pipeline**: the data cleaning pipeline `process_data.py` loads, merges and cleans the datasets with pandas, and then puts it in a SQLite database
* **Machine Learning Pipeline**: the machine learning pipeline `train_classifier.py` processes the data from the database using NPL techniques, trains and tunes a model using GridSearchCV and outputs the classification results
* **Flask Web App** for real-time data visualization and input message classification

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Shout out to Figure Eight for providing their pre-sorted disaster response messages data.



