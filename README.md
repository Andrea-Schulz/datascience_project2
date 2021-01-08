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

* Python 3.6.x with the following packages:
	* [Flask](https://flask.palletsprojects.com/en/1.1.x/)
	* [Numpy](https://numpy.org/)
	* [Pandas](https://pandas.pydata.org/)
	* [NLTK](http://www.nltk.org/index.html)
	* [Scikit-Learn](https://scikit-learn.org/stable/index.html)
	* [SQLalchemy](https://www.sqlalchemy.org/)
* Jupyter Notebook - as a useful tool to improve readability of the Jupyter Notebooks I recommend the [Unofficial Jupyter Extensions](https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/index.html)

## Project Motivation<a name="motivation"></a>

...becoming a Data Scientist, duh. The Data Scientist Nanodegree course contains 4 projects, with the questions of interest differing for each one:

The second project is concerned with ETL and Machine Learning Pipelines, with the main goal to process and categorize relevant communication (text, social media etc.) to be used by disaster response professionals.
Apart from the respective data processing and ML scripts, this project includes a web app with data visualizations, where an emergency worker can input a new message and classify it to assess its relevance.

Feel free to check out the Jupyter Notebooks as well, which I used to practice and prepare the final implementation.

## File Descriptions <a name="files"></a>

The project contains 3 main components:
* **ETL Pipeline**: the data cleaning pipeline `process_data.py` in the `data` directory...
	* loads, merges and cleans the raw datasets `disaster_categories.csv` and `disaster_categories.csv` with pandas
	* saves the cleaned data in a SQLite database
* **Machine Learning Pipeline**: the machine learning pipeline `train_classifier.py` in the `models` directory...
	* processes the data from the database using NPL techniques
	* trains and tunes a model using GridSearchCV
	* outputs the classification results
	* saves the model as a pickle `.pkl`
* **Flask Web App** in the `app` directory...
	* provides a web application with real-time input message classification and visualizations of the datasets used in the pipeline
	* includes `run.py` script to load the data and model and run the web app locally


### Usage:
1. Run the following commands in the project's root directory to set up your database and model:
	* **ETL Pipeline** - to run data processing on disaster response message data, execute: `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
	* **ML Pipeline** - to define and train model on the given dataset and save the model as a pickle file, execute: `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
		* GridSearchCV can be used by setting 'grid=True' in `build_model` and `evaluate_model` functions
		* the basic script functionality can be tested on a smaller sample of the dataset by setting 'reduced_dataset=True' when calling the `load_data` function
2. Run the following command in the app's directory to run your web app.
    `python run.py`
3. Go to localhost:3001 on your web browser to use the web app
4. Provide a sample text message in the input field at the top of the page and use the Machine Learning pipeline to classify it as shown below. The web app will highlight the matched categories in green:

![alt text](https://github.com/Andrea-Schulz/datascience_project2/blob/master/screenshots/message_example.jpg?raw=true)

## Results <a name="results"></a>

### Some Brief Thoughts on the Dataset...

The largest share of messages comes from the news or finds its way to emergency response organizations via direct channels:
![alt text](https://github.com/Andrea-Schulz/datascience_project2/blob/master/screenshots/genres.jpg?raw=true)

Almost 4 our of 5 messages in the given dataset is classified as "related" and thus relevant for emergeny response workers in general:
![alt text](https://github.com/Andrea-Schulz/datascience_project2/blob/master/screenshots/classifications.jpg?raw=true)

Around 23% of the messages are not assigned to any category and thus not relevant for emergency response:
![alt text](https://github.com/Andrea-Schulz/datascience_project2/blob/master/screenshots/category_number.jpg?raw=true)

### ... and on the Message Classification

As seen above, the given dataset is very imbalanced, i.e. the classes are represented inequally in the data.
For some categories the share of messages assigned to this category is very large or small in relation to the dataset as a whole, with some categories (i.e. "child_alone") not having a single message matched to it.
Hence, the the accuracy of the machine learning algorithm becomes distorted, and it is very likely that the classification works better for some message categories than for others.

Check out this [blog](https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/) to learn more about how to tackle imbalanced data in machine learning.

![alt text](https://github.com/Andrea-Schulz/datascience_project2/blob/master/screenshots/input_message.jpg?raw=true)

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Shout out to Figure Eight for providing their pre-processed and classified disaster response messages data [here](https://appen.com/datasets/combined-disaster-response-data/).



