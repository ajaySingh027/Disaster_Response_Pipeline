# Disaster Response Pipeline Project

### Project Description
In this project we have to analyze real messages (ie from social media) sent during disaster or emergency of any need. Then classifying these messages in a number of different categories including whether the message is related to the disaster or any other group.
with the analyzing and cleaning of the data from two different datasets first pipeline known as ETL (Extract Transform Load) is created. And output of which is then used in training machine learning model on data.

### Libraries used
- scipy and numpy: SciPy and Numpy are free and open-source Python library used for scientific computing and technical computing.

- pandas: Pandas is a software library written for the Python programming language for data manipulation and analysis.

- sklearn: Machine learning library for the Python programming language. It features for classification and regression.

- sqlalchemy: Used for saving data in sqlite database

- matplotlib: Matplotlib is a plotting library for the Python programming language and its numerical mathematics extension NumPy


### Files in Repository
 The file structure of the project:

- app
 - template
    - master.html  # main page of web app
    - go.html  # classification result page of web app
 - run.py  # Flask file that runs app

- data
    - disaster_categories.csv  # data to process 
    - disaster_messages.csv  # data to process
    - process_data.py
    - InsertDatabaseName.db   # database to save clean data to

- models
    - train_classifier.py
    - classifier.pkl  # saved model 

- README.md


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
