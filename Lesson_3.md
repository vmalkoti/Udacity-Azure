# Microsoft Scholarship Foundation course - Udacity 
[URL to the course](https://classroom.udacity.com/nanodegrees/nd00332)

## Lesson 3: Model Training

*** 

### Data Import and Transformation
Clean missing data [reference](https://docs.microsoft.com/en-us/azure/machine-learning/algorithm-module-reference/clean-missing-data).

Data Workflow [Reference](https://docs.microsoft.com/en-us/azure/machine-learning/concept-data)

* **Datastore** is an abstraction of the source of data.
  * It is the actual storage place of the data - the place to which a service will connect to create a dataset for a pipeline.
  * A datastore can be a database, set of files, or collection of data in any storage format
  * An Azure datastore contains information about this connection so that pipelines use this abstract object instead of storing connection information. The machine learning pipeline doesn't need to know where data comes from, how to connect to it or what's the form of data storage.
* **Dataset** is a pointer to a subset of the data in the datastore.
  * A dataset in Azure is the object that pipelines use to read the data.
  * A similar concept is a dataframe in pandas, or an SQL query resultset in relational databases.
  * A dataset can be created from a datastore or using files (local/remote)
  * In Azure: 
    * a dataset is not a copy of the data. Instead Azure datasets are pointers or references to the original data.
    * two types of datasets are supported - Tabular dataset and Web URL (file) dataset.
    * modifying or deleting underlying original file will invalidate dataset.

#### Lab 2

Datasets: [crime-dirty](https://introtomlsampledata.blob.core.windows.net/data/crime-data/crime-dirty.csv), [crime-spring](https://introtomlsampledata.blob.core.windows.net/data/crime-data/crime-spring.csv), [crime-winter](://introtomlsampledata.blob.core.windows.net/data/crime-data/crime-winter.csv)

***Note:*** The UI components (buttons, labels, etc) in the lab may change.

***

### Managing data
**Drift**: degradation of model predictions over time. Types of reason for that degradation:
* **Concept Drift**: Change in the definition or properties of the _target variable_.
* **Data Drift**: Change in the *feature set* due to changing statistical properties, errors in data capturing or change in the data.

References:
[Detect Drift On Datasets](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-monitor-datasets)
[Understanding and Handling Data and Concept Drift](https://www.explorium.ai/blog/understanding-and-handling-data-and-concept-drift/)

***

#### Lab 3

Datasets: [version 1](https://introtomlsampledata.blob.core.windows.net/data/nyc-taxi/nyc-taxi-sample-data-5months.csv), [version 2](https://introtomlsampledata.blob.core.windows.net/data/nyc-taxi/nyc-taxi-sample-data-5months.csv)

***

### Features
A _feature_ is a characteristic or attribute of the observation. Usually in a dataset, each row is an observation and columns represent features.

_Feature engineering_ is the process of creating new features in the dataset when original set of features is not adequate to train the model. 

Why or when would we need Feature Engineering?

Benefits of Feature Engineering?

Feature Engineering tasks:
* Aggregation
* Part-of
* Binning
* Flagging
* Frequency-based
* Embedding or Feature Learning
* Deriving by example

_Feature selection_ is the process of selecting the most important and/or most relevant features. 

Reasons we need feature selection:
* Eliminate irrelevant, redundant or (highly) correlated features
* To improve performance of algorithms that cannot handle a large number of features



_Dimensionality reduction_ is the process of reduce number of features when datasets contain a very large number of features. This helps improve performance of the machine learning algorithm. Commonly used techniques are:
* PCA (Principal Component Analysis)
* t-SNE (t-distribution Stochastic Neighboring Entities)
* Feature embedding

***

#### Lab: Engineer and Select Features

Datasets: [bike rental hourly](https://introtomlsampledata.blob.core.windows.net/data/bike-rental/bike-rental-hour.csv)

TODO: Find difference between Regular Expression and Relative Expression in Split Data module in Azure

Steps:
1. Create dataset - Bike rental daily
2. Create new pipeline in designer
3. Select compute target
4. Add dataset that we had created - Bike rental daily
5. Edit metadata to change type of columns "season" and "weathersit" to categorical
6. Exclude columns "instant", "dteday", "casual", "registered" 
7. Create two copies of the dataset
    1. With original features
        1. Use columns from data source without any changes or additions
    2. With engineered features
        1. Python script creates 12 columns storing rentals in past 12 hours (code below)
8. Take both datasets and split them into train/test sets with condition "yr"==0
9. Exclude column "yr" from all of these datasets
10. Train both training datasets with Boosted Decision Tree Regression algorithm 
10. Score both of these train sets using test sets
11. Evaluate output of scores for comparison

**Code used in Python module:**
```python
for i in np.arange(1, 13):
    prev_col_name = 'cnt' if i == 1 else 'Rentals in hour -{}'.format(i-1)
    new_col_name = 'Rentals in hour -{}'.format(i)

    dataframe1[new_col_name] = dataframe1[prev_col_name].shift(1).fillna(0)
```

****

### Data Drift

**Causes**

**Monitoring**
Process of monitoring involves:
* Specifying baseline dataset
* Specifying target dataset
* Comparing these two datasets

Different types of comparison:
* Comparing input data vs training data
* Comparing different samples of time-series data

****

### Model Training Basics

Data Science Process Steps:
1. Collect data
2. Prepare Data
3. Train model
4. Evaluate model
5. Deploy model

The goal of the training process is to determine the relationship between features (independent variables) and the target (dependent variable).

**Parameters** are the values that model learns during the training process, e.g. in a linear model, the feature weights are the parameters. The training process tries to find the best set of parameters for the model.

**Hyperparameters** are the values that we provide to the model for training, like the learning rate, regularization parameter, depth of tree, k value in k-means and k-nearest-neighbors, etc. Hyperparameter tuning is running the model training process several times and selecting the values that produce the best results on validation data.

The input dataset for training is usually divided into three parts:
1. **Training set** is used to _fit_ the model, i.e. learn the parameters
2. **Validation set** or **Cross-Validation set** is then used to verify how the trained model performed with a set of hyperparameter values. Several runs using the training and validation set are used to select best values for hyperparameters.
3. **Test set** is the unseen data set that is finally used to evaluate the model trained with tuned hyperparameters.

****

### Classification

In classification problems, the algorithm learns to assign a label to the input observation. The expected output is a discrete value from a small set of values (two or more).

Types of classification problems:
1. Binary classification assigns a label out of two possible classes. Example - classify email as spam or not-spam
2. Multi class classification assigns a label out of multiple classes. Example - classify input character image as digits 0,1,...9
3. Multi label classification assigns multiple labels. Example - assign relevant tags to a news article

****

### Regression

In regression problems, the algorithm learns to predict a numerical value for an input observation. The expected output is a continuous value not limited to any specific set.

Types of regression problems:
1. Predict arbitratry values. Example - predict house price
2. Predict values between 0 and 1. Example - probability that a patient has a disease

***

### Evaluating Model Performance - Classification

A model is ready to be deployed in production when it performs well on and test data set.

#### Confusion Matrices 

[Confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix) or *error matrix* is a table layout that allows the visualization of algorithm's output summary to help in evaluation.

|  |  | Actual |  |
| --- | --- | --- | --- |
|  |  | True   | False  |
| **Predicted** | True  | True Positive (TP)  | False Negative (FN) |
|    | False | False Positive (FP) | True Negative (TN) |

**Accuracy** is ratio of total correct prediction to all predictions made.

![accuracy](https://latex.codecogs.com/gif.latex?Accuracy&space;=&space;\frac{TP&space;&plus;&space;TN}{TP&space;&plus;&space;FP&space;&plus;&space;TN&space;&plus;&space;FN})

**Precision** is ratio of correct positive predictions to all predicted positives.

![precision](https://latex.codecogs.com/gif.latex?Precision&space;=&space;\frac{TP}{TP&space;&plus;&space;FP})

**Recall** is ratio of correct positive predictions to all actual positives

![recall](https://latex.codecogs.com/gif.latex?Recall&space;=&space;\frac{TP}{TP&space;&plus;&space;FN})

F1 Score

![F1](https://latex.codecogs.com/gif.latex?F1&space;Score&space;=&space;2&space;*&space;\frac{Precision&space;*&space;Recall}{Precision&space;&plus;&space;Recall})

#### Visual ways of model perfomance
* ROC (Receiver Operating Characteristics) is ratio of the true positives rate to the false positives rate
* AUC (Area Under the Curve) is the area under the ROC curve
* Gain and Life

***

### Evaluating Model Performance - Regression
 
 * Root Mean Squared Error (RMSE))
 * Mean Absolute Error (MAE)
 * R-squared
 * Spearman Correlation 

#### Visual ways of evaluation
 * Predicted vs True chart
 * Histogram of Residuals
 
***

 #### Lab: Train and Evaluate a Model

 Dataset: [flightdelays](https://introtomlsampledata.blob.core.windows.net/data/flightdelays/flightdelays.csv)

Steps:
1. Open Datasets tab and create a new dataset with flightdelays data
2. Open Deisgner tab
3. Select compute target
4. Add flightdelays dataset
5. Add Split Data module with mode="Relative Expression" and expression="\Month"<10
6. Add Select Columns in Dataset module and exclude columns: Month, Year, Year_R, Timezone, Timezone_R
7. Add Two-Class Logistic Regression module 
8. Add train module. 
    1. Set Label Column="ArrDel15"
    2. Connect first input to Logistic Regression
    3. Connect second input to Select Columns module
9. Add Score Model module
    1. Connect first input to Train Model 
    2. Connect second input to Split Data module
10. Add Evaluate Model module and connect first input to Score Model module
11. Click Submit to create experiment and run the pipeline

***

### Ensemble Learning

Ensembles are combination of several different machine algorithms.

Types:
* Bagging or Bootstrap aggregation
    * Reduce variance (overfitting)
    * A set of models are trained on samples from the data set
* Boosting
    * Reduce bias (underfitting)
    * A sequence of models added iteratively, each improving output of one before it
* Stacking
    * Hierarchichal layers of models, similar to a neural network

### Automated Machine Learning

***

#### Lab: Train a Two-Class Boosted Decision Tree

Dataset: [flightdelays](https://introtomlsampledata.blob.core.windows.net/data/flightdelays/flightdelays.csv)

Steps:
1. Open Datasets tab and create a new dataset with flightdelays data
2. Open Deisgner tab
3. Select compute target
4. Add flightdelays dataset
5. Add Split Data module with mode="Relative Expression" and expression="\Month"<10
6. Add Select Columns in Dataset module and exclude columns: Month, Year, Year_R, Timezone, Timezone_R
7. Add Two-Class Boosted Decision Tree module 
8. Add train module. 
    1. Set Label Column="ArrDel15"
    2. Connect first input to Logistic Regression
    3. Connect second input to Select Columns module
9. Add Score Model module
    1. Connect first input to Train Model 
    2. Connect second input to Split Data module
10. Add Evaluate Model module and connect first input to Score Model module
11. Click Submit to create experiment and run the pipeline
12. After model run is complete, go to Output tab of Evaluate Model and review visualizations

***

#### Lab: Train a Simple Classifier with Automated ML

Dataset: [flightdelays](https://introtomlsampledata.blob.core.windows.net/data/flightdelays/flightdelays.csv)

Steps:

1. Open Datasets tab and create a new dataset with flightdelays data
    1. Skip columns: Path, Month, Year, Timezone, Year_R, Timezone_R
2. Open Home tab and create a new Automated ML Run
3. Select flightdelays dataset 
4. Create new experiment with target column ArrDel15
5. Select task type: Classification
6. Select AUC as primary metric with 1 hour time and 0.7 threshold
7. After model run is finished, view model information in Model tab and metrics in Metrics tab
