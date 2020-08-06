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


#### Lab 4

Datasets: [bike rental hourly](https://introtomlsampledata.blob.core.windows.net/data/bike-rental/bike-rental-hour.csv)

TODO: Find difference between Regular Expression and Relative Expression in Split Data module in Azure

Code used in Python module:
```python
for i in np.arange(1, 13):
        prev_col_name = 'cnt' if i == 1 else 'Rentals in hour -{}'.format(i-1)
        new_col_name = 'Rentals in hour -{}'.format(i)

        dataframe1[new_col_name] = dataframe1[prev_col_name].shift(1).fillna(0)
```
