# Python and Machine Learning Cheat sheet

Python and Machine Learning cheat sheet for Data Science projects. In this cheat sheet we explore the different aspects of a DS project in the order they are generally taken. First, missing values and duplicates, then combining information into one DataFrame. After that, we go over Exploratory Data Analysis (EDA); descriptive statistics, univariate, bivariate and multivariate analysis. Feature engineering and data tranformations, oversampling/undersampling and alternative methods. Splitting data and cross-validation. Machine Learning models and hiperparameter tunning. Metrics for model evaluation.

# Imports

### General imports

There are some imports that will be needed most of the time, which are pandas and numpy for working with DataFrames and arrays, seaborn and matplotlib.pyplot for visualizations and the %matplotlib inline command for the plots to be shown every time a command line is executed. They are summarized below in a single command line:

```python
import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
import seabron as        sns
%matplotlib inline
```

### Specific imports

There are some imports that aren't always done, but you will need them quite often. You may see them repeated throughout the notebook but they are used enough as to have them here for a quick reference.

#### Data Preprocessing

**Standard Scaler**
```python
from sklearn.preprocessing import StandardScaler
```

**MinMaxScaler**
```python
from sklearn.preprocessing import MinMaxScaler
```

**RobustScaler:** Sometimes you need a scaler that isn't sensible to outliers. Then you should use a scaler that doesn't use the mean, like the RobustScaler
```python
from sklearn.preprocessing import RobustScaler
```

#### Feature Encoding

**OneHotEncoder**
```python
from sklearn.preprocessing import OneHotEncoder
```

**OrdinalEncoder**
```python
from sklearn.preprocessing import OrdinalEncoder
```

**K-Bins Discretization:** Sometimes you need to turn a continuos feature into a discrete categorical feature. For example turn a continuos value like income into: '0-100', '100-200', '>100'. It returns the categorical values in numerical form to be implemented into an ML model.
```python
from sklearn.preprocessing import KBinsDiscretizer
```

# Missing values

## Visualizing missing values:

### Numerically:

#### Pandas .isna() method:

Get the missing values by feature using the .isna() method on the DataFrame

```python
df.isna().sum()
```

### Visually:

#### 1) Seaborn
Get the missing values by using the **Seaborn** library:

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Depending on the number features, the heatmap may get small so you can add:
plt.figure(figsize=(10,10))

# Plotting the missing values heatmap
sns.heatmap(data=df.isna(), cbar=False, vmax=1, vmin=0, yticklabels=False)
```

#### 2) missingno

Get the missing values from the **missingno** library; missigno is a library dedicated to the visualization of null values. It has to be imported separately.

```python
import missingno as msno

# missingno matrix returns a heatmap with the locations of missing values in the DataFrame
msno.matrix(df)

# missingo heatmap returns a correlation map of the relationship between missing values in each feature. 
# 1 means all missing values between both features happen on the same instance
msno.heatmap(df)
```

## Checking for duplicates (repeated rows in the dataset)

First of all, you should first determine whether or not the duplicated values are in the dataset because they actually represent two different instances, or if they are repeated because of an error. A common indicator for duplicated values that should be eliminated is equal dates. If this is not the case, then pandas method **.duplicated()** will return a DataFrame with all the duplicated information shown in a True/False DataFrame.

```python
df.duplicated() # Use the 'keep' argument to determine which duplicated value to show. Use 'subset' arg. to check for duplicated in parts of the df
```

## Operations on DataFrames

### DataFrame data extraction

Below we have the different operations that can be used to get back values from the DataFrame. You can use the .loc[] method to get back a value, series or pandas DataFrame from the original DataFrame. You can use the .iloc[] method to get back a value, series or DataFrame from the original one, searching by the index value.

```python

## Searching by column
df['column_name'] # Returns a series with the column values with the same index as the df
                  # If 'column_name' is a list, then df[] returns a new DataFrame

df.column_name    # Same as above, but column name has to follow Python variable name specifications

## Searching by row
df.loc['row_index name']  # Returns a series with the values of each feature for the row_index_name (row_index_name can be a list)

df.loc['row_index_name', 'column_name']  # Returns the value located in the DataFrame at row='row_index_name' ∩ column='column_name'

## Searching by index
df.iloc[index_number]  # Returns the value or series of values located at the DataFrame's index_number 
                       # if index_number is a list, df.iloc returns a new DataFrame

df.iloc[slice]         # Using slice notation returns a series or df of the sliced values
```

### Joining DataFrames

Below are the most commonly used methods to join DataFrames. They can be joined with the .concat() or with the pd.merge() functions. Generally speaking, the concat function is used for joining on rows and the merge is used for joining on columns. In the latter, you need a common column on which to join both DataFrames.

#### Concat method

Pandas .concat() function allows you to join two DataFrames df1 and df2 by sticking them one below the other. If df1 has 12 rows and df2 has 3 rows and you want to add the 3 rows of df2 below df1, you can use the **pd.concat()** method to obatin a new DataFrame with the 15 rows.

```python
pd.concat([df1, df2])  # Joins DataFrames df1 and df2 on the last row of df1 and the first row of df2
```
