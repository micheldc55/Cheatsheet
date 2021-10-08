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



# Working with GeoPandas

### Imports

```
import geopandas as gpd

# For connecting to a PostgreSQL Database (this is only an example)
import dbconnection
```

#### Importing coordinates from SQL into GeoPandas

```
# Establish database connection
cur, conn = dbconnection.connect('database initialization file or ', 'section within the database.ini file')

sql_query = """
SELECT * FROM ...
"""

df = gpd.GeoDataFrame.from_postgis(sql_query, conn, geom_col='coordinates/geom_df_column')

conn.close()
```

**Example:** Using an iteration process to extract a GeoPandas DataFrame from a SQL query and concatenate it with others in a for loop. Notice that the gpd.GeoDataFrme.from_postgis() method doesn't allow the use of in iterable in the geom_col argument and the query returns a df with two geometry columns ('city_center_location' and 'admin_area_location'), so we are performing two queries and generating two df's. Finally we create a new column with the other geometry-type column and paste it in the df we want to use.


```
localities_country_geo_df = pd.DataFrame()
countries_missing_geo_df  = []

cur, conn = dbconnection.connect('../src/database.ini', 'lam-mea-nam-oce-sea-mnr')

sql_query = """
select 
aa.feat_id as admin_area_id,
aa.country_code_char3,
aa.feat_type,
ad1."name" as admin_level_1,
aa."name" as admin_area_name,
aa.feat_area,
mc.feat_id as city_center_id,
mc."name" as city_center_name,
mc.admin_class,
mc.display_class,
ma.value_integer as population,
mc.geom as city_center_location,
aa.geom as admin_area_location

from "_2021_09_007_{region}_{country}_{country}".mnr_citycenter as mc

join "_2021_09_007_{region}_{country}_{country}".mnr_admin_area as aa on mc.feat_id = aa.citycenter_id

join "_2021_09_007_{region}_{country}_{country}".mnr_admin_area2attribute as a2a on aa.feat_id = a2a.admin_area_id

join "_2021_09_007_{region}_{country}_{country}".mnr_attribute as ma on a2a.attribute_id = ma.attribute_id

left join (select * from "_2021_09_007_{region}_{country}_{country}".mnr_admin_area where feat_type=1112) ad1 on aa.a1_admin_id = ad1.a1_admin_id

where aa.feat_type = 1119
and ma.attribute_type = 'PO'
--and mc.name <> aa.name
"""

for region, country in countries_to_query:
    query_country = sql_query.format(country=country, region=region)
    try:
        df     = gpd.GeoDataFrame.from_postgis(query_country, conn, geom_col='city_center_location')
        df_aux = gpd.GeoDataFrame.from_postgis(query_country, conn, geom_col='admin_area_location')
        df['admin_area_location'] = df_aux['admin_area_location']
        
        localities_country_geo_df = pd.concat([localities_country_geo_df, df])

        print(f'{country} done')
    except:
        countries_missing.append((region, country))
        print(f'{country} not done')
conn.close()
```
