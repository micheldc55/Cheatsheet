# Python and Machine Learning Cheat sheet
Python and Machine Learning cheat sheet for Data Science projects

## Missing values

### Visualizing missing values:

#### Numerically:

Get the missing values by feature using the .isna() method on the DataFrame

```python
df.isna().sum()
```

#### Visually:

1) Get the missing values by using the **Seaborn** library:

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Depending on the number features, the heatmap may get small so you can add:
plt.figure(figsize=(10,10))

# Plotting the missing values heatmap
sns.heatmap(data=df.isna(), cbar=False, vmax=1, vmin=0, yticklabels=False)
```

2) Get the missing values from the **missingno** library; missigno is a library dedicated to the visualization of null values. It has to be imported separately.

```python
import missingno as msno

# missingno matrix returns a heatmap with the locations of missing values in the DataFrame
msno.matrix(df)

# missingo heatmap returns a correlation map of the relationship between missing values in each feature. 
# 1 means all missing values between both features happen on the same instance
msno.heatmap(df)
```
