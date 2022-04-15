---
layout: post
title: Visualizing Climate Change
---

In this blog, we will make visualization of climate change using historic temperature data from The Global 
Historical Climatology Network data set, compiled by the National Centers for Environmental Information of the 
US National Oceanic and Atmospheric Administration. We will be using the sqlite3, pandas, plotly express libraries. 

## First, we will import the relevant data from csv files. 


```python
import sqlite3
import pandas as pd
import plotly.express as px

#establish a connection
conn = sqlite3.connect("temps.db")

#read in the csv
df1 = pd.read_csv("temps_stacked.csv")

df2 = pd.read_csv("countries.csv")
#recode the columns to get rid of space
df2 = df2.rename(columns= {"FIPS 10-4": "code"})
df3 = pd.read_csv("station-metadata.csv")
```

Let's inspect the three files.


```python
df1.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Year</th>
      <th>Month</th>
      <th>Temp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>1</td>
      <td>-0.89</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>2</td>
      <td>2.36</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>3</td>
      <td>4.72</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>4</td>
      <td>7.73</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>5</td>
      <td>11.28</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>code</th>
      <th>ISO 3166</th>
      <th>Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AF</td>
      <td>AF</td>
      <td>Afghanistan</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AX</td>
      <td>-</td>
      <td>Akrotiri</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AL</td>
      <td>AL</td>
      <td>Albania</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AG</td>
      <td>DZ</td>
      <td>Algeria</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AQ</td>
      <td>AS</td>
      <td>American Samoa</td>
    </tr>
  </tbody>
</table>
</div>




```python
df3.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>LATITUDE</th>
      <th>LONGITUDE</th>
      <th>STNELEV</th>
      <th>NAME</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ACW00011604</td>
      <td>57.7667</td>
      <td>11.8667</td>
      <td>18.0</td>
      <td>SAVE</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AE000041196</td>
      <td>25.3330</td>
      <td>55.5170</td>
      <td>34.0</td>
      <td>SHARJAH_INTER_AIRP</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AEM00041184</td>
      <td>25.6170</td>
      <td>55.9330</td>
      <td>31.0</td>
      <td>RAS_AL_KHAIMAH_INTE</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AEM00041194</td>
      <td>25.2550</td>
      <td>55.3640</td>
      <td>10.4</td>
      <td>DUBAI_INTL</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AEM00041216</td>
      <td>24.4300</td>
      <td>54.4700</td>
      <td>3.0</td>
      <td>ABU_DHABI_BATEEN_AIR</td>
    </tr>
  </tbody>
</table>
</div>



## Next, we will convert the pandas dataframe to tables in a database.


```python
#convert to database
df1.to_sql("temperatures", conn, if_exists="replace", index=False)

df2.to_sql("countries", conn, if_exists="replace", index=False)

df3.to_sql("stations", conn, if_exists="replace", index=False)

#close the connection
conn.close()
```

    C:\Users\justi\anaconda3\lib\site-packages\pandas\core\generic.py:2872: UserWarning: The spaces in these column names will not be changed. In pandas versions < 0.14, spaces were converted to underscores.
      sql.to_sql(
    

## Now, let's write a function to output a country's yearly climate data.


```python
def query_climate_database(country, year_begin, year_end, month):
    """
    A function that takes four arguments and returns a Pandas dataframe by making a query
    """
    conn = sqlite3.connect("temps.db")
    
    #command line that gets the relevant info    
    cmd = f"SELECT S.name, S.latitude, S.longitude, C.Name, T.year, T.month, T.temp \
            FROM temperatures T \
            LEFT JOIN stations S ON T.id = S.id\
            LEFT JOIN countries C ON C.code = SUBSTRING(T.id, 1, 2)\
            WHERE C.Name='{country}' AND T.year >= {year_begin} \
            AND T.year<={year_end} AND T.month = {month}"
    
    #converts to Pandas dataframe
    df = pd.read_sql_query(cmd, conn)
    conn.close()
    return df
```

Let's test our function.


```python
query_climate_database(country = "India", 
                       year_begin = 1980, 
                       year_end = 2020,
                       month = 1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>NAME</th>
      <th>LATITUDE</th>
      <th>LONGITUDE</th>
      <th>Name</th>
      <th>Year</th>
      <th>Month</th>
      <th>Temp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1980</td>
      <td>1</td>
      <td>23.48</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1981</td>
      <td>1</td>
      <td>24.57</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1982</td>
      <td>1</td>
      <td>24.19</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1983</td>
      <td>1</td>
      <td>23.51</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1984</td>
      <td>1</td>
      <td>24.81</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3147</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1983</td>
      <td>1</td>
      <td>5.10</td>
    </tr>
    <tr>
      <th>3148</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1986</td>
      <td>1</td>
      <td>6.90</td>
    </tr>
    <tr>
      <th>3149</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1994</td>
      <td>1</td>
      <td>8.10</td>
    </tr>
    <tr>
      <th>3150</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1995</td>
      <td>1</td>
      <td>5.60</td>
    </tr>
    <tr>
      <th>3151</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1997</td>
      <td>1</td>
      <td>5.70</td>
    </tr>
  </tbody>
</table>
<p>3152 rows × 7 columns</p>
</div>



## We can then make our second function that visualizes climate change.

### We will answer this question : How does the average yearly change in temperature vary within a given country?

We will use the sklearn library to calculate the estimated yearly increase in temperature in every station.
To compute the avearage yearly increase, we'll first write a function to calculate the average change in temperature per year using linear regression.


```python
from sklearn.linear_model import LinearRegression

def coef(data_group):
    """
    A function that returns the slope of the best-fit linear regression line
    """
    X = data_group[["Year"]]
    y = data_group["Temp"]
    LR = LinearRegression()
    LR.fit(X, y)
    slope = LR.coef_[0]
    return slope
```


```python
def temperature_coefficient_plot(country, year_begin, year_end, month, min_obs, **kwargs):
    """
    This function takes in country, year_begin, year_end, and month, as the previous sql function does. It also
    takes in a minimum observation argument, which dictates the minimum amount of observation for any station's 
    data to be accepted. Anything less will be filtered out. It also accepts additional keywords arguments to 
    style the plot. 
    """
    #call the query function to get the dataframe
    df = query_climate_database(country, year_begin, year_end, month)
    
    #filters out any station not meeting the min_obs requirement
    df['len'] = df.groupby(["NAME"])["Temp"].transform(len)
    #print(df.head())
    df = df[df['len'] >= min_obs]
    
    #adds esitmated yearly increase column using our coef function
    coefs = df.groupby(["NAME", "LATITUDE", "LONGITUDE"]).apply(coef)
    coefs = coefs.reset_index()
    #print(coefs.head())
    coefs.rename(columns = {0:'Estimated Yearly Increase (°C)'}, inplace = True)
    
    #plots the data
    fig = px.scatter_mapbox(coefs, # data for the points you want to plot
                        lat = "LATITUDE", # column name for latitude informataion
                        lon = "LONGITUDE", # column name for longitude information
                        hover_name = "NAME", # what's the bold text that appears when you hover over
                        color = "Estimated Yearly Increase (°C)",
                        **kwargs) # map style
    return fig
```

Now let's test our function.


```python
from plotly.io import write_html
color_map = px.colors.diverging.RdGy_r # choose a colormap

fig = temperature_coefficient_plot("India", 1980, 2020, 1, 
                                   min_obs = 10,
                                   zoom = 2,
                                   mapbox_style="carto-positron",
                                   color_continuous_scale=color_map,
                                   title = "Estimates of Yearly Increase in Temperature in January for Stations in India, years 1980 - 2020",
                                   color_continuous_midpoint = 0,
                                   width = 1000,
                                   height = 600
                                  )

fig.show()
#saves our plot
write_html(fig, "India.html")
```

{% include India.html %}


Let's look at another country.


```python
color_map = px.colors.diverging.RdGy_r # choose a colormap

fig = temperature_coefficient_plot("China", 1980, 2020, 1, 
                                   min_obs = 10,
                                   zoom = 2,
                                   mapbox_style="carto-positron",
                                   color_continuous_scale=color_map,
                                   title = "Estimates of Yearly Increase in Temperature in January for Stations in China, years 1980 - 2020",
                                   color_continuous_midpoint = 0,
                                   width = 1000,
                                   height = 600
                                  )

fig.show()
#saves our plot
write_html(fig, "China.html")
```

{% include China.html %}



## Finally, more interesting visualizations!

### Let's look at Antartica, and *how climate change has impacted the coldest continent on our planet*. We will use the sns library to graph the distribution of temperature changes in stations of Antartica.

First, let's modify our query function. We'll look at every month of each year. We'll use the latitude parameter to limit our search to Antartica stations (<-80). 


```python
def query_climate_database_2(year_begin, year_end, lat):
    """
    A function that takes three arguments and returns a Pandas dataframe by making a query
    """
    conn = sqlite3.connect("temps.db")
    
    #command line that gets the relevant info    
    cmd = f"SELECT S.name, T.year, T.month, T.temp \
            FROM temperatures T \
            LEFT JOIN stations S ON T.id = S.id\
            WHERE T.year >= {year_begin} \
            AND T.year<={year_end} AND S.latitude < {lat}"
    
    #converts to Pandas dataframe
    df = pd.read_sql_query(cmd, conn)
    conn.close()
    return df
```


```python
#gets the data from our database
df = query_climate_database_2(1960, 2020, -80)
```


```python
import seaborn as sns
from matplotlib import pyplot as plt

def plot_antartica(df, _title, _x_label, _y_label):
    #use the same coefs function to determine the avearge yearly change
    coefs = df.groupby(["NAME", "Month"]).apply(coef)
    coefs = coefs.reset_index()

    #plot the data points
    sns.relplot(data = coefs, 
                x = 0, 
                y = "NAME",
                alpha = 0.5, 
                height = 4,
                aspect = 1.7)
    plt.plot([0,0], [0,12], color = "lightgray", zorder = 0)
    labs = plt.gca().set(xlabel = _x_label,
                         ylabel = _y_label,
                         title = _title)
```


```python
title = "Average Yearly Change in Temperature\nat Antarctic Climate Stations\n(Each dot is one month of the year)"
x = "Regression Coefficient"
y = "Station Name"

#call our function with predetermined parameters
plot_antartica(df, title, x, y)
```

![sns.png](/images/sns.png)

    



It appears that some stations in Antartica have seen significant increases in temperature. Still, most dots fall witin +-0.25 degrees per year.

### Next, let's look at a comparison between two countries with distinct climates, and see if they differ. We will answer the question: does climate change affect Egypt and Iceland differently?


```python
#First, let's define a new function to select the data.
def query_climate_database_comp(country1, country2, year_begin, year_end, month):
    """
    A function that takes four arguments and returns a Pandas dataframe by making a query
    """
    conn = sqlite3.connect("temps.db")
    
    #command line that gets the relevant info    
    cmd = f"SELECT S.name, S.latitude, S.longitude, C.Name, T.year, T.month, T.temp \
            FROM temperatures T \
            LEFT JOIN stations S ON T.id = S.id\
            LEFT JOIN countries C ON C.code = SUBSTRING(T.id, 1, 2)\
            WHERE C.Name='{country1}' OR C.Name='{country2}' AND T.year >= {year_begin} \
            AND T.year<={year_end} AND T.month = {month}"
    
    #converts to Pandas dataframe
    df = pd.read_sql_query(cmd, conn)
    conn.close()
    return df
```


```python
def temperature_coefficient_plot_comp(country1, country2, year_begin, year_end, month, min_obs, **kwargs):
    """
    This function takes in countries, year_begin, year_end, and month, as the previous sql function does. It also
    takes in a minimum observation argument, which dictates the minimum amount of observation for any station's 
    data to be accepted. Anything less will be filtered out. It also accepts additional keywords arguments to 
    style the plot. 
    """
    #call the query function to get the dataframe
    df = query_climate_database_comp(country1, country2, year_begin, year_end, month)
    
    #filters out any station not meeting the min_obs requirement
    df['len'] = df.groupby(["NAME"])["Temp"].transform(len)
    #print(df.head())
    df = df[df['len'] >= min_obs]
    
    #adds esitmated yearly increase column using our coef function
    coefs = df.groupby(["NAME", "LATITUDE", "LONGITUDE", "Name"]).apply(coef)
    coefs = coefs.reset_index()
    #print(coefs.head())
    coefs.rename(columns = {0:'Estimated Yearly Increase (°C)', "Name":"Country"}, inplace = True)
    
    #plots the data
    fig = px.scatter(data_frame=coefs, # data for the points you want to plot
                        x = "NAME", # column name for latitude informataion
                        y = "Estimated Yearly Increase (°C)", # column name for longitude information
                        facet_row = "Country",
                        **kwargs) # map style
    return fig
```


```python
#call our function
fig = temperature_coefficient_plot_comp("Iceland", "Egypt", 1990, 2020, 1, 10, 
                                   title = "Estimates of Yearly Increase in Temperature in January for Stations in Egypt vs Iceland, years 1990 - 2020",
                                   width = 1000,
                                   height = 600)

fig.show()
write_html(fig, "scatter.html")
```

{% include scatter.html %}



It appears that there is no significant difference in the yearly change in temperature, despite the different geography of Egypt and Iceland.
