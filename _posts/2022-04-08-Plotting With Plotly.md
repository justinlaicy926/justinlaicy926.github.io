---
layout: post
title: Data Visualization using Plotly
---

Tutorial: Making Interesting Data Visualization of the Palmer Penguins Data Set

This tutorial is for using the python plotly graphing library. For more information please visit https://plotly.com/python/.

## First, import data from Github. Clean up the data set.


```python
#import the relevant libraries
import pandas as pd
from plotly import express as px
from plotly.io import write_html

#import the data from github
url = "https://raw.githubusercontent.com/PhilChodrow/PIC16B/master/datasets/palmer_penguins.csv"
penguins = pd.read_csv(url)

#clean up the data
penguins = penguins.dropna(subset = ["Body Mass (g)", "Sex"])
penguins["Species"] = penguins["Species"].str.split().str.get(0)
penguins = penguins[penguins["Sex"] != "."]

#select the desired columns
cols = ["Species", "Island", "Sex", "Culmen Length (mm)", "Culmen Depth (mm)", 
        "Flipper Length (mm)", "Body Mass (g)"]
penguins = penguins[cols]
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
      <th>Species</th>
      <th>Island</th>
      <th>Sex</th>
      <th>Culmen Length (mm)</th>
      <th>Culmen Depth (mm)</th>
      <th>Flipper Length (mm)</th>
      <th>Body Mass (g)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>MALE</td>
      <td>39.1</td>
      <td>18.7</td>
      <td>181.0</td>
      <td>3750.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>FEMALE</td>
      <td>39.5</td>
      <td>17.4</td>
      <td>186.0</td>
      <td>3800.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>FEMALE</td>
      <td>40.3</td>
      <td>18.0</td>
      <td>195.0</td>
      <td>3250.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>FEMALE</td>
      <td>36.7</td>
      <td>19.3</td>
      <td>193.0</td>
      <td>3450.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>MALE</td>
      <td>39.3</td>
      <td>20.6</td>
      <td>190.0</td>
      <td>3650.0</td>
    </tr>
  </tbody>
</table>
</div>



### Next, we will make our first data visualization. Let's start from the scatter plot. 

We will take advantage of the plotly.scatter method. For its parameters, we will pass in the Culmen Length, the Culmen Depth, and color code the species. Set the opacity to 0.7 to avoid cluttering.


```python
#use the scatter function, plot Culmen Length against Culmen Depth
fig = px.scatter(data_frame = penguins, 
                x = "Culmen Length (mm)",
                y = "Culmen Depth (mm)", 
                color = "Species", 
                #modify opacity to make the plot less cluttered
                opacity = 0.7,
                width = 500,
                height = 300)

#show the completed plot
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()
#store the plot as an html file
write_html(fig, "example_fig1.html")
```


{% include example_fig1.html %}


### We will delve deeper into the scatter plots by making it faceted across the sex and islands of the penguins. 

To do this, we will take advantage of the facet_col and facet_row parameters of the scatter function. 


```python
#with the facet_col and facet_row parameters, we can make the scatter plot facted 
fig = px.scatter(data_frame = penguins, 
                x = "Culmen Length (mm)", 
                y = "Culmen Depth (mm)", 
                #differentiate the species with colors
                color = "Species", 
                #modify opacity to make the plot less cluttered
                opacity = 0.5,
                #arbitrary width and height
                width = 500,
                height = 300,
                facet_col = "Sex",
                facet_row = "Island") 

#show the completed plot
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()
write_html(fig, "example_fig2.html")
```


{% include example_fig2.html %}


### We can also make a 3D scatter plot using plotly using the scatter_3D function.


```python
#the 3D scatter plot function takes in three axis, Body Mass, Culmen Length, Culmen Depth
fig = px.scatter_3d(penguins,
                    x = "Body Mass (g)",
                    y = "Culmen Length (mm)",
                    z = "Culmen Depth (mm)",
                    #differentiate the species with colors
                    color = "Species",
                    #modify opacity to make the plot less cluttered
                    opacity = 0.5)

#show the completed plot
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()
write_html(fig, "example_fig3.html")
```


{% include example_fig3.html %}


Now you know how to make interesting variations of scatter plots using Plotly. For more plots and formats, visit https://plotly.com/python/.
