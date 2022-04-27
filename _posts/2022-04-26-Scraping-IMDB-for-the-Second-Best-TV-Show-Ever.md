---
layout: post
title: Scraping IMDB for the Second Best TV Show Ever
---


---
layout: post
title: Scraping IMDB for the Second Best TV Show Ever
---

## Web Scraping IMDB
It is common knowledge that *The Wire* by David Simon is the best TV show ever in existence, which begs the question: what comes second? In this blog post, I'll demonstrate how to find similar shows on IMDB using web scraping. 

Here is the link to my GitHub repo hosting the files for my scraper. https://github.com/justinlaicy926/IMDB-Scaper

To start this project, we will be using the Scrapy python library, a very convenient tool for customizing your own personal scraper. In the Scrapy library, a spider object is what scrapes the webpage. In the ImdbSpider object, we will be making three parse functions.

It makes sense that a show that shares the most actors with our show will be of matching quality, so we will be compiling a table of all other shows with shared actors. After that's done, we'll figure out which show is the second best.

After installing Scrapy, run these lines in python command prompt to start an empty Scrapy project.


```python
scrapy startproject IMDB_scraper
```

We will start from the main page of *The Wire*. 
https://www.imdb.com/title/tt0306414/


```python
#make sure to import scrapy at the start of the imdb_spider.py
import scrapy

#first set the start URL to https://www.imdb.com/title/tt0306414/ under the ImdbSpider class
start_urls = ['https://www.imdb.com/title/tt0306414/']
```

We will then implement three spider objects to scrape the main page for the show, the cast page, and each individual actor page. Let's start with the main parse function.


```python
def parse(self, response):
        """
        Parse method, navigates to the Cast segment of the IMDB page and calls the subsequent function
        """

        #creates new url for the credit page 
        cast_link = response.urljoin("fullcredits/")

        #navigates to said page and call the appropriate function 
        yield scrapy.Request(cast_link, callback= self.parse_full_credits)
```

This method works by starting from the show's main page, and utilizing the built-in urljoin function to move into the Full Credits page, where our second method will be deployed. We are able to do this because IMDB is an extremely structured website, and this code is easily scalable to other shows of your liking.

Now that we are on the Full Credits page, let's implement the parse_full_credits method to get to every actor's individual page.


```python
def parse_full_credits(self,response):
        """
        Starts at a Cast page of IMDB, crawl all actors, crew not included, then call the parse_actor_page function 
        """

        #a list of relative paths for each actor   
        rel_paths = [a.attrib["href"] for a in response.css("td.primary_photo a")]

        #craws each link
        if rel_paths:
            for path in rel_paths:
                actor_link = response.urljoin(path)
                yield scrapy.Request(actor_link, callback = self.parse_actor_page)
```

In this function, we are compiling a list of links to each actor's individual page in the rel_paths list. Then, we use the urljoin function again to move to their page and call our third function to finally compile all their works. 

Our third and final method will be parse_actor_page, which compiles a list of all works by one actor.


```python
def parse_actor_page(self, response):
        """
        Crawls each actor page and compiles every work that actor has starred in
        """

        #selects actor name
        actor_name = response.css("span.itemprop::text").get()

        #selects the work from the actor page
        movie_or_TV_name = response.css("div.filmo-row b")
        for movie in movie_or_TV_name:
            yield {"actor" : actor_name, "movie_or_TV_name" : movie.css("a::text").get()}
```

This method works by scraping the actor's name and all of their works first, before yielding a dictionary entry for each of their work.

After finishing the three methods, we can call our function using this command.


```python
scrapy crawl imdb_spider -o results.csv
```

We now have a CSV file with over 1000 entries. Let's now perfrom data analysis to find out what show comes close to the timeless masterpiece that is *The Wire*.


```python
import pandas as pd
```


```python
#read our CSV file into a pandas dataframe
df = pd.read_csv("results.csv")

#inspect our dataframe
df
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
      <th>actor</th>
      <th>movie_or_TV_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Thuliso Dingwall</td>
      <td>The Mechanics Rose</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Thuliso Dingwall</td>
      <td>Person of Interest</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Thuliso Dingwall</td>
      <td>Unforgettable</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Thuliso Dingwall</td>
      <td>Ex$pendable</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Thuliso Dingwall</td>
      <td>Toe to Toe</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>18313</th>
      <td>Lance Reddick</td>
      <td>Great Expectations</td>
    </tr>
    <tr>
      <th>18314</th>
      <td>Lance Reddick</td>
      <td>What the Deaf Man Heard</td>
    </tr>
    <tr>
      <th>18315</th>
      <td>Lance Reddick</td>
      <td>The Nanny</td>
    </tr>
    <tr>
      <th>18316</th>
      <td>Lance Reddick</td>
      <td>Swift Justice</td>
    </tr>
    <tr>
      <th>18317</th>
      <td>Lance Reddick</td>
      <td>New York Undercover</td>
    </tr>
  </tbody>
</table>
<p>18318 rows × 2 columns</p>
</div>



We are interested in finding out which movie or TV show has the most amount of shared actors. Let's use the groupby function for this purpose.


```python
df = df.groupby(["movie_or_TV_name"])["actor"].aggregate(["count"])
df.head()
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
      <th>count</th>
    </tr>
    <tr>
      <th>movie_or_TV_name</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>#BlackGirlhood</th>
      <td>1</td>
    </tr>
    <tr>
      <th>#Like</th>
      <td>1</td>
    </tr>
    <tr>
      <th>#Lucky Number</th>
      <td>1</td>
    </tr>
    <tr>
      <th>#MoreLife</th>
      <td>1</td>
    </tr>
    <tr>
      <th>#PrettyPeopleProblems</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = df.sort_values(("count"), ascending = False)
df.head()
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
      <th>count</th>
    </tr>
    <tr>
      <th>movie_or_TV_name</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>The Wire</th>
      <td>758</td>
    </tr>
    <tr>
      <th>Homicide: Life on the Street</th>
      <td>120</td>
    </tr>
    <tr>
      <th>Law &amp; Order</th>
      <td>104</td>
    </tr>
    <tr>
      <th>Law &amp; Order: Special Victims Unit</th>
      <td>102</td>
    </tr>
    <tr>
      <th>Veep</th>
      <td>74</td>
    </tr>
  </tbody>
</table>
</div>



Unsurprisingly, *The Wire* itself shares the most amount of actors with *The Wire*. We'll be focusing on the runner-ups. 

There appears to be a tie among *Homicide: Life on the Street*, *Law & Order*, and *Law & Order: Special Victims Unit*. All of them are of the same genre.

Here is an visualization of the shared actor scenarios with *The Wire*. Notice how most shows on the plot share very few actors with *The Wire*. It is an indication of David Simon's choice of cast members: he prefers actors that organically resemble their roles to big-names. He worked with people having little or no acting experience but an actual background on the Baltimore streets. 


```python
import plotly.express as px
from plotly.io import write_html
```


```python
fig = px.box(df, y="count",
            labels={
                     "count": "Number of Shared Actor with The Wire",
                 },
            )
fig.show()
write_html(fig, "box.html")
```


{% include box.html %}



This is a fairly extreme distribution, with most shows having little resemblence. Again, this shows how special *The Wire* really is. No movie or TV ever comes close to its execellence.

```python

```
