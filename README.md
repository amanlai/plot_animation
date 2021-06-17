# plot_animation

```python
BarchartAnimation(df, filename=None, n_bars=None, fixed_max=True, periods_per_second=1.4, frames_per_period=70,
                  period_pause=110, colors=None, title=None, barlabel_format='{:,.2f}', barlabel_position='inside', 
                  optional_text=None)
```                      
A function that creates a running bar chart animation (where each figure is size 
(12,5)) using FuncAnimation from matplotlib.animation library. Data must be a wide 
pandas DataFrame where columns represent time periods and indexes represent categories.
If a filename is passed (with extension .mp4), then the animation will be saved as
a file. If no filename is passed, it will be returned as an HTML video and embedded
into Jupyter notebook.
You must have ffmpeg installed on your computer to save videos. ffmpeg can be 
downloaded from https://www.ffmpeg.org/download.html. Restart your computer after 
installation.


```python
LineplotAnimation(df, filename=None, categories=None, rank=True, fixed_max=True, periods_per_second=2, 
                  frames_per_period=10, periods_per_frame=None, colors=None, title=None, yaxis_label=None, 
                  optional_text=None)
```

A function that creates a running line chart animation using FuncAnimation from 
matplotlib.animation library. Data must be a wide pandas DataFrame where columns
represent time periods and indexes represent categories.
If a filename is passed (with extension .mp4), then the animation will be saved as
a file. If no filename is passed, it will be returned as an HTML video and embedded
into Jupyter notebook.
You must have ffmpeg installed on your computer to save videos. ffmpeg can be downloaded
from https://www.ffmpeg.org/download.html. Restart your computer after installation.


```python

import pandas as pd
from animated_plots import BarchartAnimation, LineplotAnimation

def get_uefa_coefficient_tables(end_year=None):
    import urllib.request as request
    from bs4 import BeautifulSoup
    
    end_of_data = 2022
    urls = (['https://kassiesa.net/uefa/data/method1/crank'+str(num)+'.html' for num in range(1960,1999)] +
            ['https://kassiesa.net/uefa/data/method2/crank'+str(num)+'.html' for num in range(1999,2004)] + 
            ['https://kassiesa.net/uefa/data/method3/crank'+str(num)+'.html' for num in range(2004,2009)] + 
            ['https://kassiesa.net/uefa/data/method4/crank'+str(num)+'.html' for num in range(2009,2018)] + 
            ['https://kassiesa.net/uefa/data/method5/crank'+str(num)+'.html' for num in range(2018,end_of_data)])
    
    year_url_tuples = zip(range(1960, end_of_data), urls)
    if end_year is not None:
        year_url_tuples = [(year, url) for year, url in year_url_tuples if year<=end_year]
    
    tables = []
    for year, url in year_url_tuples:
        print(year, end='\r')
        page = request.urlopen(url)
        soup = BeautifulSoup(page)
        html_table = soup.find('table', {'class':'t1'})
        table = (pd.read_html(str(html_table))[0][['country','ranking']]
                 .rename(columns = {'ranking':str(year-1)+'-'+str(year)}))
        tables.append(table)
    return tables

def get_country_coefficients():
    country_coefficients_dfs = get_uefa_coefficient_tables()
    rankings = country_coefficients_dfs[0].copy()
    for table in country_coefficients_dfs[1:]:
        rankings = rankings.merge(table, on='country', how='outer')
    return rankings.set_index('country')

df = get_country_coefficients()


colors_dict = {'Austria':'#EF3340', 'Belgium':'purple', 'East Germany':'gray', 'England':'#cf081f', 'France':'#0072bb',
               'Germany':'black', 'Hungary':'green', 'Italy':'#4B61D1', 'Netherlands':'#dfa837', 'Spain':'#f1bf00',
               'Portugal':'#C4002D', 'Scotland':'navy', 'Soviet Union':'crimson', 'Yugoslavia':'dodgerblue'}
df_rank = df.rank(ascending=False)
top5_leagues = df[pd.notna(df_rank[df_rank<=5])].dropna(how='all').fillna(0)




LineplotAnimation(df=df.fillna(0).iloc[:,:10],
                  filename='CurrentTop5ThroughTheYears.mp4', 
                  categories=['Spain','England','Germany','France','Italy'], 
                  fixed_max=False, 
                  rank=True,
                  colors=colors_dict, 
                  title='Historical Ranks of the Top 5 European Leagues', 
                  periods_per_second=1.4,
                  frames_per_period=20,
                  periods_per_frame=10,
                  yaxis_label='UEFA Country Rank')


BarchartAnimation(df=top5_leagues, 
                  filename='Top5EuropeanLeagues.mp4', 
                  n_bars=5, 
                  fixed_max=True, 
                  periods_per_second=1.4, 
                  frames_per_period=50,
                  period_pause=100, 
                  colors=colors_dict, 
                  title='Top 5 European Leagues')

```
    
