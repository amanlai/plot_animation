# plot_animation

```python
BarchartAnimation(df, filename=None, n_bars=None, fixed_max=True, periods_per_second=1.4, frames_per_period=70,
                  period_pause=110, colors=None, title=None, barlabel_format='{:,.2f}', barlabel_position='inside', 
                  optional_text=None)
```                      
A function that creates a running bar chart animation (where each figure is size (12,5)) 
using FuncAnimation from matplotlib.animation library. Data must be a wide pandas DataFrame 
where columns represent time periods and indexes represent categories.

If a filename is passed (with an extension supported by ffmpeg), then the animation will 
be saved as a file. If no filename is passed, it will be returned as an HTML video and 
embedded into Jupyter notebook (if you are using it).


```python
LineplotAnimation(df, filename=None, categories=None, rank=True, fixed_max=True, periods_per_second=2, 
                  frames_per_period=10, periods_per_frame=None, colors=None, title=None, yaxis_label=None, 
                  optional_text=None)
```

A function that creates a running line chart animation using FuncAnimation from 
matplotlib.animation library. Data must be a wide pandas DataFrame where columns
represent time periods and indexes represent categories.

Similar to above, if a filename is passed (with an extension supported by ffmpeg), then 
the animation will be saved as a file. If no filename is passed, it will be returned as 
an HTML video and embedded into Jupyter notebook (if you are using it).

You must have ffmpeg installed on your computer to save videos. ffmpeg can be downloaded
from https://www.ffmpeg.org/download.html. Restart your computer after installation.

---
# Demo:

```python
import pandas as pd
from animated_plots import BarchartAnimation, LineplotAnimation
# get data
def get_uefa_coefficient_tables():
    import urllib.request as request
    from bs4 import BeautifulSoup
    
    methods = [1]*(1999-1960) + [2]*(2004-1999) + [3]*(2009-2004) + [4]*(2018-2009) + [5]*(2022-2018)
    urls = ['https://kassiesa.net/uefa/data/method{}/crank{}.html'.format(method, year) 
            for method,year in zip(methods,range(1960,2022))]
    
    tables = []
    for year, url in zip(range(1960, 2022), urls):
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
    
# df contains the historical UEFA country coefficients of all UEFA member countries
# notice indexes correspond to countries and columns correspond to seasons.
df = get_country_coefficients()

df_rank = df.rank(ascending=False)
# top5_leagues keeps the UEFA coefficients of only the top 5 leagues in any given year. 
# if there is a country whose league never was among the top 5, that country is dropped altogether
top5_leagues = df[pd.notna(df_rank[df_rank<=5])].dropna(how='all').fillna(0)

# the following are the countries whose leagues have ever been among the top 5 European leagues.
colors_dict = {'Austria':'#EF3340', 'Belgium':'purple', 'East Germany':'gray', 'England':'#cf081f', 
               'France':'#0072bb', 'Germany':'black', 'Hungary':'green', 'Italy':'#4B61D1', 
               'Netherlands':'#dfa837', 'Spain':'#f1bf00', 'Portugal':'#C4002D', 'Scotland':'navy', 
               'Soviet Union':'crimson', 'Yugoslavia':'dodgerblue'}

# this function creates an animation from the historical ranks of the current top 5 leagues in European football
LineplotAnimation(df=df.fillna(0),
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

# this function creates an animation from the top 5 leagues every season.
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
    
