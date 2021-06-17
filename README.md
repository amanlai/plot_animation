# plot_animation

```
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


```
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
    
    
