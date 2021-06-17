import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class CommonMethods:
    def validate_params(self):
        if self.filename and not isinstance(self.filename, str):
            raise ValueError('`filename` must be either a string or None')
        if self.filename and '.' not in self.filename:
            raise ValueError('`filename` must have an extension')
        if not isinstance(self.periods_per_second, (int, float)) or self.periods_per_second <= 0:
            raise ValueError('`periods_per_second` must be a positive real number')
        if not isinstance(self.frames_per_period, int) or self.frames_per_period <= 0:
            raise ValueError('`frames_per_period` must be a positive integer')
        if not isinstance(self.title, str):
            raise ValueError('`title` must be a string')
        if self.optional_text and not isinstance(self.optional_text, str):
            raise ValueError('`optional_text` must be a string')
    
    def generate_colormap(self, color_count=10):
        from matplotlib.colors import ListedColormap
        shade_count = int(color_count/7)
        arr = np.linspace(0,1,color_count+1)[:-1]
        arr = np.hstack(([arr[i::shade_count] for i in range(shade_count)]))
        cm = plt.cm.hsv(arr)
        lower_half = int(color_count/2)
        cm[:lower_half,:3] = np.multiply(np.array([np.linspace(0.4, 1, lower_half),]*3).T, 
                                         cm[:lower_half,:3])
        cm[lower_half:,:3] = np.multiply(np.array([np.linspace(1,2,(color_count-lower_half)),]*3).T, 
                                         cm[lower_half:,:3])
        cm[cm>1] = 1
        return ListedColormap(cm)

    def extend_data(self):
        bar_lengths = self.df.loc[:,self.df.columns.repeat(self.frames_per_period)]
        bar_lengths.loc[:,bar_lengths.columns.duplicated(keep='first')] = np.nan
        yaxis_positions = bar_lengths.rank(method='first', ascending=False).interpolate(axis=1)
        bar_lengths = bar_lengths.interpolate(axis=1)
        return bar_lengths, yaxis_positions

    def set_colors(self, colors):
        if colors is None:
            cmap = self.generate_colormap(len(self.yaxis_positions))
            colormap = {country: cmap(idx) for idx, country in enumerate(self.yaxis_positions.index.tolist())}
        elif set(self.yaxis_positions.index).issubset(colors.keys()):
            colormap = colors
        else:
            raise ValueError('`colors` must be either None or a dictionary of color codes \
where a color code is provided for every dataframe index')
        return colormap
     
    def xlim_adjuster(self, xmax):
        try:
            plotlabel_format = self.barlabel_format
        except AttributeError:
            plotlabel_format = '{}'
        try:
            longest_text = self.longest_text
        except AttributeError:
            longest_text = xmax
        f, a = plt.subplots(1,1, figsize=(12,5))
        a.set_xlim(0,1)
        text = a.annotate(plotlabel_format.format(longest_text), xy=(0,0))
        [x1,y1],[x2,y2]=a.transData.inverted().transform(text.get_tightbbox(f.canvas.get_renderer()))
        new_xmax = ( xmax * 1.1 ) / ( 0.989 - 1.1 * (x2-x1) )
        a.cla()
        f.clf()
        plt.close(f)
        return new_xmax

    
    def init_plotter_func(self):
        ax = self.fig.axes[0]
        self.plot_graph(ax, 0)
    
    def plotter_func(self, i):
        if i is None:
            return
        ax = self.fig.axes[0]
        self.graph_removal(ax)
        for text in ax.texts[:]:
            text.remove()            
        self.plot_graph(ax, i)
        print('{}% done.'.format(round(100*(i+1)/(len(self.frames)-self.frames_per_period/2),1)), end='\r')
        
        
    def get_frame_list(self, fps):
        try:
            pause_length = self.period_pause
        except AttributeError:
            pause_length = 0
        frames = []
        for frame_index in range(self.yaxis_positions.shape[1]):
            frames.append(frame_index)
            if frame_index % self.frames_per_period == 0:
                for i in range(int(pause_length * fps / 1000)):
                    frames.append(frame_index)
        frames += [frames[-1]] * int(self.frames_per_period / 2)
        return frames

    def animate(self):
        interval = self.periods_per_second / self.frames_per_period
        fps = self.frames_per_period * self.periods_per_second
        self.frames = self.get_frame_list(fps)
        
        animation = FuncAnimation(self.fig, self.plotter_func, self.frames, self.init_plotter_func, interval=interval)
        if self.filename:
            extension = self.filename.split('.')[-1]
            writer = 'html' if extension=='html' else plt.rcParams['animation.writer']
            video = animation.save(self.filename, fps=fps, writer=writer) 
        else:
            from IPython.display import HTML
            video = HTML(animation.to_jshtml())
        print('Done!       ')
        plt.close()
        return video





class _RunningBarchart(CommonMethods):
    
    def __init__(self, df, filename, n_bars, fixed_max, periods_per_second, frames_per_period, period_pause,
                 colors, title, barlabel_format, barlabel_position, optional_text):
        
        self.df = df
        self.filename = filename
        self.n_bars = n_bars if n_bars else len(df)
        self.fixed_max = fixed_max
        self.periods_per_second = periods_per_second
        self.frames_per_period = frames_per_period
        self.period_pause = period_pause
        self.title = title if title else '[Insert Title Here]'
        self.barlabel_format = barlabel_format 
        self.barlabel_position = barlabel_position
        self.optional_text = optional_text
        self.validate_params()
        self.bar_lengths, self.yaxis_positions = self.extend_data()
        self.colors = self.set_colors(colors)
        self.periods = self.bar_lengths.columns.astype(str).tolist()
        self.fig = self.create_figure()

    def validate_params(self):
        super().validate_params()
        if not isinstance(self.fixed_max, bool):
            raise ValueError('`fixed_max` must be boolean')
        if not isinstance(self.n_bars, int) or self.n_bars <= 0:
            raise ValueError('`n_bars` must be a positive integer')
        if not isinstance(self.period_pause, (int, float)) or self.period_pause <= 0:
            raise ValueError('`period_pause` must be a positive real number')
        if self.barlabel_position not in ('inside','outside'):
            raise ValueError('`barlabel_position` must be either "inside" or "outside"')
        
    def create_figure(self):
        fig = plt.figure(figsize=(12,5), dpi=144)
        ax = fig.add_subplot()
        if self.fixed_max:
            self.xmax = self.adjust_xmax(self.bar_lengths.max().max())
            ax.set_xlim(None, self.xmax)
        ax.set_ylim(0.6,self.n_bars+0.7)
        ax.set_title(self.title, fontsize=25)
        ax.tick_params(top=False, bottom=False, left=False, right=False, labelleft=True, labelbottom=False, 
                       color=[0,0,0,0.5])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_ylabel('')
        ax.set_xlabel('')
        return fig
    
    def adjust_xmax(self, max_bar_length):
        if self.barlabel_position == 'inside':
            new_xmax = max_bar_length * 1.1
        else:
            new_xmax = self.xlim_adjuster(max_bar_length)
        return new_xmax
    
    def plot_graph(self, ax, i):
        i_th_barplot_positions = self.yaxis_positions.iloc[:,i]
        # must subtract the positions from the number of bars so that we have a descending bar plot
        i_th_barplot_positions = self.n_bars + 1 - i_th_barplot_positions[i_th_barplot_positions<self.n_bars+1]
        bar_labels = i_th_barplot_positions.index.tolist()
        i_th_bar_lengths = self.bar_lengths.iloc[:,i][bar_labels]
        bar_colors = [self.colors[label] for label in bar_labels]
        
        ax.barh(i_th_barplot_positions, i_th_bar_lengths, tick_label=bar_labels, 
                color=bar_colors, edgecolor=bar_colors, height=0.8, alpha=0.7)

        if not self.fixed_max:
            xmax = self.adjust_xmax(i_th_bar_lengths.max())
            ax.set_xlim(None, xmax)
        else:
            xmax = self.xmax

        if len(ax.texts) == 0:
            ax.text(s=self.periods[i], transform=ax.transAxes, size=25, x=0.95, y=0.1, ha='left', va='bottom')
        else:
            ax.texts[0].set_text(self.periods[i])
            
        if self.barlabel_position == 'inside':
            ha = 'right' 
            adj = -xmax / 100
            color = 'white'
        else:
            ha = 'left'
            adj = xmax / 100
            color = 'black'
        for x,y in zip(i_th_bar_lengths, i_th_barplot_positions):
            ax.annotate(self.barlabel_format.format(x), xy=(x+adj, y), ha=ha, color=color, fontsize=10)
            
        if self.optional_text:
            ax.text(s=self.optional_text, transform=ax.transAxes, size=8, x=0.999, y=0.001, ha='left', va='bottom') 

    def graph_removal(self, ax):
        for bar in ax.containers:
            bar.remove()
            
    




#################################################################################################################


class _RunningLineplots(CommonMethods):
    
    def __init__(self, df, filename, categories, rank, fixed_max, periods_per_second, frames_per_period, 
                 periods_per_frame, colors, title, yaxis_label, optional_text):
    
        self.df = df
        self.filename = filename if filename else None
        self.categories = categories if categories else df.index.tolist() 
        self.fixed_max = fixed_max
        self.rank = rank
        self.periods_per_second = periods_per_second
        self.frames_per_period = frames_per_period
        self.periods_per_frame = periods_per_frame if not self.fixed_max and periods_per_frame is not None else self.df.shape[1]
        self.title = title if title else '[Insert Title Here]'
        self.yaxis_label = yaxis_label
        self.optional_text = optional_text
        data, data_rank = self.extend_data()
        self.yaxis_positions = data_rank.loc[self.categories] if self.rank else data.loc[self.categories]
        self.validate_params()
        self.longest_text = max(((cat, len(cat)) for cat in self.categories), key=lambda el:el[1])[0]
        self.number_of_data_points = self.frames_per_period * self.periods_per_frame
        self.colors = self.set_colors(colors)
        self.fig = self.create_figure()
        
    def validate_params(self):
        super().validate_params()
        if not set(self.categories).issubset(self.df.index):
            raise ValueError('`categories` must be a subset of dataframe indexes')
        if not isinstance(self.periods_per_frame, int) or self.periods_per_frame <=0:
            raise ValueError('`periods_per_frame` must be a positive integer')
        if self.periods_per_frame > self.df.shape[1]:
            raise ValueError('`periods_per_frame` must not be greater than the number of dataframe columns')
        if self.yaxis_label and not isinstance(self.yaxis_label, str):
            raise ValueError('`yaxis_label` must be None or a string')
        
    def create_figure(self):
        fig = plt.figure(figsize=(12,5), dpi=144)
        ax = fig.add_subplot()
        figw, figh = (12,5)
        fig.subplots_adjust(left=1.2/figw, right=1-1.4/figw, bottom=1.05/figh, top=1-1.1/figh)
        ax.set_title(self.title, fontsize=20, y=1.12)
        ax.tick_params(which='both', 
                       top=False, bottom=True, left=True, right=False, labelleft=True, labelbottom=True, 
                       color=[0,0,0,0.2], labelsize=8)
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)
        if self.fixed_max:
            ymin = int(self.yaxis_positions.min().min())
            ymax = int(self.yaxis_positions.max().max()) + 1
            xmax = self.yaxis_positions.shape[1] + 1
            xticklabels = [label if idx % (self.frames_per_period*5)==0 else '' 
                           for idx, label in enumerate(self.yaxis_positions.columns)]
            self.set_axes_limits(ax, ymin, ymax, 0, xmax, xticklabels)
            ax.xaxis.set_ticks(range(0, xmax, self.frames_per_period), minor=True)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel('')
        ax.set_ylabel(self.yaxis_label)
        if self.rank:
            ax.invert_yaxis()
        return fig

    def set_axes_limits(self, ax, ymin, ymax, xmin, xmax, xticklabels):
        ymin, ymax = ymin - ymin % 5, ymax + ymax % 5
        ax.set_ylim((ymin, ymax))
        ytick_step = max(1, int((ymax-ymin)/10)) if self.rank else max(1, int((ymax-ymin)/5))
        yticklabels = range(ymin+1, ymax+1, ytick_step)
        ax.yaxis.set_ticks(yticklabels)
        ax.set_yticklabels(yticklabels)
        ax.set_xlim((xmin, xmax))
        ax.xaxis.set_ticks([x for x, label in zip(range(xmin,xmax), xticklabels) if label!=''])
        ax.set_xticklabels([label for label in xticklabels if label!=''])
        if self.rank:
            ax.yaxis.set_ticks(range(ymin,ymax), minor=True)


    def get_current_data(self, i):
        if self.fixed_max:
            i_th_plot_positions = self.yaxis_positions.iloc[:,:i]
            line_labels = i_th_plot_positions.index.tolist()
        else:
            if i <= self.number_of_data_points:
                i_th_plot_positions = self.yaxis_positions.iloc[:,:i]
                line_labels = i_th_plot_positions.index.tolist()
            else:
                i_th_plot_positions = self.yaxis_positions.iloc[:,i-self.number_of_data_points:i]
                line_labels = i_th_plot_positions.index.tolist()
        return i_th_plot_positions, line_labels
    
    
    def plot_graph(self, ax, i):
        i_th_plot_data, line_labels = self.get_current_data(i)
        xmin = max(0, i-self.number_of_data_points)
        xdata = range(xmin, i)
        for label in line_labels:
            ydata = i_th_plot_data.loc[label].values
            ax.plot(xdata, ydata, color=self.colors[label], linewidth=0.7, alpha=0.7)
            if i>0:
                ax.annotate(label, xy=(xdata[-1],ydata[-1]), color=self.colors[label], xycoords='data',
                            xytext=(6,0), textcoords='offset points', va='center', size=9)
        if not self.fixed_max: 
            if i > self.number_of_data_points:
                ymin = int(i_th_plot_data.min().min())
                ymax = int(i_th_plot_data.max().max())+1
                xticklabels = np.array(i_th_plot_data.columns)
            else:
                initial_data = self.yaxis_positions.iloc[:,:self.number_of_data_points]
                ymin = int(initial_data.min().min())
                ymax = int(initial_data.max().max())+1
                xticklabels = np.array(initial_data.columns)
            xmax = max(i+1, self.number_of_data_points)
            unique_indexes = np.unique(xticklabels, return_index=True)[1]
            xticklabels = [label if idx in unique_indexes else '' for idx, label in enumerate(xticklabels)]
            xticklabels[0] = '' if xticklabels[self.frames_per_period-1] != xticklabels[0] else xticklabels[0]
            self.set_axes_limits(ax, ymin, ymax, xmin, xmax, xticklabels)
            if self.rank:
                ax.invert_yaxis()
        if self.optional_text:
            ax.annotate(self.optional_text, transform=ax.transAxes, size=8, x=0.999, y=0.001, ha='right',va='bottom')
            
    def graph_removal(self, ax):
        for line in ax.lines:
            line.remove()

        
def BarchartAnimation(df, filename=None, n_bars=None, fixed_max=True, periods_per_second=1.4, frames_per_period=70,
                      period_pause=110, colors=None, title=None, barlabel_format='{:,.2f}', barlabel_position='inside', 
                      optional_text=None):
    """
    A function that creates a running bar chart animation where for a given dataframe df, 
    the top `n_bars` values for each column will be plotted as bars ranked from 1st to 
    `n_bars`th and joined together using FuncAnimation from matplotlib.animation library 
    to create a running animation. Data must be a wide pandas DataFrame where columns 
    represent time periods and indexes represent categories.
    If a filename is passed (with an extension supported by ffmpeg), then the animation will 
    be saved as a file. If no filename is passed, it will be returned as an HTML video and 
    embedded into Jupyter notebook (if you are using it).
    You must have ffmpeg installed on your computer to save videos. ffmpeg can be 
    downloaded from https://www.ffmpeg.org/download.html. Restart your computer after 
    installation.
    ------------------------------------------------------------------------------------
    Parameters:
    -----------
    df: pandas DataFrame.
        Must be a wide DataFrame where each column represents a single time period and
        each row represents the values of that category across time. The animation will
        use the index of this DataFrame as bar labels.
    filename: str or None, default: None
        If you want to save the animation, a filename must be provided. The allowed 
        extensions are: .mp4, .mov or (any other extensions supported by ffmpeg) or .html. 
        If no filename is provided, the animation will be embedded into the Jupyter 
        Notebook file as jsanimation provided the video size is not too large.
    n_bars: int, default: len(df)
        Number of bars to plot in each frame. Must be an integer not greater than the 
        length of the dataframe.
    fixed_max: bool, default: True
        Whether to fix the maximum value of the x-axes of the plots across time. If 
        True, the maximum value of the x-axis will be fixed as the maximum possible 
        value in the passed DataFrame. If False, then for each period, the maximum 
        value of the x-axis will be set as the length of the longest bar and will 
        change in each period.
    periods_per_second: int or float, default: 1.4
        Number of time periods (i.e. DataFrame columns) to show per second. Increasing 
        this number makes the animation shorter.
    frames_per_period: int, default: 70
        Number frames to show from one time period to the next (i.e. one DataFrame 
        column to the next column). Increasing this number makes the animation smoother.
    period_pause: int or float, default: 100
        Number of milliseconds to show each period in the data on top of 
        (1000 / periods_per_second / frames_per_period) milliseconds, which is the 
        amount of time each frame is shown.
    colors: dict or None, default: None
        How to color bars depending on their label. If a dictionary is passed, its keys
        must match the DataFrame index, i.e. a color code must be provided for each 
        DataFrame index. If None is passed, bar colors are chosen from a modified hsv 
        colormap (modified to make similar colors slightly visually different from one 
        another) from the matplotlib colormap object.
    title: str or None, default: '[Insert Title Here]'
        Title of the plots/animation as a string.
    barlabel_format: str or None, default: '{:,.2f}'
        Determines how to show the bar length labels on bars. Default is to show the
        length in up to 2 decimal points.
    barlabel_position: str or None, default: 'inside'
        Where to place bar labels relative to the bars. Can be either "inside" or 
        "outside".
    optional_text: str or None, default: None
        An optional text as a string to be shown at the bottom right corner of the 
        plots/animation. 
    -------------------------------------------------------------------------------------
    Returns:
    --------
    If no filename is passed, a HTML animation will be returned. If a filename with 
    extension .mp4 is passed, the animation will be saved and None will be returned.
    -------------------------------------------------------------------------------------
    Example:
    --------
    BarchartAnimation(df=df, 
                      filename='top_4_uefa_leagues.mp4', 
                      n_bars=4, 
                      fixed_max=True, 
                      periods_per_second=2, 
                      frames_per_period=50,
                      period_pause=100, 
                      colors=None, 
                      title='An example animation', 
                      data_unit='fractions', 
                      barlabel_position='inside',
                      optional_text='Data from kassiessa.net.')
    """
    instnce = _RunningBarchart(df, filename, n_bars, fixed_max, periods_per_second, frames_per_period, period_pause,
                               colors, title, barlabel_format, barlabel_position, optional_text)
    return instnce.animate()        
        


    
    
    
########################################################################################################


def LineplotAnimation(df, filename=None, categories=None, rank=True, fixed_max=True, periods_per_second=2, 
                      frames_per_period=10, periods_per_frame=None, colors=None, title=None, yaxis_label=None, 
                      optional_text=None):
    """
    A function that creates a running line chart animation using FuncAnimation from 
    matplotlib.animation library. Data must be a wide pandas DataFrame where columns
    represent time periods and indexes represent categories. For a given dataframe df and 
    categories (which is a subset of df.index), each animation frame plots the value of each
    category in a particular column of df.
    If a filename is passed (with extension .mp4), then the animation will be saved as
    a file. If no filename is passed, it will be returned as an HTML video and embedded
    into Jupyter notebook.
    You must have ffmpeg installed on your computer to save videos. ffmpeg can be downloaded
    from https://www.ffmpeg.org/download.html. Restart your computer after installation.
    ------------------------------------------------------------------------------------
    Parameters:
    -----------
    df: pandas DataFrame.
        Must be a wide DataFrame where each column represents a single time period and
        each row represents the values of that category across time. The animation will
        use the index of this DataFrame as lineplot labels and its columns as x-axis labels.
    filename: str or None, default: None
        If you want to save the animation, a filename must be provided. The allowed 
        extensions are: .mp4, .mov or (any other extensions supported by ffmpeg) or .html. 
        If no filename is provided, the animation will be embedded into the Jupyter 
        Notebook file as jsanimation.
    categories: list, default: df.index
        List of categories that will be graphed. Must be a subset of df.index.
    rank: bool, default: True
        Whether to plot the ranks or the actual data values of the passed dataframe. 
        If False, the y-axis of the plots/animation will be the data values of the passed 
        categories in the dataframe. If True, the y-axis will be the ranks of the passed 
        categories relative to other categories. For example, if 'France' is a category 
        in df.index, then setting rank=True will plot the ranks of 'France' across time 
        periods.
    fixed_max: bool, default: True
        Whether to fix the maximum value of the the plot axes across time. If True, 
        the maximum value of the axes will be fixed as the maximum possible values 
        in the passed DataFrame. If False, then for each period, the maximum 
        values of the axes will be set as the length of the time periods to be shown and 
        the maximum value of the values to be shown in that particular frame.
    periods_per_second: int or float, default: 2
        Number of time periods (i.e. DataFrame columns) to show per second. Increasing 
        this number makes the animation shorter.
    frames_per_period: int, default: 20
        Number frames to show from one time period to the next (i.e. one DataFrame 
        column to the next column). Increasing this number makes the animation smoother.
    periods_per_frame: int, default: df.shape[1]
        Number of time periods to show in a single frame. Only relevant if fixed_max=True.
        Default value is df.shape[1]. Must be a positive integer no more than df.shape[1].
    colors: dict or None, default: None
        How to color bars depending on their label. If a dictionary is passed, its keys
        must match the DataFrame index, i.e. a color code must be provided for each 
        DataFrame index. If None is passed, bar colors are chosen from a modified hsv 
        colormap (modified to make similar colors slightly visually different from one 
        another) from the matplotlib colormap object.
    title: str or None, default: '[Insert Title Here]'
        Title of the plots/animation as a string.
    yaxis_label: str or None, default: None
        y-axis label as a string.
    optional_text: str or None, default: None
        An optional text as a string to be shown at the bottom right corner of the 
        plots/animation. 
    -------------------------------------------------------------------------------------
    Returns:
    --------
    If no filename is passed, a HTML animation will be returned. If a filename with 
    extension .mp4 is passed, the animation will be saved and None will be returned.
    -------------------------------------------------------------------------------------
    Example:
    --------
    LineplotAnimation(df=df, 
                      filename='top_5_uefa_leagues.mp4', 
                      categories=['Spain','England','Germany','France','Italy'], 
                      fixed_max=True, 
                      rank=True,
                      periods_per_second=2, 
                      frames_per_period=20,
                      periods_per_frame=10, 
                      colors=None, 
                      title='An example animation', 
                      yaxis_label='Country Rank'
                      optional_text='Data from kassiessa.net.')
    """
    instnce = _RunningLineplots(df, filename, categories, rank, fixed_max, periods_per_second, frames_per_period, 
                                periods_per_frame, colors, title, yaxis_label, optional_text)
    return instnce.animate()
