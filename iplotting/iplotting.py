import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ipywidgets import widgets
from IPython.display import display
import statsmodels.tsa.api as tsa
from src.spike_detection import detect_spikes


class PlotTS:
    def __init__(self, df):
        self.df = df.copy()
        self.ui_col = widgets.SelectMultiple(
            options=df.columns,
            description="Column",
            rows=10,
        )
        self.ui_mode = widgets.Dropdown(
            options=("lines+markers", "lines", "markers"),
            value="lines",
            description="Plot style",
        )
        self.ui_dropna = widgets.Checkbox(
            value=False,
            description="Drop NA",
        )
        self.ui_cumulative = widgets.Checkbox(
            value=False,
            description="Cumulative"
        )
        self.ui_rm_spikes = widgets.Checkbox(
            value=False,
            description="Remove Spikes",
        )
        self.ui_spikes_window = widgets.BoundedIntText(
            value=5,
            min=1,
            max=21,
            step=1,
            description="Spike Window"
        )
        self.ui_spikes_alpha = widgets.BoundedIntText(
            value=3,
            min=1,
            max=10,
            step=1,
            description="Spike Alpha"
        )
        self.ui_spikes_delta = widgets.BoundedIntText(
            value=1,
            min=0,
            max=5,
            step=1,
            description="Spike Delta"
        )
        self.params = dict(
            df=widgets.fixed(df),
            columns=self.ui_col,
            mode=self.ui_mode,
            cumulative=self.ui_cumulative,
            dropna=self.ui_dropna,
            rm_spikes=self.ui_rm_spikes,
            spikes_window=self.ui_spikes_window,
            spikes_alpha=self.ui_spikes_alpha,
            spikes_delta=self.ui_spikes_delta,
        )
        self.ui_main = widgets.VBox([self.ui_mode, self.ui_cumulative, self.ui_dropna])
        self.ui_spikes = widgets.VBox([
            self.ui_spikes_window,
            self.ui_spikes_alpha,
            self.ui_spikes_delta,
            self.ui_rm_spikes
        ])
        self.ui = widgets.HBox([self.ui_col, self.ui_main, self.ui_spikes])
        self.out = widgets.interactive_output(
            self.plot,
            self.params,
        )

    @staticmethod
    def plot(
        df,
        columns=None,
        mode="lines",
        cumulative=False,
        dropna=False,
        rm_spikes=False,
        spikes_window=5,
        spikes_alpha=3,
        spikes_delta=1,
    ):
        if columns is None or len(columns) == 0:
            return go.Figure()
        fig = make_subplots(rows=len(columns), cols=1)
        for i, col in enumerate(columns):
            ts = df.loc[:, col].copy()
            ts.name = col
            if cumulative:
                ts = ts.cumsum()
            if dropna:
                ts = ts.dropna()
            if rm_spikes:
                spikes = detect_spikes(ts, window=spikes_window, alpha=spikes_alpha, delta=spikes_delta)
                ts = ts[~spikes].copy()
            fig.add_trace(
                go.Scatter(
                    x=ts.index,
                    y=ts,
                    mode=mode,
                    name=ts.name,
                ),
                row=i+1,
                col=1,
            )
        return fig.show()

    def show(self):
        display(self.ui, self.out)


class PlotScatter:
    def __init__(self, df):
        self.df = df.copy()
        self.ui_col_x = widgets.Dropdown(
            options=df.columns,
            description="Column X",
        )
        self.ui_col_y = widgets.Dropdown(
            options=df.columns,
            description="Column Y",
        )
        self.ui_dropna = widgets.Checkbox(
            value=False,
            description="Drop NA"
        )
        self.ui_rm_zeros = widgets.Checkbox(
            value=False,
            description="Remove Zeros",
        )
        self.ui_rm_spikes = widgets.Checkbox(
            value=False,
            description="Remove Spikes",
        )
        self.ui_spikes_window = widgets.BoundedIntText(
            value=5,
            min=1,
            max=21,
            step=1,
            description="Spike Window"
        )
        self.ui_spikes_alpha = widgets.BoundedIntText(
            value=3,
            min=1,
            max=10,
            step=1,
            description="Spike Alpha"
        )
        self.ui_spikes_delta = widgets.BoundedIntText(
            value=1,
            min=0,
            max=5,
            step=1,
            description="Spike Delta"
        )
        self.params = dict(
            df=widgets.fixed(self.df),
            col_x=self.ui_col_x,
            col_y=self.ui_col_y,
            dropna=self.ui_dropna,
            rm_zeros=self.ui_rm_zeros,
            rm_spikes=self.ui_rm_spikes,
            spikes_window=self.ui_spikes_window,
            spikes_alpha=self.ui_spikes_alpha,
            spikes_delta=self.ui_spikes_delta,
        )
        self.ui_main = widgets.VBox([self.ui_col_x, self.ui_col_y, self.ui_dropna, self.ui_rm_zeros])
        self.ui_spikes = widgets.VBox([
            self.ui_spikes_window,
            self.ui_spikes_alpha,
            self.ui_spikes_delta,
            self.ui_rm_spikes
        ])
        self.ui = widgets.HBox([self.ui_main, self.ui_spikes])
        self.out = widgets.interactive_output(
            self.plot,
            self.params,
        )

    @staticmethod
    def plot(
        df,
        col_x,
        col_y,
        dropna=False,
        rm_zeros=False,
        rm_spikes=False,
        spikes_window=5,
        spikes_alpha=3,
        spikes_delta=1,
    ):
        if col_x is None or col_y is None:
            return go.Figure()
        x = df[col_x]
        y = df[col_y]
        if dropna:
            x = x.dropna()
            y = y.dropna()
        if rm_spikes:
            x_spikes = detect_spikes(x, window=spikes_window, alpha=spikes_alpha, delta=spikes_delta)
            y_spikes = detect_spikes(y, window=spikes_window, alpha=spikes_alpha, delta=spikes_delta)
            x = x[~x_spikes].copy()
            y = y[~y_spikes].copy()
        if rm_zeros:
            x = x[x != 0].copy()
            y = y[y != 0].copy()
        xy = pd.concat([x, y], axis=1, join="inner").dropna()
        fig = px.scatter(x=xy.iloc[:, 0], y=xy.iloc[:, 1], render_mode="webgl")
        return fig.show()

    def show(self):
        display(self.ui, self.out)


class PlotHistogram:
    def __init__(self, df):
        self.df = df.copy()
        self.ui_col = widgets.SelectMultiple(
            options=df.columns,
            rows=10,
            description="Column",
        )
        self.ui_bins = widgets.IntSlider(
            value=0,
            min=0,
            max=100,
            description="Bins",
            continuous_update=False,
        )
        self.ui_rm_spikes = widgets.Checkbox(
            value=False,
            description="Remove Spikes",
        )
        self.ui_spikes_window = widgets.BoundedIntText(
            value=5,
            min=1,
            max=21,
            step=1,
            description="Spike Window"
        )
        self.ui_spikes_alpha = widgets.BoundedIntText(
            value=3,
            min=1,
            max=10,
            step=1,
            description="Spike Alpha"
        )
        self.ui_spikes_delta = widgets.BoundedIntText(
            value=1,
            min=0,
            max=5,
            step=1,
            description="Spike Delta"
        )
        self.params = dict(
            df=widgets.fixed(df),
            columns=self.ui_col,
            bins=self.ui_bins,
            rm_spikes=self.ui_rm_spikes,
            spikes_window=self.ui_spikes_window,
            spikes_alpha=self.ui_spikes_alpha,
            spikes_delta=self.ui_spikes_delta,
        )
        self.ui_main = widgets.VBox([self.ui_col, self.ui_bins])
        self.ui_spikes = widgets.VBox([
            self.ui_spikes_window,
            self.ui_spikes_alpha,
            self.ui_spikes_delta,
            self.ui_rm_spikes
        ])
        self.ui = widgets.HBox([self.ui_main, self.ui_spikes])
        self.out = widgets.interactive_output(
            self.plot,
            self.params,
        )

    @staticmethod
    def plot(
        df,
        columns,
        bins=None,
        rm_spikes=False,
        spikes_window=5,
        spikes_alpha=3,
        spikes_delta=1,
    ):
        if columns is None or len(columns) == 0:
            return go.Figure()
        if bins == 0:
            bins = None

        fig = make_subplots(
            rows=len(columns),
            cols=1,
        )
        for i, col in enumerate(columns):
            ts = df.loc[:, col].dropna()
            ts.name = col

            if rm_spikes:
                spikes = detect_spikes(ts, window=spikes_window, alpha=spikes_alpha, delta=spikes_delta)
                ts = ts[~spikes].copy()

            fig.add_trace(
                go.Histogram(
                    x=ts,
                    name=ts.name,
                    nbinsx=bins,
                ),
                row=i+1,
                col=1,
            )

        return fig.show()

    def show(self):
        display(self.ui, self.out)


class PlotACF:
    def __init__(self, df):
        self.df = df.copy()
        self.ui_col = widgets.Dropdown(
            options=df.columns,
            description="Column",
        )
        self.ui_fft = widgets.Checkbox(
            value=False,
            description="Use FFT"
        )
        self.ui_lags = widgets.IntSlider(
            value=0,
            min=0,
            max=100,
            description="Lags",
            continuous_update=False,
        )
        self.ui_missing = widgets.RadioButtons(
            options=['none', 'conservative', 'drop'],
            description="Missing Value Policy"
        )
        self.params = dict(
            df=widgets.fixed(df),
            columns=self.ui_col,
            lags=self.ui_lags,
            missing=self.ui_missing,
            fft=self.ui_fft,
        )
        self.ui_main = widgets.VBox([self.ui_col, self.ui_fft, self.ui_lags, self.ui_missing])
        self.out = widgets.interactive_output(
            self.plot,
            self.params,
        )

    @staticmethod
    def plot(df, columns=None, lags=None, missing="none", fft=False):
        if columns is None:
            return plt.Figure()
        if lags == 0:
            lags = None
        ts = df.loc[:, columns].copy()
        tsa.graphics.plot_acf(ts, zero=False, lags=lags, missing=missing, fft=fft)

    def show(self):
        display(self.ui_main, self.out)
