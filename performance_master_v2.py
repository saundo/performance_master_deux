import ipywidgets
import collections
import datetime
import requests
import pandas as pd
import numpy as np
from IPython.display import display

from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, HoverTool, Span
from bokeh.io import output_notebook, push_notebook, show

from bokeh.io import output_notebook, show
from bokeh.plotting import figure
from bokeh.models import LinearAxis, Range1d, NumeralTickFormatter
from bokeh.layouts import column

class site_summary():

    def __init__(self, df, ad_server='DFP'):
        self.df = df
        self.ad_server = ad_server

        self.d1 = ipywidgets.DatePicker(
            disabled=False,
            description='Start',
            value=df['date'].max() - datetime.timedelta(7))

        self.d2 = ipywidgets.DatePicker(
            disabled=False,
            description='End',
            value=df['date'].max())

    def execute(self, d1, d2):
        if self.ad_server == 'DFP':
            imps = 'dfp_impressions'
        elif self.ad_server == '3P':
            imps = 'normalized_clicks'
        else:
            raise ValueError('ad_server kwarg should be "DFP", "3P"')

        dfx = self.df[(self.df['date'] >= d1) & (self.df['date'] <= d2)]

        dfx = dfx.groupby(('creative_type', 'site'), as_index=False).sum()

        dfx = dfx[['creative_type', 'site', imps]]
        dfx = dfx.sort_values(imps, ascending=False)

        dfx['share'] = (dfx[imps] / dfx[imps].sum()) * 100
        dfx['share cumsum'] = dfx['share'].cumsum()
        dfx['share cumsum'] = dfx['share cumsum'].astype(int)
        dfx['share'] = dfx['share'].astype(int)

        dfx.index = range(len(dfx))
        dfx[imps] = dfx[imps].apply(lambda x: format(x, ','))
        return dfx

class metric_report():
    sites = ('qz', 'wrk', 'zty')

    creative_type_metrics = {
        'branded driver': ('DFP CTR', '3P CTR'),
        'traffic driver': ('DFP CTR', '3P CTR'),
        'video': ('DFP CTR','3P CTR', 'VSR'),
        'interactive non video': ('DFP CTR', '3P CTR','IR'),
        'interactive video': ('DFP CTR', '3P CTR','IR', 'VSR'),
    }

    metric_dict = {
        'DFP': {
            'branded driver':
                [('DFP CTR'),
                 ('dfp_clicks', 'dfp_impressions')],
            'traffic driver':
                [('DFP CTR'),
                 ('dfp_clicks', 'dfp_impressions')],
            'video':
                [('DFP CTR', 'VSR'),
                 ('dfp_clicks', 'dfp_impressions', 'result_5')],
            'interactive non video':
                [('DFP CTR','IR'),
                 ('dfp_clicks', 'dfp_impressions', 'int_sessions')],
            'interactive video':
                [('DFP CTR', 'IR', 'VSR'),
                 ('dfp_clicks', 'dfp_impressions', 'int_sessions','result_5')]
        },
        '3P': {
            'branded driver':
                [('3P CTR'), ('normalized_impressions', 'normalized_clicks')],
            'traffic driver':
                [('3P CTR'), ('normalized_impressions', 'normalized_clicks')],
            'video':
                [('3P CTR', '3P VSR'),
                 ('normalized_impressions', 'normalized_clicks', 'result_5')],
            'interactive non video':
                [('3P CTR','3P IR'),
                 ('normalized_impressions', 'normalized_clicks', 'int_sessions')],
            'interactive video':
                [('3P CTR', '3P IR', '3P VSR'),
                 ('normalized_impressions', 'normalized_clicks', 'int_sessions', 'result_5')]
        }
    }

    creative_types = {
        'branded driver',
        'brand survey',
        'co-branded driver',
        'interactive non video',
        'interactive video',
        'no_match',
        'traffic driver',
        'video',
        'video autoplay'
    }

    def __init__(self, df, site='qz',  ad_server='DFP'):
        """
        """
        self.df = df

        if site not in self.sites:
            raise ValueError('site kwarg should be "qz", "wrk", "zty"')
        self.site = site

        # if creative_type not in self.creative_type_metrics.keys():
        #     raise ValueError('creative_type kwarg should be: ' + (
        #         "; ".join(sorted(self.creative_type_metrics.keys()
        #             ))))
        # self.creative_type = creative_type

        self.ad_server = ad_server

        self.d1 = ipywidgets.DatePicker(
            disabled=False,
            description='Start',
            value=df['date'].max() - datetime.timedelta(7))

        self.d2 = ipywidgets.DatePicker(
            disabled=False,
            description='End',
            value=df['date'].max())

        co = df.fillna(0).groupby('creative_type', as_index=False)['dfp_impressions'].sum()
        co = co.sort_values('dfp_impressions', ascending=False)
        self.creative_type_dropdown = ipywidgets.Dropdown(
            options=co['creative_type'],
            value=co.iloc[0]['creative_type'],
            disabled=False)

        self.slider = ipywidgets.IntSlider(
            value=20,
            min=5,
            max=50,
            step=1,
            continuous_update=False,
            description='display size')

    def metric_calcs(self, df, metric='DFP CTR'):
        if metric == 'DFP CTR':
            x = (df['dfp_clicks'] / df['dfp_impressions']) * 100
            return x.round(2)

        elif metric == '3P CTR':
            x = (df['normalized_impressions'] / df['normalized_clicks']) * 100
            return x.round(2)

        elif metric == 'VSR':
            x = (df['result_5'] / df['dfp_impressions']) * 100
            return x.round(2)

        elif metric == '3P VSR':
            x = (df['result_5'] / df['normalized_clicks']) * 100
            return x.round(2)

        elif metric == 'IR':
            x = ((df['int_sessions'] + df['dfp_clicks']) /
                 df['dfp_impressions']) * 100
            return x.round(2)

        elif metric == '3P IR':
            x = ((df['int_sessions'] + df['normalized_impressions']) /
                 df['normalized_clicks']) * 100
            return x.round(2)

        elif metric == 'View %':
            x = (df['ad_server_impressions'] /
                 df['dfp_impressions']) * 100
            return x.astype(int)

        else:
            raise ValueError('unrecognized metric')


    def execute(self, d1, d2, creative_type, slider):
        """
        INPUTS

            d1 - beginning date inclusive
            d2 - end date inclusive
        """

        #check for kwarg errors
        if self.ad_server == 'DFP':
            view_cols = ['ad_server_impressions']
        elif self.ad_server == '3P':
            view_cols = ['ad_server_impressions',
                         'dfp_impressions']
        else:
             raise ValueError('ad_server kwarg should be "DFP", "3P"')

        groupons = ['advertiser', 'placement']
        metrics = self.metric_dict[self.ad_server][creative_type][0]
        metric_components = self.metric_dict[self.ad_server][creative_type][1]

        categories = groupons + view_cols + list(metric_components)
        imp_col = [i for i in categories if 'impressions' in i and 'server' not in i][0]

        dfx = self.df[(self.df['date'] >= d1) & (self.df['date'] <= d2)]
        dfx = dfx[(dfx['creative_type'] == creative_type) & (dfx['site'] == self.site)]
        dfx = dfx.groupby(groupons, as_index=False).sum()[categories]
        dfx = dfx.sort_values(imp_col, ascending=False)

        if isinstance(metrics, str):
            dfx[metrics] = self.metric_calcs(dfx, metric=metrics)
            display_cols = groupons + [imp_col, 'share', 'share cumsum'] + [metrics] + ['View %']

        elif isinstance(metrics, (list, tuple)):
            for metric in metrics:
                dfx[metric] = self.metric_calcs(dfx, metric=metric)
            display_cols = groupons + [imp_col, 'share', 'share cumsum'] + list(metrics) + ['View %']

        dfx['View %'] = self.metric_calcs(dfx, metric='View %')
        dfx['share'] = (dfx[imp_col] / dfx[imp_col].sum()) * 100
        dfx['share cumsum'] = dfx['share'].cumsum()
        dfx['share cumsum'] = dfx['share cumsum'].astype(int)
        dfx['share'] = dfx['share'].astype(int)
        dfx.index = range(len(dfx))


        return dfx[display_cols].head(slider)

class no_match_sorting():

    def __init__(self, df):
        self.df = df

        self.d1 = ipywidgets.DatePicker(
            disabled=False,
            description='Start',
            value=df['date'].max() - datetime.timedelta(7))

        self.d2 = ipywidgets.DatePicker(
            disabled=False,
            description='End',
            value=df['date'].max())

        self.slider = ipywidgets.IntSlider(
            value=1000,
            min=100,
            max=5000,
            step=100,
            continuous_update=False,
            description='imp threshold')

    def execute(self, d1, d2, imp_thresh):
        """
        df - weekly dataframe
        d1 - start date, inclusive
        d2 - end date, inclusive
        imp_thresh - the minimum number of impressions to evaluate
        """

        no_match = collections.namedtuple(
            'no_match', (
                'advertiser', 'order', 'site', 'line_item', 'status', 'impressions'
            )
        )

        # import pdb; pdb.set_trace()
        DF = self.df[(self.df['date'] >= d1) & (self.df['date'] <= d2) & (self.df['creative_type'] == 'no_match')]

        s1 = []
        for order in set(DF['order']):
            dfx = DF[DF['order'] == order]


            dfx = dfx.groupby(('advertiser', 'site', 'line_item'), as_index=False).sum()
            # if not dfx.empty:

            dfx = dfx[dfx['dfp_impressions'] > imp_thresh]

            if not dfx.empty:
                for i, row in dfx.iterrows():
                    advert = row['advertiser']
                    site = row['site']
                    line_item = row['line_item']
                    status = 'no_match'
                    impressions = row['dfp_impressions']
                    s1.append(
                        no_match(
                            advert, order, site, line_item, status, impressions
                        )
                    )

        no_match = pd.DataFrame(s1)
        no_match = no_match.sort_values('impressions', ascending=False)
        no_match = no_match.reset_index(drop=True)
        return no_match

class dashboard_control():
    """
    class that does everything for the dashboard
    """

    def __init__(self, df, df_bm, bm_process_func, colormap):
        """
        + initialize class with dataframe
        + initiatlive all ipywidgets
        """
        self.df = df
        self.df_bm = df_bm
        self.bm_process_func = bm_process_func
        self.DF = pd.DataFrame()
        self.colormap = colormap

        self.advertisers = ['all'] + sorted(list(set(self.df['advertiser'])))
        self.creative_types = ['all'] + sorted(list(set(self.df['creative_type'])))

        self.advert_dropdown = ipywidgets.Dropdown(
            options=self.advertisers,
            value=self.advertisers[0],
            disabled=False)

        self.creative_type_dropdown = ipywidgets.Dropdown(
            options=self.creative_types,
            value=self.creative_types[0],
            disabled=False)

        yesterday = datetime.datetime.now() - datetime.timedelta(1)
        t_60 = yesterday - datetime.timedelta(60)

        self.d1_DatePicker = ipywidgets.DatePicker(disabled=False, value=t_60)
        self.d2_DatePicker = ipywidgets.DatePicker(disabled=False, value=yesterday)

    def update_creative_types(self, change):
        if self.advert_dropdown.value != 'all':
            sb1 = self.df['advertiser'] == self.advert_dropdown.value
            x1 = self.df[sb1]['creative_type']
        else:
            x1 = self.df['creative_type']

        self.creative_type_dropdown.options = ['all'] + list(set(x1))

    def update_bm_data(self, change):
        d1 = self.d1_DatePicker.value
        d2 = self.d2_DatePicker.value
        if (d1 is not None) and (d2 is not None):
            self.DF = self.bm_process_func(self.df, self.df_bm, d1, d2)

            current_advert = self.advert_dropdown.value
            self.advertisers = ['all'] + sorted(list(set(self.DF['advertiser'])))
            self.advert_dropdown.options = self.advertisers
            self.advert_dropdown.value = current_advert

    def execute_dashboard(self):

        self.advert_dropdown.observe(self.update_creative_types, names='value')

        self.d1_DatePicker.observe(self.update_bm_data, names='value')
        self.d2_DatePicker.observe(self.update_bm_data, names='value')

        master_interact = ipywidgets.interact(self.prep_data,
            advert = self.advert_dropdown, creative = self.creative_type_dropdown,
            date1 = self.d1_DatePicker, date2 = self.d2_DatePicker)

        return master_interact

    def prep_data(self, advert, creative, date1, date2):
        if self.DF.empty:
            print('select some shit')
            return

        if self.advert_dropdown.value != 'all':
            sb1 = self.DF['advertiser'] == self.advert_dropdown.value
        else:
            sb1 = self.DF['advertiser'] != 'POPPER'

        if self.creative_type_dropdown.value != 'all':
            sb2 = self.DF['creative_type'] == self.creative_type_dropdown.value
        else:
            sb2 = self.DF['creative_type'] != 'POPPER'

        self.dfx = self.DF[sb1 & sb2].copy()

        advert_list = sorted(set(self.dfx['advertiser']))
        colors = get_n_colors(len(advert_list), self.colormap)
        advert_color_mapping = dict(zip(advert_list, colors))
        self.dfx['color'] = self.dfx['advertiser'].map(advert_color_mapping)



        self.bokeh_bubbles()

    def bokeh_bubbles(self):
        """

        """
        hover = HoverTool(names=['circle'], tooltips=[
        ("Advertiser", "@advertiser"),
        ("Site", "@site"),
        ("creative_name", "@creative_name_version"),
        ("creative_type", "@creative_type"),
        ("KPI", "@KPI"),
        ("KPI_value", '@metric_prty'),
        ("impressions", "@dfp_impressions"),
        ('placement', '@placement')
        ])

        TOOLS = [hover, "crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom"]
        source = ColumnDataSource(self.dfx)

        ymin = self.dfx['delta'].quantile(.01)
        ymax = self.dfx['delta'].quantile(.975)

        p = figure(
            width=1300,
            height=700,
            x_axis_type="log",
            y_range=(ymin, ymax), # dfx['value'].max() * 1.10),
            tools=TOOLS,
            toolbar_location="right"
        )

        hline = Span(location=0, dimension='width',
            line_dash='dashed', line_color='green', line_width=3)
        p.renderers.extend([hline])
        p.circle('dfp_impressions', 'delta',
            source=source, size=8, name='circle', color='color')#, size='size',  color='color')

        p.xaxis.axis_label = 'Impressions (log)'
        p.yaxis.axis_label = 'KPI difference to benchmark'
        show(p)

class viewability_control():
    """
    class that controls viewability output
    """
    atlas_palette = dict(
        Light_magenta = '#D580AA',
        Magenta = '#EC44BB',
        Dark_magenta = '#AE488A',
        Darkest_magenta = '#7A3056',
        Grey = '#C2C2C2',
        Dark_grey = '#676767',
        Light_blue = '#2EAFE1',
        Blue = '#0583E1',
        Dark_blue = '#00559D',
        Darkest_blue = '#074462'
        )

    def __init__(self, df):
        self.df = df
        self.DF = pd.DataFrame()

        self.adunit = ['engage', 'marquee', 'inline', 'spotlight']
        self.metrics = ['Viewability', 'DFP_CTR', '3P_CTR', 'IR', 'VSR']
        self.site = ['qz', 'wrk', 'zty']

        self.adunit_dropdown = ipywidgets.Dropdown(
            options=self.adunit,
            value=self.adunit[0],
            disabled=False)

        self.metric_dropdown = ipywidgets.Dropdown(
            options=self.metrics,
            value="Viewability",
            disabled=False)

        self.site_dropdown = ipywidgets.Dropdown(
            options=self.site,
            value=self.site[0],
            disabled=False)

        self.rolling = ipywidgets.IntSlider(
            value=4,
            min=1,
            max=7,
            step=1,
            description='rolling',
            disabled=False)

        yesterday = datetime.datetime.now() - datetime.timedelta(1)
        t_60 = yesterday - datetime.timedelta(60)

        self.d1_DatePicker = ipywidgets.DatePicker(disabled=False, value=t_60)
        self.d2_DatePicker = ipywidgets.DatePicker(disabled=False, value=yesterday)

        self.left_box = ipywidgets.VBox(
            [self.site_dropdown, self.metric_dropdown, self.adunit_dropdown]
        )

        self.right_box = ipywidgets.VBox(
            [self.d1_DatePicker, self.d2_DatePicker, self.rolling]
        )

        self.controls = ipywidgets.HBox([self.left_box, self.right_box])

    def metric_calculator(self, df, metric, rolling):
        """
        calculate metrics
        """
        metrics = ['3P_CTR', 'DFP_CTR', 'IR', 'VSR', 'Viewability']
        metric_lookup = {
            'DFP_CTR':
                ('dfp_clicks', 'dfp_impressions'),
            '3P_CTR':
                ('normalized_clicks', 'normalized_impressions'),
            'Viewability':
                ('ad_server_impressions', 'dfp_impressions'),
            'VSR':
                ('result_5', 'dfp_impressions'),
            'IR':
                ('int_sessions','dfp_impressions')
        }

        num = metric_lookup[metric][0]
        den = metric_lookup[metric][1]

        df['num'] = 0
        df['den'] = 0

        for device in ('mobile', 'desktop', 'tablet'):
            index = np.where(df['device'] == device)[0]
            df.loc[index, 'num'] = df.loc[index, num].rolling(rolling).sum()

            index = np.where(df['device'] == device)[0]
            df.loc[index, 'den'] = df.loc[index, den].rolling(rolling).sum()

        df[metric] = (df['num'] / df['den']) * 100

        # imp_roll = df[dem].rolling(rolling).mean()
        return df

    def update_bm_data(self, change):
        d1 = self.d1_DatePicker.value
        d2 = self.d2_DatePicker.value
        if (d1 is not None) and (d2 is not None):
            self.DF = self.df[(self.df['date'] >=d1) & (self.df['date'] <= d2)]

    def update_metric_calcs(self, change):
        metric = self.metric_dropdown.value

    def apply_color(self, df):
        c_mobile = self.atlas_palette['Dark_magenta']
        c_desktop = self.atlas_palette['Dark_blue']
        c_tablet = self.atlas_palette['Grey']

        mapping = zip(
            ['mobile', 'desktop', 'tablet'],
            [c_mobile, c_desktop, c_tablet]
        )
        df['color'] = ''

        for device, color in mapping:
            index = np.where(df['device'] == device)[0]
            df.loc[index, 'color'] = color

        return df

    def apply_size(self, df):
        max_size = 30
        min_size = 1

        ratio = df['den'].max() / max_size
        df['size'] = df['den'] / ratio

        index = np.where(df['size'] < min_size)[0]
        df.loc[index, 'size'] = min_size

        return df

    def pretty_columns(self, df, metric, rolling):
        #pretty date
        df['date_pretty'] = df['date'].dt.strftime("%Y-%m-%d")

        #impression (denominator)
        df['imp_pretty'] = ''
        index = np.where(df['den'].notnull())[0]
        df.loc[index, 'imp_pretty'] = (df.loc[index, 'den'] / rolling).apply(lambda x: format(int(x), ','))



        return df

    def execute_dashboard(self):

        self.output = ipywidgets.interactive_output(self.prep_data, {
            'adunit': self.adunit_dropdown, 'metric': self.metric_dropdown,
            'site': self.site_dropdown, 'rolling': self.rolling,
            'date1': self.d1_DatePicker, 'date2': self.d2_DatePicker
            } )

        self.dash = display(self.controls, self.output)

    def prep_data(self, adunit, metric, site, rolling, date1, date2):
        # if self.DF.empty:
        #     print('select some shit')
        #     return

        vals = [
            'dfp_impressions', 'dfp_clicks',
            'normalized_impressions', 'normalized_clicks',
            'ad_server_impressions',
            'result_5',
            'int_sessions'
        ]

        sb1 = self.df['site'] == site
        sb2 = self.df['adunit'] == adunit
        sb3 = self.df['date'] >= date1
        sb4 = self.df['date'] <= date2

        DF = self.df[(sb1) & (sb2) & (sb3) & (sb4)]
        DF = DF.groupby(('date', 'device', 'site'))[vals].sum().reset_index()

        # metric
        # rolling
        self.DF = self.metric_calculator(DF, metric, rolling)
        self.DF = self.apply_color(DF)
        self.DF = self.apply_size(DF)
        self.DF = self.pretty_columns(DF, metric, rolling)

        sb = self.DF['device'].isin(('mobile', 'desktop', 'tablet'))

        self.bokeh_bubbles(self.DF[sb])
        #display()

    def bokeh_bubbles(self, dfx, ymin=0, ymax=100):
        """

        """

        metric = self.metric_dropdown.value

        ymin = dfx[metric].min() * .9
        ymax = dfx[metric].max() * 1.1

        hover = HoverTool(
            names=['circle'],
            tooltips=[
                ("Date", "@date_pretty"),
                (metric, "@"+metric),
                ("device", "@device"),
                ("site", "@site"),
                ("avg impressions", "@imp_pretty")
                ]
        )

        TOOLS = [hover, "crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom"]
        source = ColumnDataSource(dfx)

        p = figure(
            width=1300,
            height=700,
            x_axis_type="datetime",
            y_range=(ymin, ymax), # dfx['value'].max() * 1.10),
            tools=TOOLS,
            toolbar_location="right"
        )

        p.circle(
            'date',
            metric,
            source=source,
            size='size',
            name='circle',
            color='color'
        )

        for i, device in enumerate(set(dfx['device'])):
            x1 = dfx[dfx['device'] == device]
            my_plot = p.line(
                        x1['date'],
                        x1[metric],
                        legend=device,
                        color=set(x1['color']).pop())
        show(p)

class performance_sumary():

    placement_list = [
        'all',
        'engage mobile',
        'engage desktop',
        'engage tablet',
        'marquee mobile',
        'marquee desktop',
        'marquee tablet',
        'inline mobile',
        'inline desktop',
        'inline tablet'
        ]

    def __init__(self, df, df_bm):
        self.df = df
        self.df_bm = df_bm

        self.placement_dropdown = ipywidgets.Dropdown(
            options=self.placement_list,
            value=self.placement_list[0],
            disabled=False)

    def pre_process(self, start, end, rolling=7):
        df_time = self.date_processing(start, end, rolling=rolling)

        s = []
        for i, row in df_time.iterrows():
            start = row['start']
            end = row['end']
            dft = bm_sub(self.df, self.df_bm, start, end, min_imps=1000)

            x = self.placement_report(dft, end, rolling=rolling)
            s.append(x)

        self.df_proc = pd.concat(s)
        self.df_proc['date'] = pd.to_datetime(self.df_proc['date'])

    def output_select(self, placement_select):
        self.placement_select = placement_select
        self.plotx = self.df_proc[self.df_proc['placement'] == placement_select]
        self.bokeh_plot()

    def date_processing(self, start, end, rolling=7):
        """
        creates date ranges based on rolling to pass to the bm_sb
        returns dataframe timeframe
        """
        dates = pd.date_range(start, end, freq='1D')
        df = pd.DataFrame({'start':dates, 'end':dates.shift(rolling)})
        df = df[['start', 'end']]
        df = df[df['end'] <= end]
        df['start'] = df['start'].dt.strftime("%Y-%m-%d")
        df['end'] = df['end'].dt.strftime("%Y-%m-%d")

        return df

    def placement_report(self, df, d2, rolling=7):
        """
        looks at creatives above / below; includes
        + creatives
        + 'all'

        returns df; cols:
        - placement
        - d2
        - num_crtv_abm
        - pct_crtv_abm
        - num_imp
        - pct_imp_abm
        """

        crt_count = df.groupby('placement', as_index=False)['creative_type'].count()
        crt_count_abm = df[df['delta'] > 0].groupby('placement', as_index=False)['creative_type'].count()
        x1 = pd.merge(crt_count, crt_count_abm, on='placement', how='left')
        x1.columns = ['placement', 'crt_tot', 'crt_abm']
        x1 = x1.fillna(0)

        imp_count = df.groupby('placement', as_index=False)['dfp_impressions'].sum()
        imp_count_abm = df[df['delta'] > 0].groupby('placement', as_index=False)['dfp_impressions'].sum()
        x2 = pd.merge(imp_count, imp_count_abm, on='placement', how='left')
        x2.columns = ['placement', 'imp_tot', 'imp_abm']
        x2 = x2.fillna(0)

        DF = pd.merge(x1, x2, on='placement', how='left')
        DF['crt_share'] = (DF['crt_tot'] / DF['crt_tot'].sum() * 100).round(1)
        DF['imp_share'] = (DF['imp_tot'] / DF['imp_tot'].sum() * 100).round(1)

        DF_all = DF.sum()
        DF_all['placement'] = 'all'
        DF = DF.append(DF_all, ignore_index=True)

        DF['pct_crtv_abm'] = ((DF['crt_abm'] / DF['crt_tot'])*100).astype(int)
        DF['pct_imp_abm'] = ((DF['imp_abm'] / DF['imp_tot'])*100).astype(int)

        DF['imp_tot_day_avg'] = DF['imp_tot'] / rolling
        DF['crt_tot_day_avg'] = DF['crt_tot'] / rolling


        DF['date'] = d2

        return DF

    def bokeh_plot(self):
        output_notebook()

        #### PERFORMANCE #### PERFORMANCE #### PERFORMANCE
        ################## S3 ##################
        s3 = figure(x_axis_type="datetime",
                    title = ("Performance of " + self.placement_select),
                    plot_width=1400,
                    plot_height=400,
                    toolbar_location="above")

        s3.line(self.plotx['date'], self.plotx['pct_crtv_abm'],
                color="#D5E1DD",
                line_color="green",
                legend="creatives",
                line_width=2,
                line_dash='solid')

        s3.yaxis.axis_label = 'Percent above benchmark'

        # Setting the second y axis range name and range
        s3.line(self.plotx['date'],
                self.plotx['pct_imp_abm'],
                color="black",
                legend='impressions',
                line_width=2,
                line_dash='solid')

        s3.legend.location = "bottom_left"

        s3.yaxis.formatter=NumeralTickFormatter(format="0.0a")


        ######## IMPRESSIONS
        ################## S1 ##################
        s1 = figure(x_axis_type="datetime",
                    title= 'Impressions',
                    plot_width=1400,
                    plot_height=180,
                    toolbar_location="above")

        s1.vbar(x=self.plotx['date'],
                top=self.plotx['imp_tot_day_avg'],
                width=36000000,
                color='black',
                alpha=0.25)

        s1.yaxis.axis_label = 'Total'

        # Setting the second y axis range name and range
        s1.add_layout(
            LinearAxis(
                y_range_name="percent",
                axis_label='Impression share'),
            'right')

        y_max = self.plotx['imp_share'].max() * 1.1
        s1.extra_y_ranges = {"percent": Range1d(start=0, end=y_max)}

        s1.line(self.plotx['date'],
                self.plotx['imp_share'],
                color="black",
                y_range_name="percent",
                line_width=2,
                line_dash='solid')

        s1.legend.location = "bottom_left"

        s1.yaxis.formatter=NumeralTickFormatter(format="0.0a")

        ######## CREATIVES
        ################## S2 ##################
        s2 = figure(x_axis_type="datetime",
                    title = "Creatives - 1k imp rolling",
                    plot_width=1400,
                    plot_height=180,
                    toolbar_location="above")

        s2.vbar(x=self.plotx['date'],
                top=self.plotx['crt_tot'],
                width=36000000,
                color='green',
                alpha=0.25)

        s2.yaxis.axis_label = 'Total'

        # Setting the second y axis range name and range
        s2.add_layout(
            LinearAxis(
                y_range_name="percent",
                axis_label='Creative share'),
            'right')
        y_max = self.plotx['imp_share'].max() * 1.1
        s2.extra_y_ranges = {"percent": Range1d(start=0, end=y_max)}

        s2.line(self.plotx['date'],
                self.plotx['crt_share'],
                color="green",
                y_range_name="percent",
                line_width=2,
                line_dash='solid')

        s2.legend.location = "bottom_left"

        s2.yaxis.formatter=NumeralTickFormatter(format="0.0a")

        show(column(s3, s1, s2))

def mismatched_checker(df, d1, d2, imp_thresh=1000):
    """
    Finds all campaigns where creative_type is pulling in an incorrect type.
    Returns creative versions where the main KPI for that creative_type is receiving no actions.
        ex: VSR = NaN, IR = NaN, CTR = 0.0000


    Inputs:
    df = DataFrame of all creatives and interactions
    d1 = start date
    d2 = end date

    Outputs:
    DataFrame with all "mis-matches" with impressions greater than imp_thresh

    """
    storage=[]

    metric_dict= {
        'branded driver':
            ['dfp_clicks', 'dfp_impressions'],
        'traffic driver':
            ['dfp_clicks', 'dfp_impressions'],
        'video autoplay':
            ['dfp_clicks', 'dfp_impressions'],
        'co-branded driver':
            ['dfp_clicks', 'dfp_impressions'],
        'video':
            ['result_5', 'dfp_impressions'],
        'interactive non video':
            ['int_sessions','dfp_impressions'],
        'brand survey':
            ['int_sessions','dfp_impressions'],
        'interactive video':
            ['result_5','dfp_impressions'],
        'no_match':
            ['dfp_clicks', 'dfp_impressions']
    }


    df = df[(df['date'] >= d1) & (df['date'] <= d2)]


    for creative_type in set(df['creative_type']):
        groupons = ['advertiser', 'placement', 'creative_name_version', 'site', 'creative_type']
        if creative_type == 'interactive non video' or 'survey' in creative_type:

            metrics = metric_dict[creative_type]

            dfx = df[(df['creative_type'] == creative_type)]
            dfx = dfx.groupby(groupons, as_index=False)[metrics].sum()
            dfx = dfx[dfx['dfp_impressions'] >= imp_thresh]
            dfx = dfx[dfx['int_sessions'].isnull()].copy()

            if dfx.empty:
                print('no '+creative_type+' mismatches')

            dfx['mis_match'] = 'no_kpi_actions'
            dfx = dfx.sort_values('dfp_impressions', ascending=False)
            del dfx['int_sessions']
            storage.append(dfx)


        elif 'no_match' in creative_type:

            metrics = metric_dict[creative_type]
            groupons = ['advertiser', 'placement', 'site', 'creative_type']

            dfx = df[(df['creative_type'] == creative_type)].copy()
            dfx = dfx.groupby(groupons, as_index=False)[metrics].sum()
            dfx = dfx[dfx['dfp_impressions'] >= imp_thresh]
            dfx['creative_name_version'] = 'no_match'
            dfx = dfx[dfx['dfp_clicks']==0].copy()

            if dfx.empty:
               print('no '+creative_type+' mismatches')


            dfx['mis_match'] = 'no_clicks'
            dfx = dfx.sort_values('dfp_impressions', ascending=False)
            del dfx['dfp_clicks']
            storage.append(dfx)


        elif 'driver' in creative_type or 'autoplay' in creative_type:
            metrics = metric_dict[creative_type]

            dfx = df[(df['creative_type'] == creative_type)]
            dfx = dfx.groupby(groupons, as_index=False)[metrics].sum()
            dfx = dfx[dfx['dfp_impressions'] >= imp_thresh]
            dfx = dfx[dfx['dfp_clicks']==0].copy()

            if dfx.empty:
               print('no '+creative_type+' mismatches')

            dfx['mis_match'] = 'no_kpi_actions'
            dfx = dfx.sort_values('dfp_impressions', ascending=False)
            del dfx['dfp_clicks']
            storage.append(dfx)

        elif 'video' in creative_type:
            metrics = metric_dict[creative_type]

            dfx = df[(df['creative_type'] == creative_type)]
            dfx = dfx.groupby(groupons, as_index=False)[metrics].sum()
            dfx = dfx[dfx['dfp_impressions'] >= imp_thresh]
            dfx = dfx[dfx['result_5'].isnull()].copy()

            if dfx.empty:
               print('no '+creative_type+' mismatches')

            dfx['mis_match'] = 'no_kpi_actions'
            dfx = dfx.sort_values('dfp_impressions', ascending=False)
            del dfx['result_5']
            storage.append(dfx)





    df_all = pd.concat(storage)
    df_all=df_all.sort_values('dfp_impressions',ascending=False)
    col_order=['advertiser', 'site','creative_name_version','placement',
       'creative_type', 'dfp_impressions', 'mis_match']
    df_all=df_all[col_order]
    df_all = df_all.reset_index(drop=True)
    return df_all

def get_n_colors(n, colormap):
    slices = [int(256*i/n) for i in range(n)]
    return [colormap[256][i] for i in slices]

def load_hoon_data(days=60):
    yesterday = datetime.datetime.now() - datetime.timedelta(1)
    t_n = yesterday - datetime.timedelta(days)

    yesterday = datetime.datetime.strftime(yesterday, '%Y-%m-%d')
    t_n = datetime.datetime.strftime(t_n, '%Y-%m-%d')

    url_endpoint = 'http://analytics.qz.com/api/ads/csv'
    mydict = {'startDate': t_n, 'endDate': yesterday, 'type':'display'}
    response = requests.get(url_endpoint, params=mydict, stream=True)

    data = response.json()
    df = pd.DataFrame(data)
    df.columns = df.loc[0]
    df = df.loc[1:]
    df = df.reset_index(drop=True)

    # set None creative_types to 'no match'
    null_indices = np.where(df["creative_type"].isnull())[0]
    df.loc[null_indices, 'creative_type'] = 'no_match'

    #make null versions an empty string
    null_indices = np.where(df["version"] == 'null')[0]
    df.loc[null_indices, 'version'] = ''

    #create creative_name_version
    df['creative_name'] = df['creative_name'].apply(lambda x: str(x))
    df['creative_name_version'] = df['creative_name'] + '_' + df['version']

    #create creative_type
    df['creative_type'] = df['creative_type'].astype(str)
    df['creative_type'] = df['creative_type'].apply(lambda x: 'no_match' if x == '0' else x)

    #make date a datetime object
    df['date'] = pd.to_datetime(df['date'])

    #make adunit
    placement_to_adunit = {
        'Atlas':'atlas',
        'ICP desktop': 'icp',
        'ICP mobile': 'icp',
        'ICP oth': 'icp',
        'ICP tablet': 'icp',
        'Out-of-page': 'oop',
        'engage desktop':'engage',
        'engage mobile':'engage',
        'engage oth': 'engage',
        'engage tablet':'engage',
        'inline desktop':'inline',
        'inline mobile':'inline',
        'inline oth':'inline',
        'inline tablet':'inline',
        'marquee desktop':'marquee',
        'marquee mobile':'marquee',
        'marquee oth':'marquee',
        'marquee tablet':'marquee',
        'no match desktop':'no match',
        'no match mobile': 'no match',
        'no match tablet': 'no match',
        'oth': 'oth',
        'spotlight desktop':'spotlight',
        'spotlight mobile':'spotlight',
        'spotlight tablet': 'spotlight'
    }

    df['adunit'] = df['placement'].map(placement_to_adunit)

    dates = list(pd.to_datetime(df['date']))
    dates = sorted(dates)
    date_set = set(dates[0] + datetime.timedelta(x) for x in range((dates[-1] - dates[0]).days))
    missing = sorted(date_set - set(dates))
    print('missing dates over range:', missing)

    return df

def bm_sub(dfh, df_bm, d1, d2, min_imps=1000):
    metric_dict= {
            'branded driver': 'CTR',
            'traffic driver': 'CTR',
            'video autoplay': 'VID',
            'co-branded driver': 'CTR',
            'video': 'VID',
            'interactive non video': 'IR',
            'brand survey': 'IR',
            'interactive video': 'IR',
            'no_match': 'CTR'
    }

    groupons = ('advertiser', 'placement', 'creative_name_version', 'creative_type', 'site')
    x = dfh[(dfh['date'] >= d1) & (dfh['date'] <= d2)]

    x = x.groupby(groupons)[['dfp_impressions', 'dfp_clicks', 'int_sessions', 'result_5']].sum()
    x = x[x['dfp_impressions'] >= min_imps].copy()

    x = x.reset_index()

    x['KPI_type'] = x['creative_type'].map(metric_dict)

    # place the KPI's based upon the KPI_type
    x['metric'] = 0
    index = np.where(x["KPI_type"] == 'CTR')[0]
    x.loc[index, 'metric'] = x.loc[index, 'dfp_clicks'] / x.loc[index, 'dfp_impressions']

    index = np.where(x["KPI_type"] == 'IR')[0]
    x.loc[index, 'metric'] = x.loc[index, 'int_sessions'] / x.loc[index, 'dfp_impressions']

    index = np.where(x["KPI_type"] == 'VID')[0]
    x.loc[index, 'metric'] = x.loc[index, 'int_sessions'] / x.loc[index, 'dfp_impressions']

    #load in the benchmark data
    dfbm = df_bm.copy()
    dfbm['placement'] = dfbm['Placement'].str.lower()
    dfbm = dfbm[dfbm['Data Source'] == 'DFP']
    dfbm = dfbm[['KPI', 'placement', '1H2017 BM']]

    #merge
    z = pd.merge(x, dfbm, left_on=('KPI_type', 'placement'), right_on=('KPI', 'placement'), how='left')

    z['delta'] = (z['metric'] - z['1H2017 BM']) * 100
    z['delta_pct'] = (z['delta'] / z['1H2017 BM']) * 100
    z['metric_prty'] = z['metric'] * 100
    return z
