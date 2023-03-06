import streamlit as st
import pandas as pd
import numpy as np
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')
from bokeh.models import HoverTool
from sklearn.ensemble import IsolationForest

file = st.sidebar.file_uploader('Select a csv file', type={"csv", "txt"})

def readCSV(file):
    df = pd.read_csv(file, parse_dates=['timestamp'])
    df['category'] = np.random.randint(4, size=(df['value'].size))
    return df

def filter(st, df):
    category = st.sidebar.number_input('Category', 0, 3)
    year = st.sidebar.number_input('Year', min(df['timestamp'].dt.year), max(df['timestamp'].dt.year))
    return df.loc[(df['category'] == category) & (df['timestamp'].dt.year == year), ['timestamp', 'value']]

def overview():
    Hourly = hv.Curve(df.set_index('timestamp').resample('H').mean()).opts(
        opts.Curve(title="New York City Taxi Demand Hourly", xlabel="", ylabel="Demand",
                width=700, height=300,tools=['hover'],show_grid=True))

    Daily = hv.Curve(df.set_index('timestamp').resample('D').mean()).opts(
        opts.Curve(title="New York City Taxi Demand Daily", xlabel="", ylabel="Demand",
                width=700, height=300,tools=['hover'],show_grid=True))

    Weekly = hv.Curve(df.set_index('timestamp').resample('W').mean()).opts(
        opts.Curve(title="New York City Taxi Demand Weekly", xlabel="Date", ylabel="Demand",
                width=700, height=300,tools=['hover'],show_grid=True))


    st.bokeh_chart(hv.render((Hourly + Daily + Weekly).opts(shared_axes=False).cols(1)))

threshold = 0.0

if file is not None:
    df = readCSV(file)
    df = filter(st, df)
    threshold = st.sidebar.slider('Select the threshold', 0.0, 1.0)
    print('thres: ', threshold)

if st.sidebar.button("APPLY"):
    if 'df' in locals():
        overview()

        # Feature Engineering ----------------------------------------------------------------------------------------------------------
        df_hourly = df.set_index('timestamp').resample('H').mean().reset_index()
        df_daily = df.set_index('timestamp').resample('D').mean().reset_index()
        df_weekly = df.set_index('timestamp').resample('W').mean().reset_index()

        for DataFrame in [df_hourly, df_daily]:
            DataFrame['Weekday'] = (pd.Categorical(DataFrame['timestamp'].dt.strftime('%A'),
                                                categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday','Friday', 'Saturday', 'Sunday'])
                                )
            DataFrame['Hour'] = DataFrame['timestamp'].dt.hour
            DataFrame['Day'] = DataFrame['timestamp'].dt.weekday
            DataFrame['Month'] = DataFrame['timestamp'].dt.month
            DataFrame['Year'] = DataFrame['timestamp'].dt.year
            DataFrame['Month_day'] = DataFrame['timestamp'].dt.day
            DataFrame['Lag'] = DataFrame['value'].shift(1)
            DataFrame['Rolling_Mean'] = DataFrame['value'].rolling(7, min_periods=1).mean()
            DataFrame = DataFrame.dropna()


        # More Visual Exploration -------------------------------------------------------------------------------------------------------
        
        # Overall Value Distribution
        st.bokeh_chart(hv.render((hv.Distribution(df['value'])
            .opts(opts.Distribution(title="Overall Value Distribution",
                                    xlabel="Value",
                                    ylabel="Density",
                                    width=700, height=300,
                                    tools=['hover'],show_grid=True)
                ))))

        # Demand Density by Day & Hour
        by_weekday = df_hourly.groupby(['Hour','Weekday']).mean()['value'].unstack()
        plot = hv.Distribution(by_weekday['Monday'], label='Monday') * hv.Distribution(by_weekday['Tuesday'], label='Tuesday') * hv.Distribution(by_weekday['Wednesday'], label='Wednesday') * hv.Distribution(by_weekday['Thursday'], label='Thursday') * hv.Distribution(by_weekday['Friday'], label='Friday') * hv.Distribution(by_weekday['Saturday'], label='Saturday') *hv.Distribution(by_weekday['Sunday'], label='Sunday').opts(opts.Distribution(title="Demand Density by Day & Hour"))
        st.bokeh_chart(hv.render(plot.opts(opts.Distribution(width=800, height=300,tools=['hover'],show_grid=True, ylabel="Demand", xlabel="Demand"))))

        # New York City Taxi Demand by Day
        st.bokeh_chart(hv.render(hv.Bars(df_hourly[['value','Weekday']].groupby('Weekday').mean()).opts(
            opts.Bars(title="New York City Taxi Demand by Day", xlabel="", ylabel="Demand",
                    width=700, height=300,tools=['hover'],show_grid=True))))

        # New York City Taxi Demand Hourly
        st.bokeh_chart(hv.render(hv.Curve(df_hourly[['value','Hour']].groupby('Hour').mean()).opts(
            opts.Curve(title="New York City Taxi Demand Hourly", xlabel="Hour", ylabel="Demand",
                    width=700, height=300,tools=['hover'],show_grid=True))))

        # Average Demand by Day & Hour
        by_weekday = df_hourly.groupby(['Hour','Weekday']).mean()['value'].unstack()
        plot = hv.Curve(by_weekday['Monday'], label='Monday') * hv.Curve(by_weekday['Tuesday'], label='Tuesday') * hv.Curve(by_weekday['Wednesday'], label='Wednesday') * hv.Curve(by_weekday['Thursday'], label='Thursday') * hv.Curve(by_weekday['Friday'], label='Friday') * hv.Curve(by_weekday['Saturday'], label='Saturday') *hv.Curve(by_weekday['Sunday'], label='Sunday').opts(opts.Curve(title="Average Demand by Day & Hour"))
        st.bokeh_chart(hv.render(plot.opts(opts.Curve(width=800, height=300,tools=['hover'],show_grid=True, ylabel="Demand"))))

        # More Feature Engineering-----------------------------------------------------------------------------------------------------------

        df_hourly = (df_hourly
                    .join(df_hourly.groupby(['Hour','Weekday'])['value'].mean(),
                        on = ['Hour', 'Weekday'], rsuffix='_Average')
                    )

        df_daily = (df_daily
                    .join(df_daily.groupby(['Hour','Weekday'])['value'].mean(),
                        on = ['Hour', 'Weekday'], rsuffix='_Average')
                )
        
        # Average Saturday vs Busiest Saturday
        sat_max = (df_hourly
                .query("Day == 5")
                .set_index('timestamp')
                .loc['2015-01-31':'2015-01-31']
                .reset_index()['value']
                )


        avg_sat = (df_hourly
                .groupby(['Weekday','Hour'])['value']
                .mean()
                .unstack()
                .T['Saturday']
                )

        avg_max_comparison = hv.Curve(avg_sat, label='Average Saturday') * hv.Curve(sat_max, label='Busiest Saturday').opts(opts.Curve(title="Average Saturday vs Busiest Saturday"))
        st.bokeh_chart(hv.render(avg_max_comparison.opts(opts.Curve(width=800, height=300,tools=['hover'],show_grid=True, ylabel="Demand", show_legend=False))))

        # Models----------------------------------------------------------------------------------------------------------------------------

        # Choose Features for model
        df_hourly.dropna(inplace=True)
        df_daily_model_data = df_daily[['value', 'Hour', 'Day',  'Month','Month_day','Rolling_Mean']].dropna()
        model_data = df_hourly[['value', 'Hour', 'Day', 'Month_day', 'Month','Rolling_Mean','Lag', 'timestamp']].set_index('timestamp').dropna()

        # Fit Model & View Outliers
        def run_isolation_forest(model_data: pd.DataFrame, contamination=0.005, n_estimators=200, max_samples=0.7) -> pd.DataFrame:
            IF = (IsolationForest(random_state=0,
                                contamination=contamination,
                                n_estimators=n_estimators,
                                max_samples=max_samples)
                )
            
            IF.fit(model_data)
            
            output = pd.Series(IF.predict(model_data)).apply(lambda x: 1 if x == -1 else 0)
            
            score = IF.decision_function(model_data)
            
            return output, score
        
        outliers, score = run_isolation_forest(model_data)
        df_hourly = (df_hourly
                    .assign(Outliers = outliers)
                    .assign(Score = score)
                    )
        
        IF = IsolationForest(random_state=0, contamination=0.005, n_estimators=200, max_samples=0.7)
        IF.fit(model_data)

        # New Outliers Column
        df_hourly['Outliers'] = pd.Series(IF.predict(model_data)).apply(lambda x: 1 if x == -1 else 0)

        # Get Anomaly Score
        score = IF.decision_function(model_data)

        # New Anomaly Score column
        df_hourly['Score'] = score

        # Viewing the Anomalies--------------------------------------------------------------------------------------------------------------

        def outliers(thresh):
            print(f'Number of Outliers below Anomaly Score Threshold {thresh}:')
            print(len(df_hourly.query(f"Outliers == 1 & Score <= {thresh}")))


        # New York City Taxi Demand Anomalies
        tooltips = [
            ('Weekday', '@Weekday'),
            ('Day', '@Month_day'),
            ('Month', '@Month'),
            ('Value', '@value'),
            ('Average Value', '@value_Average'),
            ('Outliers', '@Outliers'),
            ('Score', '@Score')
        ]
        hover = HoverTool(tooltips=tooltips)

        st.bokeh_chart(hv.render(hv.Points(df_hourly.query("Outliers == 1")).opts(size=10, color='#ff0000') * hv.Curve(df_hourly).opts(opts.Curve(title="New York City Taxi Demand Anomalies", xlabel="", ylabel="Demand" , height=300, responsive=True,tools=[hover,'box_select', 'lasso_select', 'tap'], show_grid=True))))

        # Assessing Outliers------------------------------------------------------------------------------------------------------------------

        frequencies, edges = np.histogram(score, 50)
        st.bokeh_chart(hv.render(hv.Histogram((edges, frequencies)).opts(width=800, height=300,tools=['hover'], xlabel='Score')))

        hover = HoverTool(tooltips=tooltips)

        # New York City Taxi Demand
        print(threshold)
        st.bokeh_chart(hv.render(hv.Points(df_hourly.query(f"Outliers == 1 & Score <= {threshold}")).opts(size=10, color='#ff0000') * hv.Curve(df_hourly).opts(opts.Curve(title="New York City Taxi Demand", xlabel="", ylabel="Demand" , height=300, responsive=True,tools=[hover,'box_select', 'lasso_select', 'tap'],show_grid=True))))
    else:
        st.sidebar.error('Please select a file', icon='🚨')