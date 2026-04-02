import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set paths
base_path = r'd:\SLIIT 4 Y 2S\ML\Project\ML-assignment'
energy_path = os.path.join(base_path, 'data', 'energy_dataset.csv')
weather_path = os.path.join(base_path, 'data', 'weather_features.csv')

def load_and_preprocess():
    print("Loading data...")
    energy_df = pd.read_csv(energy_path, parse_dates=['time'])
    weather_df = pd.read_csv(weather_path, parse_dates=['dt_iso'])
    
    # Preprocessing logic from the notebook
    weather_agg = (
        weather_df
        .rename(columns={'dt_iso': 'time'})
        .groupby('time')[['temp', 'wind_speed', 'clouds_all']]
        .mean()
        .reset_index()
    )
    
    energy_df['time'] = pd.to_datetime(energy_df['time'], utc=True).dt.tz_localize(None)
    weather_agg['time'] = pd.to_datetime(weather_agg['time'], utc=True).dt.tz_localize(None)
    
    df = energy_df.merge(weather_agg, on='time', how='left')
    
    core_cols = [
        'time',
        'price actual',
        'total load actual',
        'generation solar',
        'generation wind onshore',
        'temp',
    ]
    df = df[core_cols].copy()
    df.columns = ['time', 'price', 'load', 'solar', 'wind', 'temp']
    
    df.dropna(subset=['price', 'load'], inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)
    
    df['renewable_gen'] = df['solar'] + df['wind']
    df['net_load'] = (df['load'] - df['renewable_gen']).clip(lower=0)
    df['hour'] = df['time'].dt.hour
    
    return df

def visualize(df):
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Energy Price Trend (First week of 2015)
    plt.subplot(2, 1, 1)
    mask = (df['time'] >= '2015-01-01') & (df['time'] < '2015-01-08')
    sns.lineplot(data=df[mask], x='time', y='price', color='blue')
    plt.title('Daily Electricity Price Variations (€/MWh) - First Week 2015')
    plt.ylabel('Price')

    # Plot 2: Average Net Load vs Hour of Day
    plt.subplot(2, 1, 2)
    hourly_avg = df.groupby('hour')['net_load'].mean().reset_index()
    sns.barplot(data=hourly_avg, x='hour', y='net_load', palette='viridis')
    plt.title('Average Net Load by Hour of Day (Total Load - Renewables)')
    plt.ylabel('Load (MW)')
    
    plt.tight_layout()
    plot_path = os.path.join(base_path, 'visuals', 'eda_verification.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    print(f"Visualization saved to: {plot_path}")

if __name__ == "__main__":
    df = load_and_preprocess()
    print("Preprocessed Data Summary:")
    print(df.info())
    print("\nSample Data:")
    print(df.head())
    visualize(df)
