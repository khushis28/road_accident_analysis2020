
# TASK 2: INITIAL DATA EXPLORATION


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("2. INITIAL DATA EXPLORATION")
print("="*60)


try:
    df = pd.read_pickle('data/df_loaded.pkl')
    print("Dataset loaded from pickle file!")
except FileNotFoundError:
    try:
        df = pd.read_csv('data/Road_Accident_Data_2020.csv')
        print("Dataset loaded from CSV file!")
    except FileNotFoundError:
        print("Error: Dataset not found. Please run task1_data_loading.py first.")
        exit()


print("\n2.1 Dataset Overview:")
print(f"Number of rows: {len(df)}")
print(f"Number of columns: {len(df.columns)}")
print(f"Column names: {list(df.columns)}")


print("\n2.2 First 5 rows:")
print(df.head())


print("\n2.3 Last 5 rows:")
print(df.tail())


print("\n2.4 Dataset Info:")
print(df.info())


print("\n2.5 Statistical Summary:")
print(df.describe())


print("\n2.6 Unique Values per Column:")
for col in df.columns:
    if col != 'Count':
        unique_count = df[col].nunique()
        print(f"{col}: {unique_count} unique values")
        if unique_count <= 10:
            print(f"  Values: {list(df[col].unique())}")
    else:
        print(f"{col}: Numerical column - Min: {df[col].min()}, Max: {df[col].max()}")
    print()


print("\n2.7 Memory Usage:")
print(f"Total memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")


print("\n2.8 Quick Data Distribution:")
print("\nCities with most records:")
print(df['Million Plus Cities'].value_counts().head(10))

print("\nCause categories distribution:")
print(df['Cause category'].value_counts())

print("\nOutcome types distribution:")
print(df['Outcome of Incident'].value_counts())

print("\n" + "="*60)
print("TASK 2 COMPLETED SUCCESSFULLY!")
print("="*60)
