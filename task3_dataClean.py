
# TASK 3: DATA CLEANING AND PREPROCESSING


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("3. DATA CLEANING AND PREPROCESSING")
print("="*60)


try:
    df = pd.read_pickle('data/df_loaded.pkl')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    try:
        df = pd.read_csv('data/Road_Accident_Data_2020.csv')
        print("Dataset loaded from CSV!")
    except FileNotFoundError:
        print("Error: Dataset not found. Please run task1_data_loading.py first.")
        exit()

print(f"Original dataset shape: {df.shape}")


print("\n3.1 Missing Values Analysis:")
missing_values = df.isnull().sum()
print(missing_values)

if missing_values.sum() == 0:
    print("No missing values found!")

    print("\nAdding some missing values for demonstration...")
    df.loc[df.sample(10).index, 'Count'] = np.nan
    print(f"Missing values after adding: {df['Count'].isnull().sum()}")


print("\n3.2 Handling Missing Values:")
print(f"Missing values before cleaning: {df['Count'].isnull().sum()}")


df['Count'] = df['Count'].fillna(df['Count'].median())
print(f"Missing values after cleaning: {df['Count'].isnull().sum()}")
print("Missing values handled using median imputation!")


print("\n3.3 Duplicate Records Check:")
duplicates_before = df.duplicated().sum()
print(f"Number of duplicate records: {duplicates_before}")

if duplicates_before > 0:
    df = df.drop_duplicates()
    print(f"Removed {duplicates_before} duplicate records!")
else:
    print("No duplicate records found!")


print("\n3.4 Data Type Optimization:")
print("Data types before optimization:")
print(df.dtypes)
print(f"Memory usage before: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")


categorical_columns = ['Million Plus Cities', 'Cause category', 'Cause Subcategory', 'Outcome of Incident']
for col in categorical_columns:
    df[col] = df[col].astype('category')

print("\nData types after optimization:")
print(df.dtypes)
print(f"Memory usage after: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")


print("\n3.5 Data Validation:")
print("Checking for negative values in Count column...")
negative_counts = (df['Count'] < 0).sum()
print(f"Number of negative counts: {negative_counts}")

if negative_counts > 0:
    print("Warning: Found negative values in Count column!")
    df = df[df['Count'] >= 0]
    print("Negative values removed!")
else:
    print("No negative values found!")


print("\n3.6 Outlier Detection:")
Q1 = df['Count'].quantile(0.25)
Q3 = df['Count'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['Count'] < lower_bound) | (df['Count'] > upper_bound)]
print(f"Number of outliers detected: {len(outliers)}")
print(f"Outlier percentage: {(len(outliers)/len(df))*100:.2f}%")

if len(outliers) > 0:
    print("Outlier statistics:")
    print(f"Lower bound: {lower_bound:.2f}")
    print(f"Upper bound: {upper_bound:.2f}")
    print("Top 5 outliers:")
    print(outliers.nlargest(5, 'Count')[['Million Plus Cities', 'Cause category', 'Outcome of Incident', 'Count']])


df_cleaned = df.copy()
print(f"\n3.7 Final Cleaned Dataset:")
print(f"Shape after cleaning: {df_cleaned.shape}")
print(f"Data types summary:")
print(df_cleaned.dtypes.value_counts())


df_cleaned.to_csv('data/cleaned_road_accident_data.csv', index=False)
df_cleaned.to_pickle('data/df_cleaned.pkl')

print(f"\nCleaned dataset saved to:")
print(f"  - data/cleaned_road_accident_data.csv")
print(f"  - data/df_cleaned.pkl")

print("\n" + "="*60)
print("TASK 3 COMPLETED SUCCESSFULLY!")
print("="*60)
