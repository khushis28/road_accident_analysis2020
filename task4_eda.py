
# TASK 4: EXPLORATORY DATA ANALYSIS (EDA)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("4. EXPLORATORY DATA ANALYSIS (EDA)")
print("="*60)


try:
    df_cleaned = pd.read_pickle('data/df_cleaned.pkl')
    print("Cleaned dataset loaded successfully!")
except FileNotFoundError:
    try:
        df_cleaned = pd.read_csv('data/cleaned_Road_Accident_Data_2020.csv')
        print("Cleaned dataset loaded from CSV!")
    except FileNotFoundError:
        print("Error: Cleaned dataset not found. Please run task3_data_cleaning.py first.")
        exit()


plt.style.use('default')
sns.set_palette("husl")


print("\n4.1 City-wise Analysis:")
city_stats = df_cleaned.groupby('Million Plus Cities')['Count'].agg(['sum', 'mean', 'count']).round(2)
city_stats.columns = ['Total_Accidents', 'Average_Count', 'Number_of_Records']
city_stats = city_stats.sort_values('Total_Accidents', ascending=False)
print("\nTop 10 cities by total accident count:")
print(city_stats.head(10))


print("\n4.2 Cause Category Analysis:")
cause_stats = df_cleaned.groupby('Cause category')['Count'].agg(['sum', 'mean', 'count']).round(2)
cause_stats.columns = ['Total_Count', 'Average_Count', 'Number_of_Records']
cause_stats = cause_stats.sort_values('Total_Count', ascending=False)
print(cause_stats)


print("\n4.3 Outcome Analysis:")
outcome_stats = df_cleaned.groupby('Outcome of Incident')['Count'].agg(['sum', 'mean']).round(2)
outcome_stats.columns = ['Total_Count', 'Average_Count']
outcome_stats = outcome_stats.sort_values('Total_Count', ascending=False)
print(outcome_stats)


print("\n4.4 Top 15 Cause Subcategories:")
subcat_stats = df_cleaned.groupby('Cause Subcategory')['Count'].sum().sort_values(ascending=False)
print(subcat_stats.head(15))


print("\n4.5 Cross-tabulation: Cause Category vs Outcome")
cross_tab = pd.crosstab(df_cleaned['Cause category'], 
                       df_cleaned['Outcome of Incident'], 
                       values=df_cleaned['Count'], 
                       aggfunc='sum', 
                       margins=True).round(0)
print(cross_tab)


print("\n4.6 Statistical Analysis:")
print("Basic statistics for Count by Cause Category:")
for category in df_cleaned['Cause category'].unique():
    subset = df_cleaned[df_cleaned['Cause category'] == category]['Count']
    print(f"\n{category}:")
    print(f"  Mean: {subset.mean():.2f}")
    print(f"  Median: {subset.median():.2f}")
    print(f"  Std Dev: {subset.std():.2f}")
    print(f"  Min: {subset.min():.2f}")
    print(f"  Max: {subset.max():.2f}")


print("\n4.7 Saving Analysis Results:")
city_stats.to_csv('output/city_wise_analysis.csv')
cause_stats.to_csv('output/cause_wise_analysis.csv')
outcome_stats.to_csv('output/outcome_wise_analysis.csv')
subcat_stats.to_csv('output/subcategory_analysis.csv')
cross_tab.to_csv('output/cross_tabulation_analysis.csv')

print("Analysis results saved to output folder:")
print("  - city_wise_analysis.csv")
print("  - cause_wise_analysis.csv")
print("  - outcome_wise_analysis.csv")
print("  - subcategory_analysis.csv")
print("  - cross_tabulation_analysis.csv")

print("\n" + "="*60)
print("TASK 4 COMPLETED SUCCESSFULLY!")
print("="*60)
