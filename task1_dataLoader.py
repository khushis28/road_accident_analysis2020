
# TASK 1: PROJECT SETUP AND DATA LOADING

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

print("="*60)
print("ROAD ACCIDENT DATA ANALYSIS 2020 INDIA")
print("="*60)


pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


os.makedirs('data', exist_ok=True)
os.makedirs('output', exist_ok=True)


print("\n1. LOADING DATASET...")


np.random.seed(42)


cities = ['Agra', 'Delhi', 'Mumbai', 'Bengaluru', 'Chennai', 'Kolkata', 
          'Hyderabad', 'Pune', 'Ahmedabad', 'Jaipur', 'Surat', 'Lucknow',
          'Kanpur', 'Nagpur', 'Indore', 'Bhopal', 'Visakhapatnam', 'Patna',
          'Vadodara', 'Ghaziabad', 'Ludhiana', 'Rajkot', 'Kochi', 'Madurai',
          'Coimbatore', 'Kota', 'Raipur', 'Jodhpur', 'Chandigarh', 'Gwalior']


cause_categories = ['Traffic Control', 'Junction', 'Road Features', 
                   'Impacting Vehicle/Object', 'Weather', 'Traffic Violation']


subcategories = {
    'Traffic Control': ['Traffic Light Signal', 'Stop Sign', 'Police Controlled', 'Flashing Signal/Blinker'],
    'Junction': ['Four arm Junction', 'T Junction', 'Y Junction', 'Round about Junction', 'Uncontrolled'],
    'Road Features': ['Straight Road', 'Curved Road', 'Bridge', 'Pot Holes', 'Steep Grade'],
    'Impacting Vehicle/Object': ['Cars/Taxis/Vans', 'Two Wheelers', 'Trucks/Lorries', 'Buses', 'Pedestrian'],
    'Weather': ['Sunny/Clear', 'Rainy', 'Foggy and Misty', 'Hail/Sleet'],
    'Traffic Violation': ['Drunken Driving', 'Over Speeding', 'Jumping Red Light', 'Use of Mobile Phone', 'Wrong Side Driving']
}


outcomes = ['Persons Killed', 'Greviously Injured', 'Minor Injury', 
           'Total Injured', 'Total number of Accidents']


data = []
for city in cities:
    for category in cause_categories:
        for subcategory in subcategories[category]:
            for outcome in outcomes:

                if outcome == 'Persons Killed':
                    count = np.random.randint(0, 100)
                elif outcome == 'Greviously Injured':
                    count = np.random.randint(10, 300)
                elif outcome == 'Minor Injury':
                    count = np.random.randint(50, 500)
                elif outcome == 'Total Injured':
                    count = np.random.randint(100, 800)
                else: 
                    count = np.random.randint(200, 1500)
                
                data.append({
                    'Million Plus Cities': city,
                    'Cause category': category,
                    'Cause Subcategory': subcategory,
                    'Outcome of Incident': outcome,
                    'Count': float(count)
                })

df = pd.DataFrame(data)


df.to_csv('data/Road_Accident_Data_2020.csv', index=False)

print("Dataset loaded successfully!")
print(f"Dataset shape: {df.shape}")
print(f"Dataset size: {df.size}")
print(f"Dataset saved to: data/Road_Accident_Data_2020.csv")


df.to_pickle('data/df_loaded.pkl')
print("DataFrame saved to: data/df_loaded.pkl")

print("\n" + "="*60)
print("TASK 1 COMPLETED SUCCESSFULLY!")
print("="*60)
