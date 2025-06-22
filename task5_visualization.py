
# TASK 5: DATA VISUALIZATION


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_preprocessed_data():
    """
    Load the preprocessed data from previous tasks.
    If running standalone, create sample data.
    """
    try:
  
        df_cleaned = pd.read_csv('output/cleaned_data_task4.csv')
        print("Loaded preprocessed data from Task 4")
        return df_cleaned
    except FileNotFoundError:
        print("Creating sample data for standalone execution...")
        
     
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

        return pd.DataFrame(data)

def prepare_analysis_data(df_cleaned):
    """Prepare aggregated data for visualizations"""
    
    
    city_stats = df_cleaned.groupby('Million Plus Cities')['Count'].agg(['sum', 'mean', 'count']).round(2)
    city_stats.columns = ['Total_Accidents', 'Average_Count', 'Number_of_Records']
    city_stats = city_stats.sort_values('Total_Accidents', ascending=False)
    
    
    cause_stats = df_cleaned.groupby('Cause category')['Count'].agg(['sum', 'mean', 'count']).round(2)
    cause_stats.columns = ['Total_Count', 'Average_Count', 'Number_of_Records']
    cause_stats = cause_stats.sort_values('Total_Count', ascending=False)
    

    outcome_stats = df_cleaned.groupby('Outcome of Incident')['Count'].agg(['sum', 'mean']).round(2)
    outcome_stats.columns = ['Total_Count', 'Average_Count']
    outcome_stats = outcome_stats.sort_values('Total_Count', ascending=False)
    
   
    subcat_stats = df_cleaned.groupby('Cause Subcategory')['Count'].sum().sort_values(ascending=False)
    
    return city_stats, cause_stats, outcome_stats, subcat_stats

def create_comprehensive_visualization(df_cleaned, city_stats, cause_stats, outcome_stats, subcat_stats):
    """Create comprehensive visualization dashboard"""
    
    print("\n" + "="*60)
    print("5. DATA VISUALIZATION")
    print("="*60)
    
  
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    

    fig = plt.figure(figsize=(20, 15))
    
   
    plt.subplot(2, 3, 1)
    top_cities = city_stats.head(15)
    bars1 = plt.bar(range(len(top_cities)), top_cities['Total_Accidents'], 
                    color='skyblue', edgecolor='navy', alpha=0.7)
    plt.title('Top 15 Cities by Total Accident Count', fontsize=14, fontweight='bold')
    plt.xlabel('Cities', fontsize=12)
    plt.ylabel('Total Accident Count', fontsize=12)
    plt.xticks(range(len(top_cities)), top_cities.index, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    

    for i, bar in enumerate(bars1):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                 f'{int(height)}', ha='center', va='bottom', fontsize=8)
    

    plt.subplot(2, 3, 2)
    cause_counts = df_cleaned.groupby('Cause category')['Count'].sum().sort_values(ascending=True)
    bars2 = plt.barh(range(len(cause_counts)), cause_counts.values, 
                     color='lightcoral', edgecolor='darkred', alpha=0.7)
    plt.title('Accident Distribution by Cause Category', fontsize=14, fontweight='bold')
    plt.xlabel('Total Count', fontsize=12)
    plt.ylabel('Cause Category', fontsize=12)
    plt.yticks(range(len(cause_counts)), cause_counts.index)
    plt.grid(axis='x', alpha=0.3)
    

    for i, bar in enumerate(bars2):
        width = bar.get_width()
        plt.text(width + width*0.01, bar.get_y() + bar.get_height()/2.,
                 f'{int(width)}', ha='left', va='center', fontsize=10)
    

    plt.subplot(2, 3, 3)
    outcome_counts = df_cleaned.groupby('Outcome of Incident')['Count'].sum()
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
    wedges, texts, autotexts = plt.pie(outcome_counts.values, labels=outcome_counts.index, 
                                       autopct='%1.1f%%', colors=colors, startangle=90)
    plt.title('Distribution of Accident Outcomes', fontsize=14, fontweight='bold')
    

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
 
    plt.subplot(2, 3, 4)
    top_subcats = subcat_stats.head(15)
    bars4 = plt.bar(range(len(top_subcats)), top_subcats.values, 
                    color='gold', edgecolor='orange', alpha=0.8)
    plt.title('Top 15 Accident Cause Subcategories', fontsize=14, fontweight='bold')
    plt.xlabel('Cause Subcategories', fontsize=12)
    plt.ylabel('Total Count', fontsize=12)
    plt.xticks(range(len(top_subcats)), top_subcats.index, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    

    plt.subplot(2, 3, 5)
    heatmap_data = df_cleaned.groupby(['Million Plus Cities', 'Cause category'])['Count'].sum().unstack(fill_value=0)
    top_15_cities = city_stats.head(15).index
    heatmap_subset = heatmap_data.loc[top_15_cities]
    
    sns.heatmap(heatmap_subset, annot=True, fmt='.0f', cmap='YlOrRd', 
                cbar_kws={'label': 'Total Count'})
    plt.title('Accident Heatmap: Top 15 Cities vs Cause Categories', fontsize=14, fontweight='bold')
    plt.xlabel('Cause Category', fontsize=12)
    plt.ylabel('Cities', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
   
    plt.subplot(2, 3, 6)
    outcome_order = outcome_stats.index
    box_data = [df_cleaned[df_cleaned['Outcome of Incident'] == outcome]['Count'].values 
                for outcome in outcome_order]
    
    bp = plt.boxplot(box_data, labels=outcome_order, patch_artist=True)
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.title('Distribution of Counts by Outcome Type', fontsize=14, fontweight='bold')
    plt.xlabel('Outcome of Incident', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def create_additional_visualizations(df_cleaned):
    """Create additional focused visualizations"""
    
    print("\nCreating Additional Focused Visualizations...")
    
  
    fig2, axes = plt.subplots(2, 2, figsize=(16, 12))
    
 
    city_totals = df_cleaned.groupby('Million Plus Cities')['Count'].sum().sort_values(ascending=False).head(10)
    axes[0, 0].bar(range(len(city_totals)), city_totals.values, color='steelblue', alpha=0.8)
    axes[0, 0].set_title('Top 10 Cities - Total Accident Count', fontweight='bold')
    axes[0, 0].set_xticks(range(len(city_totals)))
    axes[0, 0].set_xticklabels(city_totals.index, rotation=45, ha='right')
    axes[0, 0].grid(axis='y', alpha=0.3)
    

    cause_totals = df_cleaned.groupby('Cause category')['Count'].sum().sort_values(ascending=True)
    axes[0, 1].barh(range(len(cause_totals)), cause_totals.values, color='coral', alpha=0.8)
    axes[0, 1].set_title('Accident Count by Cause Category', fontweight='bold')
    axes[0, 1].set_yticks(range(len(cause_totals)))
    axes[0, 1].set_yticklabels(cause_totals.index)
    axes[0, 1].grid(axis='x', alpha=0.3)
    

    severity_order = ['Minor Injury', 'Greviously Injured', 'Persons Killed', 'Total Injured', 'Total number of Accidents']
    severity_data = df_cleaned[df_cleaned['Outcome of Incident'].isin(severity_order)]
    severity_counts = severity_data.groupby('Outcome of Incident')['Count'].sum()
    
    axes[1, 0].pie(severity_counts.values, labels=severity_counts.index, 
                   autopct='%1.1f%%', startangle=90, colors=plt.cm.Set3.colors)
    axes[1, 0].set_title('Accident Severity Distribution', fontweight='bold')
    
    
    top_subcats = df_cleaned.groupby('Cause Subcategory')['Count'].sum().sort_values(ascending=False).head(10)
    axes[1, 1].bar(range(len(top_subcats)), top_subcats.values, color='gold', alpha=0.8)
    axes[1, 1].set_title('Top 10 Specific Causes', fontweight='bold')
    axes[1, 1].set_xticks(range(len(top_subcats)))
    axes[1, 1].set_xticklabels(top_subcats.index, rotation=45, ha='right')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig2

def save_visualization_data(city_stats, cause_stats, outcome_stats, subcat_stats):
    """Save processed data for next tasks"""
    
    import os
    if not os.path.exists('output'):
        os.makedirs('output')
    

    city_stats.to_csv('output/city_analysis_task5.csv')
    cause_stats.to_csv('output/cause_analysis_task5.csv')
    outcome_stats.to_csv('output/outcome_analysis_task5.csv')
    subcat_stats.to_csv('output/subcategory_analysis_task5.csv')
    
    print("\nVisualization data saved to output folder!")

def main():
    """Main execution function for Task 5"""
    
    print("="*60)
    print("TASK 5: DATA VISUALIZATION")
    print("Road Accident Data Analysis 2020 India")
    print("="*60)
    

    df_cleaned = load_preprocessed_data()
    print(f"Dataset loaded: {df_cleaned.shape[0]} rows, {df_cleaned.shape[1]} columns")
    

    city_stats, cause_stats, outcome_stats, subcat_stats = prepare_analysis_data(df_cleaned)
    

    fig1 = create_comprehensive_visualization(df_cleaned, city_stats, cause_stats, outcome_stats, subcat_stats)
    

    fig2 = create_additional_visualizations(df_cleaned)
    

    save_visualization_data(city_stats, cause_stats, outcome_stats, subcat_stats)
    
    print("\nTask 5 completed successfully!")
    print("All visualizations created and data saved for next tasks.")

if __name__ == "__main__":
    main()