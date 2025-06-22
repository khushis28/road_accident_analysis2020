
# TASK 6: ADVANCED ANALYSIS


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

def load_data_from_previous_tasks():
    """Load data from previous tasks"""
    try:

        df_cleaned = pd.read_csv('output/cleaned_data_task4.csv')
        

        city_stats = pd.read_csv('output/city_analysis_task5.csv', index_col=0)
        cause_stats = pd.read_csv('output/cause_analysis_task5.csv', index_col=0)
        outcome_stats = pd.read_csv('output/outcome_analysis_task5.csv', index_col=0)
        subcat_stats = pd.read_csv('output/subcategory_analysis_task5.csv', index_col=0)
        
        print("Loaded data from previous tasks")
        return df_cleaned, city_stats, cause_stats, outcome_stats, subcat_stats
        
    except FileNotFoundError:
        print("Previous task data not found. Creating sample data...")
        return create_sample_data()

def create_sample_data():
    """Create sample data for standalone execution"""
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

    df_cleaned = pd.DataFrame(data)
    
   
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
    
    return df_cleaned, city_stats, cause_stats, outcome_stats, subcat_stats

def perform_cross_tabulation_analysis(df_cleaned):
    """Perform cross-tabulation analysis"""
    
    print("\n6.1 CROSS-TABULATION ANALYSIS")
    print("-" * 40)
    

    print("Cross-tabulation: Cause Category vs Outcome")
    cross_tab = pd.crosstab(df_cleaned['Cause category'], 
                           df_cleaned['Outcome of Incident'], 
                           values=df_cleaned['Count'], 
                           aggfunc='sum', 
                           margins=True).round(2)
    print(cross_tab)
    

    print("\nCross-tabulation: Top 10 Cities vs Cause Categories")
    top_10_cities = df_cleaned.groupby('Million Plus Cities')['Count'].sum().nlargest(10).index
    city_cause_cross = pd.crosstab(df_cleaned[df_cleaned['Million Plus Cities'].isin(top_10_cities)]['Million Plus Cities'], 
                                   df_cleaned[df_cleaned['Million Plus Cities'].isin(top_10_cities)]['Cause category'], 
                                   values=df_cleaned[df_cleaned['Million Plus Cities'].isin(top_10_cities)]['Count'], 
                                   aggfunc='sum').round(2)
    print(city_cause_cross)
    
    return cross_tab, city_cause_cross

def analyze_risky_combinations(df_cleaned):
    """Analyze risky combinations of factors"""
    
    print("\n6.2 RISKY COMBINATIONS ANALYSIS")
    print("-" * 40)
    
   
    print("Top 15 Risky Combinations (Category + Subcategory + Outcome):")
    risky_combinations = df_cleaned.groupby(['Cause category', 'Cause Subcategory', 'Outcome of Incident'])['Count'].sum()
    risky_combinations = risky_combinations.sort_values(ascending=False).head(15)
    
    for i, (combination, count) in enumerate(risky_combinations.items(), 1):
        print(f"{i:2d}. {combination[0]} → {combination[1]} → {combination[2]}: {count:,.0f}")
    
    
    print("\nTop 10 Deadly Combinations (Persons Killed):")
    deadly_data = df_cleaned[df_cleaned['Outcome of Incident'] == 'Persons Killed']
    deadly_combinations = deadly_data.groupby(['Cause category', 'Cause Subcategory'])['Count'].sum()
    deadly_combinations = deadly_combinations.sort_values(ascending=False).head(10)
    
    for i, (combination, count) in enumerate(deadly_combinations.items(), 1):
        print(f"{i:2d}. {combination[0]} → {combination[1]}: {count:,.0f} deaths")
    
    return risky_combinations, deadly_combinations

def analyze_city_safety_metrics(df_cleaned):
    """Analyze city safety metrics"""
    
    print("\n6.3 CITY SAFETY METRICS")
    print("-" * 40)
    

    print("Top 15 Cities by Fatalities (Persons Killed):")
    fatality_data = df_cleaned[df_cleaned['Outcome of Incident'] == 'Persons Killed']
    city_fatalities = fatality_data.groupby('Million Plus Cities')['Count'].sum().sort_values(ascending=False)
    print(city_fatalities.head(15))
    

    print("\nTop 15 Cities by Total Injuries:")
    injury_data = df_cleaned[df_cleaned['Outcome of Incident'] == 'Total Injured']
    city_injuries = injury_data.groupby('Million Plus Cities')['Count'].sum().sort_values(ascending=False)
    print(city_injuries.head(15))
    

    print("\nCity Safety Index (Fatality Rate %):")
    total_accidents = df_cleaned[df_cleaned['Outcome of Incident'] == 'Total number of Accidents'].groupby('Million Plus Cities')['Count'].sum()
    total_fatalities = df_cleaned[df_cleaned['Outcome of Incident'] == 'Persons Killed'].groupby('Million Plus Cities')['Count'].sum()
    
    
    safety_index = ((total_fatalities / total_accidents) * 100).round(2)
    safety_index = safety_index.sort_values(ascending=True)
    
    print("Top 15 Safest Cities (lowest fatality rate):")
    print(safety_index.head(15))
    
    print("\nTop 15 Most Dangerous Cities (highest fatality rate):")
    print(safety_index.tail(15).sort_values(ascending=False))
    
    return city_fatalities, city_injuries, safety_index

def perform_statistical_analysis(df_cleaned):
    """Perform statistical analysis"""
    
    print("\n6.4 STATISTICAL ANALYSIS")
    print("-" * 40)
    
  
    print("Correlation Analysis between Different Outcomes:")
    outcome_pivot = df_cleaned.pivot_table(index=['Million Plus Cities', 'Cause category'], 
                                          columns='Outcome of Incident', 
                                          values='Count', 
                                          aggfunc='sum', 
                                          fill_value=0)
    
    correlation_matrix = outcome_pivot.corr().round(3)
    print(correlation_matrix)
    
  
    print("\nStatistical Tests:")
    

    fatality_counts = df_cleaned[df_cleaned['Outcome of Incident'] == 'Persons Killed']['Count']
    shapiro_stat, shapiro_p = stats.shapiro(fatality_counts.sample(min(5000, len(fatality_counts))))
    print(f"Shapiro-Wilk Test for Normality (Fatalities): Statistic={shapiro_stat:.4f}, p-value={shapiro_p:.4f}")
    

    cause_groups = [group['Count'].values for name, group in df_cleaned.groupby('Cause category')]
    f_stat, f_p = stats.f_oneway(*cause_groups)
    print(f"ANOVA Test (Cause Categories): F-statistic={f_stat:.4f}, p-value={f_p:.4f}")
    
    return correlation_matrix, outcome_pivot

def perform_clustering_analysis(df_cleaned):
    """Perform clustering analysis on cities"""
    
    print("\n6.5 CLUSTERING ANALYSIS")
    print("-" * 40)
    
 
    city_features = df_cleaned.groupby(['Million Plus Cities', 'Outcome of Incident'])['Count'].sum().unstack(fill_value=0)
    
    
    scaler = StandardScaler()
    city_features_scaled = scaler.fit_transform(city_features)
    
 
    n_clusters = 4
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(city_features_scaled)
    

    city_clusters = pd.DataFrame({
        'City': city_features.index,
        'Cluster': clusters
    })
    
    print(f"Cities grouped into {n_clusters} clusters based on accident patterns:")
    for cluster_id in range(n_clusters):
        cluster_cities = city_clusters[city_clusters['Cluster'] == cluster_id]['City'].tolist()
        print(f"\nCluster {cluster_id + 1}: {', '.join(cluster_cities)}")
    

    print(f"\nCluster Characteristics:")
    for cluster_id in range(n_clusters):
        cluster_mask = clusters == cluster_id
        cluster_data = city_features.iloc[cluster_mask]
        cluster_mean = cluster_data.mean()
        
        print(f"\nCluster {cluster_id + 1} Average Profile:")
        for outcome in cluster_mean.index:
            print(f"  {outcome}: {cluster_mean[outcome]:.1f}")
    
    return city_clusters, city_features

def create_advanced_visualizations(df_cleaned, correlation_matrix, city_clusters, risky_combinations):
    """Create advanced visualizations"""
    
    print("\n6.6 ADVANCED VISUALIZATIONS")
    print("-" * 40)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    

    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                ax=axes[0, 0], square=True)
    axes[0, 0].set_title('Correlation Matrix: Accident Outcomes', fontweight='bold')
    

    top_risks = risky_combinations.head(10)
    risk_labels = [f"{combo[0][:15]}...\n{combo[1][:15]}..." for combo in top_risks.index]
    axes[0, 1].barh(range(len(top_risks)), top_risks.values, color='red', alpha=0.7)
    axes[0, 1].set_yticks(range(len(top_risks)))
    axes[0, 1].set_yticklabels(risk_labels, fontsize=8)
    axes[0, 1].set_title('Top 10 Risky Factor Combinations', fontweight='bold')
    axes[0, 1].set_xlabel('Count')
    
    
    city_features = df_cleaned.groupby(['Million Plus Cities', 'Outcome of Incident'])['Count'].sum().unstack(fill_value=0)
    colors = ['red', 'blue', 'green', 'orange']
    
    for cluster_id in city_clusters['Cluster'].unique():
        cluster_cities = city_clusters[city_clusters['Cluster'] == cluster_id]['City']
        cluster_data = city_features.loc[cluster_cities]
        
        axes[1, 0].scatter(cluster_data['Persons Killed'], 
                          cluster_data['Total number of Accidents'],
                          c=colors[cluster_id], alpha=0.7, 
                          label=f'Cluster {cluster_id + 1}')
    
    axes[1, 0].set_xlabel('Persons Killed')
    axes[1, 0].set_ylabel('Total Accidents')
    axes[1, 0].set_title('City Clusters: Fatalities vs Total Accidents', fontweight='bold')
    axes[1, 0].legend()
    
   
    severity_data = df_cleaned.groupby(['Cause category', 'Outcome of Incident'])['Count'].sum().unstack(fill_value=0)
    severity_ratio = (severity_data['Persons Killed'] / severity_data['Total number of Accidents'] * 100).sort_values(ascending=False)
    
    axes[1, 1].bar(range(len(severity_ratio)), severity_ratio.values, 
                   color='darkred', alpha=0.8)
    axes[1, 1].set_xticks(range(len(severity_ratio)))
    axes[1, 1].set_xticklabels(severity_ratio.index, rotation=45, ha='right')
    axes[1, 1].set_title('Fatality Rate by Cause Category (%)', fontweight='bold')
    axes[1, 1].set_ylabel('Fatality Rate (%)')
    
    plt.tight_layout()
    plt.show()
    
    return fig

def generate_risk_assessment_report(df_cleaned, city_fatalities, safety_index, risky_combinations):
    """Generate comprehensive risk assessment report"""
    
    print("\n6.7 COMPREHENSIVE RISK ASSESSMENT REPORT")
    print("=" * 60)
    
  
    total_accidents = df_cleaned[df_cleaned['Outcome of Incident'] == 'Total number of Accidents']['Count'].sum()
    total_fatalities = df_cleaned[df_cleaned['Outcome of Incident'] == 'Persons Killed']['Count'].sum()
    total_injured = df_cleaned[df_cleaned['Outcome of Incident'] == 'Total Injured']['Count'].sum()
    
    print(f"\nOVERALL STATISTICS:")
    print(f"Total Accidents: {total_accidents:,.0f}")
    print(f"Total Fatalities: {total_fatalities:,.0f}")
    print(f"Total Injured: {total_injured:,.0f}")
    print(f"Overall Fatality Rate: {(total_fatalities/total_accidents)*100:.2f}%")
    
   
    print(f"\n🚨 HIGH-RISK CITIES:")
    high_risk_cities = safety_index.tail(5).sort_values(ascending=False)
    for i, (city, rate) in enumerate(high_risk_cities.items(), 1):
        fatalities = city_fatalities.get(city, 0)
        print(f"{i}. {city}: {rate:.2f}% fatality rate ({fatalities:,.0f} deaths)")
    

    print(f"\nSAFEST CITIES:")
    safe_cities = safety_index.head(5)
    for i, (city, rate) in enumerate(safe_cities.items(), 1):
        fatalities = city_fatalities.get(city, 0)
        print(f"{i}. {city}: {rate:.2f}% fatality rate ({fatalities:,.0f} deaths)")
    
 
    print(f"\nCRITICAL RISK FACTORS:")
    top_deadly = risky_combinations.head(5)
    for i, (combo, count) in enumerate(top_deadly.items(), 1):
        print(f"{i}. {combo[0]} → {combo[1]} → {combo[2]}: {count:,.0f}")
    
    
    print(f"\nPRIORITY RECOMMENDATIONS:")
    print("1. Immediate intervention needed in high-risk cities")
    print("2. Focus on top deadly cause combinations")
    print("3. Implement targeted safety measures for critical factors")
    print("4. Regular monitoring of fatality rates")
    print("5. City-specific safety programs based on cluster analysis")

def save_advanced_analysis_results(cross_tab, risky_combinations, city_fatalities, safety_index, city_clusters):
    """Save advanced analysis results"""
    
    import os
    if not os.path.exists('output'):
        os.makedirs('output')
  

    cross_tab.to_csv('output/cross_tabulation_task6.csv')
    risky_combinations.to_csv('output/risky_combinations_task6.csv')
    city_fatalities.to_csv('output/city_fatalities_task6.csv')
    safety_index.to_csv('output/safety_index_task6.csv')
    city_clusters.to_csv('output/city_clusters_task6.csv', index=False)
    
    print("\nAdvanced analysis results saved to output folder!")

def main():
    """Main execution function for Task 6"""
    
    print("="*60)
    print("TASK 6: ADVANCED ANALYSIS")
    print("Road Accident Data Analysis 2020 India")
    print("="*60)
 

    df_cleaned, city_stats, cause_stats, outcome_stats, subcat_stats = load_data_from_previous_tasks()
    print(f"Dataset loaded: {df_cleaned.shape[0]} rows, {df_cleaned.shape[1]} columns")
    
 
    cross_tab, city_cause_cross = perform_cross_tabulation_analysis(df_cleaned)
    

    risky_combinations, deadly_combinations = analyze_risky_combinations(df_cleaned)
    
   
    city_fatalities, city_injuries, safety_index = analyze_city_safety_metrics(df_cleaned)
    

    correlation_matrix, outcome_pivot = perform_statistical_analysis(df_cleaned)
    
    
    city_clusters, city_features = perform_clustering_analysis(df_cleaned)
    
    
    fig = create_advanced_visualizations(df_cleaned, correlation_matrix, city_clusters, risky_combinations)
    

    generate_risk_assessment_report(df_cleaned, city_fatalities, safety_index, risky_combinations)
    
   
    save_advanced_analysis_results(cross_tab, risky_combinations, city_fatalities, safety_index, city_clusters)
    
    print("\nTask 6 completed successfully!")
    print("Advanced analysis completed with statistical insights and risk assessment.")

if __name__ == "__main__":
    main()
