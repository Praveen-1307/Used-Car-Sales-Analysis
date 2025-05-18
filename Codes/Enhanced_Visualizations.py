import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.ticker as mtick

# Set style for all plots
plt.style.use('ggplot')
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Load the dataset
df = pd.read_csv('../Dataset/dataset.csv')

# 1. Distribution of Car Prices
plt.figure(figsize=(14, 8))
sns.histplot(df['Price'], kde=True, bins=30, color='darkblue')
plt.title('Distribution of Used Car Prices', fontsize=18)
plt.xlabel('Price (in Lakh ₹)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.axvline(df['Price'].mean(), color='red', linestyle='--', label=f'Mean: {df["Price"].mean():.2f}')
plt.axvline(df['Price'].median(), color='green', linestyle='--', label=f'Median: {df["Price"].median():.2f}')
plt.legend()
plt.tight_layout()
plt.savefig('../Graphs & Visualization/Price_Distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Box plot of Price by Manufacturer
plt.figure(figsize=(18, 10))
# Get top 10 manufacturers by count
top_manufacturers = df['Manfacturer'].value_counts().nlargest(10).index
df_top10 = df[df['Manfacturer'].isin(top_manufacturers)]
sns.boxplot(x='Manfacturer', y='Price', data=df_top10, palette='viridis')
plt.title('Price Distribution by Top 10 Manufacturers', fontsize=18)
plt.xlabel('Manufacturer', fontsize=14)
plt.ylabel('Price (in Lakh ₹)', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('../Graphs & Visualization/Price_by_Manufacturer.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Relationship between Year_of_Purchase, Years_used and Price
plt.figure(figsize=(14, 8))
scatter = plt.scatter(df['Years_used'], df['Price'], c=df['Year_of_Purchase'], 
                     cmap='viridis', alpha=0.7, s=100)
plt.colorbar(scatter, label='Year of Purchase')
plt.title('Relationship between Years Used and Price', fontsize=18)
plt.xlabel('Years Used', fontsize=14)
plt.ylabel('Price (in Lakh ₹)', fontsize=14)
plt.tight_layout()
plt.savefig('../Graphs & Visualization/Years_Used_vs_Price.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Fuel Type Analysis
plt.figure(figsize=(16, 12))

# Subplot 1: Count by Fuel Type
plt.subplot(2, 2, 1)
fuel_count = df['Fuel_Type'].value_counts()
sns.barplot(x=fuel_count.index, y=fuel_count.values, palette='viridis')
plt.title('Count of Cars by Fuel Type', fontsize=16)
plt.xlabel('Fuel Type', fontsize=12)
plt.ylabel('Count', fontsize=12)

# Subplot 2: Average Price by Fuel Type
plt.subplot(2, 2, 2)
avg_price_by_fuel = df.groupby('Fuel_Type')['Price'].mean().sort_values(ascending=False)
sns.barplot(x=avg_price_by_fuel.index, y=avg_price_by_fuel.values, palette='viridis')
plt.title('Average Price by Fuel Type', fontsize=16)
plt.xlabel('Fuel Type', fontsize=12)
plt.ylabel('Average Price (in Lakh ₹)', fontsize=12)

# Subplot 3: Box plot of Price by Fuel Type
plt.subplot(2, 2, 3)
sns.boxplot(x='Fuel_Type', y='Price', data=df, palette='viridis')
plt.title('Price Distribution by Fuel Type', fontsize=16)
plt.xlabel('Fuel Type', fontsize=12)
plt.ylabel('Price (in Lakh ₹)', fontsize=12)

# Subplot 4: Violin plot of Price by Fuel Type
plt.subplot(2, 2, 4)
sns.violinplot(x='Fuel_Type', y='Price', data=df, palette='viridis')
plt.title('Price Distribution (Violin) by Fuel Type', fontsize=16)
plt.xlabel('Fuel Type', fontsize=12)
plt.ylabel('Price (in Lakh ₹)', fontsize=12)

plt.tight_layout()
plt.savefig('../Graphs & Visualization/Fuel_Type_Analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Relationship between Kilometers_Driven and Price with Transmission
plt.figure(figsize=(14, 8))
sns.scatterplot(data=df, x='Kilometers_Driven', y='Price', hue='Transmission', 
                palette='viridis', size='Engine', sizes=(20, 200), alpha=0.7)
plt.title('Relationship between Kilometers Driven and Price by Transmission Type', fontsize=18)
plt.xlabel('Kilometers Driven', fontsize=14)
plt.ylabel('Price (in Lakh ₹)', fontsize=14)
plt.legend(title='Transmission Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('../Graphs & Visualization/Kilometers_Price_Transmission.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. Feature Importance Plot (Using Random Forest model or display from existing model)
try:
    # Try to load the Random Forest model if it exists
    rf_model = pickle.load(open('../Random Forest', 'rb'))
    
    # Get feature importance
    feature_names = df.drop(['Price'], axis=1).columns
    importances = rf_model.best_estimator_.feature_importances_
    
    # Sort importances
    indices = np.argsort(importances)[::-1]
    sorted_importances = importances[indices]
    sorted_features = [feature_names[i] for i in indices]
    
    # Plot top 15 features
    plt.figure(figsize=(14, 8))
    plt.bar(range(15), sorted_importances[:15], align='center', color='darkgreen')
    plt.xticks(range(15), sorted_features[:15], rotation=45, ha='right')
    plt.title('Top 15 Feature Importances (Random Forest)', fontsize=18)
    plt.xlabel('Features', fontsize=14)
    plt.ylabel('Importance', fontsize=14)
    plt.tight_layout()
    plt.savefig('../Graphs & Visualization/Feature_Importance.png', dpi=300, bbox_inches='tight')
    plt.close()
except:
    print("Random Forest model file not found or cannot be loaded. Skipping feature importance plot.")

# 7. Model Performance Comparison
# Define models and their scores
models = ['Linear Regression', 'Lasso', 'Ridge', 'Random Forest', 'Extra Trees', 'XGBoost', 'CatBoost', 'LightGBM']
r2_scores = [76.90, 72.26, 76.56, 89.17, 91.13, 88.46, 92.02, 88.82]  # From the notebook results
mae_scores = [2.97, 3.37, 2.98, 1.83, 1.67, 1.86, 1.65, 1.83]  # From the notebook results

plt.figure(figsize=(16, 10))

# R2 Score Comparison
plt.subplot(2, 1, 1)
bars = plt.bar(models, r2_scores, color=sns.color_palette('viridis', len(models)))
plt.title('Model Comparison: R² Score', fontsize=18)
plt.xlabel('Models', fontsize=14)
plt.ylabel('R² Score (%)', fontsize=14)
plt.ylim(60, 100)  # Set y-axis limits for better visualization
plt.xticks(rotation=45)

# Add value labels on the bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{height:.2f}%', ha='center', va='bottom', fontsize=10)

# Mean Absolute Error Comparison
plt.subplot(2, 1, 2)
bars = plt.bar(models, mae_scores, color=sns.color_palette('viridis', len(models)))
plt.title('Model Comparison: Mean Absolute Error', fontsize=18)
plt.xlabel('Models', fontsize=14)
plt.ylabel('MAE (Lower is Better)', fontsize=14)
plt.xticks(rotation=45)

# Add value labels on the bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
            f'{height:.2f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('../Graphs & Visualization/Model_Performance_Comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 8. Correlation Matrix with Annotations (Enhanced version of existing heatmap)
plt.figure(figsize=(16, 12))
corr_matrix = df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0, annot=True, 
            fmt='.2f', square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title('Enhanced Correlation Matrix', fontsize=20)
plt.tight_layout()
plt.savefig('../Graphs & Visualization/Enhanced_Correlation_Matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# 9. Price by Transmission and Owner Type
plt.figure(figsize=(16, 8))
sns.boxplot(x='Transmission', y='Price', hue='Owner_Type', data=df, palette='viridis')
plt.title('Price Distribution by Transmission and Owner Type', fontsize=18)
plt.xlabel('Transmission Type', fontsize=14)
plt.ylabel('Price (in Lakh ₹)', fontsize=14)
plt.legend(title='Owner Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('../Graphs & Visualization/Price_by_Transmission_Owner.png', dpi=300, bbox_inches='tight')
plt.close()

# 10. Pairplot of key numerical features
key_features = ['Price', 'Years_used', 'Kilometers_Driven', 'Engine', 'Power', 'Seats']
plt.figure(figsize=(20, 16))
sns.pairplot(df[key_features], diag_kind='kde', plot_kws={'alpha': 0.6, 's': 80, 'edgecolor': 'k'}, 
             diag_kws={'shade': True})
plt.suptitle('Pairplot of Key Numerical Features', fontsize=24, y=1.02)
plt.savefig('../Graphs & Visualization/Key_Features_Pairplot.png', dpi=300, bbox_inches='tight')
plt.close()

print("Enhanced visualizations created successfully and saved to the Graphs & Visualization directory.") 