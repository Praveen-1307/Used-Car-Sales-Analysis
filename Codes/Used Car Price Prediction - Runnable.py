#!/usr/bin/env python
# coding: utf-8

# Car Price Prediction Model

import datetime
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import pickle 
import warnings
warnings.filterwarnings('ignore')

# Function to print a section header
def print_section(title):
    print("\n" + "="*50)
    print(f" {title} ".center(50, "="))
    print("="*50 + "\n")

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the project root
project_dir = os.path.dirname(current_dir)
# Set path to dataset
dataset_path = os.path.join(project_dir, 'Dataset', 'dataset.csv')

print_section("LOADING AND CLEANING DATA")
print(f"Loading dataset from: {dataset_path}")

# Read the dataset
try:
    df = pd.read_csv(dataset_path)
    print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
except FileNotFoundError:
    print(f"Dataset not found at {dataset_path}")
    print("Please make sure the dataset file 'dataset.csv' is in the Dataset directory")
    exit(1)

# Dropping unnecessary columns
df.drop(['Unnamed: 0','Location','New_Price'], axis=1, inplace=True, errors='ignore')
print("Dropped unnecessary columns")

# Check for null values
print("\nNull values in dataset:")
print(df.isna().sum())

# Extract manufacturer name
Manfacturer = df['Name'].str.split(" ", expand=True)
df['Manfacturer'] = Manfacturer[0]
print("\nExtracted manufacturer name from 'Name' column")

# Calculate years used
curr_time = datetime.datetime.now()
df['Years Used'] = df['Year'].apply(lambda x: curr_time.year - x)
print("Added 'Years Used' column")

# Drop Name and Year columns
df.drop(['Name', 'Year'], axis=1, inplace=True)
print("Dropped 'Name' and 'Year' columns")

# Clean and process Mileage column
Mileage = df['Mileage'].str.split(" ", expand=True)
df['Mileage'] = pd.to_numeric(Mileage[0], errors='coerce')
df['Mileage'].fillna(df['Mileage'].astype('float').mean(), inplace=True)
print("Processed 'Mileage' column")

# Clean and process Engine column
Engine = df['Engine'].str.split(" ", expand=True)
df['Engine'] = pd.to_numeric(Engine[0], errors='coerce')
df['Engine'].fillna(df['Engine'].astype('float').mean(), inplace=True)
print("Processed 'Engine' column")

# Clean and process Power column
Power = df['Power'].str.split(" ", expand=True)
df['Power'] = pd.to_numeric(Power[0], errors='coerce')
df['Power'].fillna(df['Power'].astype('float').mean(), inplace=True)
print("Processed 'Power' column")

# Fill missing values in Seats column
df['Seats'].fillna(df['Seats'].astype('float').mean(), inplace=True)
print("Filled missing values in 'Seats' column")

# Check for null values after cleaning
print("\nNull values after cleaning:")
print(df.isna().sum())

# Data Visualization (EXPANDED)
print_section("DATA VISUALIZATION")
print("Generating visualizations...")

# Create a directory for visualizations if it doesn't exist
graphs_dir = os.path.join(project_dir, 'Graphs & Visualization')
os.makedirs(graphs_dir, exist_ok=True)

# 1. Count of Cars by Manufacturer (Original)
plt.figure(figsize=(20, 10))
Cars = df['Manfacturer'].value_counts()
plot = sns.barplot(x=Cars.index, y=Cars.values)
plt.xticks(rotation=90)
for p in plot.patches:
    plot.annotate(p.get_height(), (p.get_x() + p.get_width() / 2.0, p.get_height()),
                 ha='center', va='center', xytext=(0, 5), textcoords='offset points')
plt.title('Count of Cars by Manufacturer', fontsize=16)
plt.xlabel('Manufacturer', fontsize=12)
plt.ylabel('Count of Cars', fontsize=12)
plt.savefig(os.path.join(graphs_dir, 'Count of Cars.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved 'Count of Cars.png'")

# 2. Correlation Heatmap (Enhanced)
plt.figure(figsize=(14, 12))
correlation = df.corr()
mask = np.triu(correlation)
sns.heatmap(correlation, mask=mask, cmap='coolwarm', annot=True, fmt='.2f', 
           linewidths=0.5, cbar=True, square=True)
plt.title('Correlation Heatmap', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, 'Correlation Heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved 'Correlation Heatmap.png'")

# 3. NEW: Distribution of Car Prices
plt.figure(figsize=(12, 6))
sns.histplot(df['Price'], bins=30, kde=True)
plt.title('Distribution of Car Prices', fontsize=16)
plt.xlabel('Price', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.axvline(df['Price'].mean(), color='red', linestyle='--', 
           label=f'Mean Price: {df["Price"].mean():,.2f}')
plt.axvline(df['Price'].median(), color='green', linestyle='-.', 
           label=f'Median Price: {df["Price"].median():,.2f}')
plt.legend()
plt.savefig(os.path.join(graphs_dir, 'Price Distribution.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved 'Price Distribution.png'")

# 4. NEW: Car Price by Manufacturer (Boxplot)
plt.figure(figsize=(20, 12))
top_manufacturers = df['Manfacturer'].value_counts().nlargest(15).index
df_top_manufacturers = df[df['Manfacturer'].isin(top_manufacturers)]
sns.boxplot(x='Manfacturer', y='Price', data=df_top_manufacturers)
plt.title('Car Price by Top 15 Manufacturers', fontsize=16)
plt.xticks(rotation=90)
plt.xlabel('Manufacturer', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.savefig(os.path.join(graphs_dir, 'Price by Manufacturer.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved 'Price by Manufacturer.png'")

# 5. NEW: Years Used vs Price (Scatter plot with regression line)
plt.figure(figsize=(12, 8))
sns.regplot(x='Years Used', y='Price', data=df, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
plt.title('Car Price vs Years Used', fontsize=16)
plt.xlabel('Years Used', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.savefig(os.path.join(graphs_dir, 'Price vs Years Used.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved 'Price vs Years Used.png'")

# 6. NEW: Mileage vs Price
plt.figure(figsize=(12, 8))
sns.regplot(x='Mileage', y='Price', data=df, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
plt.title('Car Price vs Mileage', fontsize=16) 
plt.xlabel('Mileage', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.savefig(os.path.join(graphs_dir, 'Price vs Mileage.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved 'Price vs Mileage.png'")

# 7. NEW: Engine Power vs Price
plt.figure(figsize=(12, 8))
sns.regplot(x='Power', y='Price', data=df, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
plt.title('Car Price vs Engine Power', fontsize=16)
plt.xlabel('Engine Power', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.savefig(os.path.join(graphs_dir, 'Price vs Power.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved 'Price vs Power.png'")

# 8. NEW: Price by Fuel Type
plt.figure(figsize=(14, 8))
sns.boxplot(x='Fuel_Type', y='Price', data=df)
plt.title('Car Price by Fuel Type', fontsize=16)
plt.xlabel('Fuel Type', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.savefig(os.path.join(graphs_dir, 'Price by Fuel Type.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved 'Price by Fuel Type.png'")

# 9. NEW: Price by Transmission Type
plt.figure(figsize=(12, 8))
sns.boxplot(x='Transmission', y='Price', data=df)
plt.title('Car Price by Transmission Type', fontsize=16)
plt.xlabel('Transmission Type', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.savefig(os.path.join(graphs_dir, 'Price by Transmission.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved 'Price by Transmission.png'")

# 10. NEW: Price by Number of Seats
plt.figure(figsize=(14, 8))
sns.boxplot(x='Seats', y='Price', data=df)
plt.title('Car Price by Number of Seats', fontsize=16)
plt.xlabel('Number of Seats', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.savefig(os.path.join(graphs_dir, 'Price by Seats.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved 'Price by Seats.png'")

# 11. NEW: Pairplot of Numerical Features
plt.figure(figsize=(16, 16))
numerical_columns = ['Price', 'Mileage', 'Engine', 'Power', 'Seats', 'Years Used']
sns.pairplot(df[numerical_columns], diag_kind='kde', plot_kws={'alpha': 0.6})
plt.savefig(os.path.join(graphs_dir, 'Numerical Features Pairplot.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved 'Numerical Features Pairplot.png'")

# Prepare data for modeling
print_section("DATA PREPROCESSING")
print("Splitting data into features and target...")

x = df.drop(['Price'], axis=1)
y = df['Price']

# Split into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=45)
print(f"Train set: {x_train.shape[0]} samples")
print(f"Test set: {x_test.shape[0]} samples")

# Create dummy variables for categorical columns
x_train = pd.get_dummies(x_train, columns=['Manfacturer', 'Fuel_Type', 'Transmission', 'Owner_Type'], drop_first=True)
x_test = pd.get_dummies(x_test, columns=['Manfacturer', 'Fuel_Type', 'Transmission', 'Owner_Type'], drop_first=True)

# Fix missing columns in test data
miss_col = set(x_train.columns) - set(x_test.columns)
for col in miss_col:
    x_test[col] = 0
x_test = x_test[x_train.columns]

print(f"Features after encoding: {x_train.shape[1]}")

# Standardize the features
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
print("Features standardized")

# Define function for model fitting
def FitModel(x, y, algo_name, algorithm, GridSearchParams, cv):
    print(f"\nTraining {algo_name}...")
    np.random.seed(10)
    grid = GridSearchCV(estimator=algorithm, param_grid=GridSearchParams, cv=cv,
                       scoring='r2', verbose=0, n_jobs=-1)
    grid_result = grid.fit(x_train, y_train)
    pred = grid_result.predict(x_test)
    best_params = grid_result.best_params_
    
    # Create Pickle directory if it doesn't exist
    pickle_dir = os.path.join(project_dir, 'Pickle')
    os.makedirs(pickle_dir, exist_ok=True)
    
    # Save the model
    try:
        import pickle
        pickle.dump(grid_result, open(os.path.join(pickle_dir, algo_name), 'wb'))
    except Exception as e:
        print(f"Warning: Could not save model - {e}")
    
    # Calculate metrics
    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    rmse = sqrt(mean_squared_error(y_test, pred))
    
    print(f"Algorithm Name: {algo_name}")
    print(f"Best Params: {best_params}")
    print(f"Percentage of R2 Score: {100 * r2:.2f}%")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    
    # NEW: Create regression plot for actual vs predicted
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title(f'Actual vs Predicted Prices - {algo_name}', fontsize=16)
    plt.xlabel('Actual Price', fontsize=12)
    plt.ylabel('Predicted Price', fontsize=12)
    # Add metrics to the plot
    plt.annotate(f'R² = {r2:.3f}\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}', 
                 xy=(0.05, 0.95), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="orange", alpha=0.8))
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, f'Regression_Plot_{algo_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved regression plot for {algo_name}")
    
    return r2

# Define function for boosted models
def BoostModel(x, y, algo_name, algorithm, GridSearchParams, cv):
    print(f"\nTraining boosted {algo_name}...")
    np.random.seed(10)
    grid = GridSearchCV(estimator=algorithm, param_grid=GridSearchParams, cv=cv,
                       scoring='r2', verbose=0, n_jobs=-1)
    grid_result = grid.fit(x_train, y_train)
    try:
        from sklearn.ensemble import AdaBoostRegressor
        AB = AdaBoostRegressor(base_estimator=grid_result, learning_rate=1)
        boostmodel = AB.fit(x_train, y_train)
        pred = boostmodel.predict(x_test)
        
        # Create Pickle directory if it doesn't exist
        pickle_dir = os.path.join(project_dir, 'Pickle')
        os.makedirs(pickle_dir, exist_ok=True)
        
        # Save the model
        try:
            import pickle
            pickle.dump(boostmodel, open(os.path.join(pickle_dir, algo_name), 'wb'))
        except Exception as e:
            print(f"Warning: Could not save model - {e}")
        
        # Calculate metrics
        r2 = r2_score(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        rmse = sqrt(mean_squared_error(y_test, pred))
        
        print(f"Algorithm Name: {algo_name}")
        print(f"Percentage of R2 Score: {100 * r2:.2f}%")
        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"Root Mean Squared Error: {rmse:.2f}")
        
        # NEW: Create regression plot for actual vs predicted
        plt.figure(figsize=(10, 8))
        plt.scatter(y_test, pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.title(f'Actual vs Predicted Prices - {algo_name}', fontsize=16)
        plt.xlabel('Actual Price', fontsize=12)
        plt.ylabel('Predicted Price', fontsize=12)
        # Add metrics to the plot
        plt.annotate(f'R² = {r2:.3f}\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}', 
                     xy=(0.05, 0.95), xycoords='axes fraction', 
                     bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="orange", alpha=0.8))
        plt.tight_layout()
        plt.savefig(os.path.join(graphs_dir, f'Regression_Plot_{algo_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved regression plot for {algo_name}")
        
        return r2
        
    except Exception as e:
        print(f"Error during boosting: {e}")
        # Fall back to original model if boosting fails
        pred = grid_result.predict(x_test)
        r2 = r2_score(y_test, pred)
        print(f"Using original model instead. R2 Score: {100 * r2:.2f}%")
        return r2

print_section("MODEL TRAINING")
print("Training basic regression models first...")

results = {}

# Linear Regression
param = {}
results["Linear Regression"] = FitModel(x, y, 'Linear Regression', LinearRegression(), param, cv=10)

# Lasso
results["Lasso"] = FitModel(x, y, 'Lasso', Lasso(), param, cv=10)

# Ridge
results["Ridge"] = FitModel(x, y, 'Ridge', Ridge(), param, cv=10)

# Random Forest
params = {'n_estimators': [44, 109, 314], 'random_state': [45]}
results["Random Forest"] = FitModel(x, y, 'Random Forest', RandomForestRegressor(), params, cv=10)

# Extra Trees
results["Extra Tree"] = FitModel(x, y, 'Extra Tree', ExtraTreesRegressor(), params, cv=10)

# Optional: Advanced models - only run if the libraries are available
try:
    print("\nAttempting to train advanced models...")
    
    # XG Boost
    from xgboost import XGBRegressor
    results["XG Boost"] = FitModel(x, y, 'XG Boost', XGBRegressor(), params, cv=10)
    
    # Cat Boost
    try:
        from catboost import CatBoostRegressor
        params = {'verbose': [0]}
        results["Cat Boost"] = FitModel(x, y, 'Cat Boost', CatBoostRegressor(), params, cv=10)
    except ImportError:
        print("CatBoost not available, skipping...")
    
    # Light GBM
    try:
        from lightgbm import LGBMRegressor
        param = {}
        results["Light GBM"] = FitModel(x, y, 'Light GBM', LGBMRegressor(), param, cv=10)
    except ImportError:
        print("LightGBM not available, skipping...")
        
except ImportError:
    print("Some advanced model libraries are not available. Continuing with basic models only.")

# Boost models (optional, depending on what succeeded earlier)
print_section("BOOSTING MODELS")
print("Training boosted versions of models...")

# Only try boosting if we have at least some results
if results:
    # Boosted Linear Regression
    param = {}
    results["Boosted Linear Regression"] = BoostModel(x, y, 'Boosted Linear Regression', LinearRegression(), param, cv=10)
    
    # Boosted Lasso
    results["Boosted Lasso"] = BoostModel(x, y, 'Boosted Lasso', Lasso(), param, cv=10)
    
    # Boosted Ridge
    results["Boosted Ridge"] = BoostModel(x, y, 'Boosted Ridge', Ridge(), param, cv=10)
    
    # Boosted Random Forest
    params = {'n_estimators': [44, 109, 314], 'random_state': [45]}
    results["Boosted Random Forest"] = BoostModel(x, y, 'Boosted Random Forest', RandomForestRegressor(), params, cv=10)
    
    # Boosted Extra Trees
    results["Boosted Extra Tree"] = BoostModel(x, y, 'Boosted Extra Tree', ExtraTreesRegressor(), params, cv=10)

print_section("RESULTS SUMMARY")

# NEW: Create model comparison bar chart
plt.figure(figsize=(14, 8))
models = list(results.keys())
scores = [results[model] * 100 for model in models]

# Sort models by performance
sorted_indices = np.argsort(scores)
sorted_models = [models[i] for i in sorted_indices]
sorted_scores = [scores[i] for i in sorted_indices]

# Create a color gradient based on scores
colors = plt.cm.viridis(np.array(sorted_scores)/100)

bars = plt.barh(sorted_models, sorted_scores, color=colors)
plt.xlabel('R² Score (%)', fontsize=12)
plt.title('Model Performance Comparison', fontsize=16)
plt.xlim(0, 100)

# Add value labels to the bars
for bar in bars:
    plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
             f'{bar.get_width():.2f}%', va='center')

plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, 'Model_Comparison.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved 'Model_Comparison.png'")

# Find the best model
if results:
    best_model = max(results.items(), key=lambda x: x[1])
    print(f"Best model: {best_model[0]} with R2 score of {best_model[1]*100:.2f}%")
    
    print("\nAll model results (sorted by performance):")
    for model, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{model}: {score*100:.2f}%")
else:
    print("No models were successfully trained.")

print("\nCompleted! All models have been trained and visualizations created in 'Graphs & Visualization' directory.")
print("For future predictions, use the model with the highest R2 score.")
print("Example usage:")
print("    import pickle")
print("    model = pickle.load(open('Pickle/MODEL_NAME', 'rb'))")
print("    prediction = model.predict(new_data)") 