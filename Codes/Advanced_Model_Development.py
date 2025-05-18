import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import time
from math import sqrt
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('../Dataset/dataset.csv')
print(f"Dataset loaded with shape: {df.shape}")

# --------------------- FEATURE ENGINEERING ---------------------
print("\nPerforming feature engineering...")

# Function to create new features
def engineer_features(df):
    # Create a copy to avoid modifying the original dataframe
    df_new = df.copy()
    
    # Add Years_used column based on Year
    df_new['Years_used'] = 2023 - df_new['Year']
    
    # Price per kilometer (value retention)
    df_new['Price_per_km'] = df_new['Price'] / (df_new['Kilometers_Driven'] + 1)  # Adding 1 to avoid division by zero
    
    # Power to weight ratio
    if 'Engine' in df_new.columns and 'Power' in df_new.columns:
        df_new['Power_to_Engine_Ratio'] = df_new['Power'] / df_new['Engine']
    
    # Age impact (exponential decay of value)
    if 'Years_used' in df_new.columns:
        df_new['Age_Impact'] = np.exp(-0.1 * df_new['Years_used'])
    
    # Manufacturer prestige score (based on average price per manufacturer)
    manufacturer_avg_price = df_new.groupby('Name')['Price'].mean()  # Changed from Manfacturer to Name
    df_new['Manufacturer_Prestige'] = df_new['Name'].map(manufacturer_avg_price)  # Changed from Manfacturer to Name
    
    # Fuel efficiency proxy (if available)
    if 'Mileage' in df_new.columns:
        try:
            df_new['Mileage_Value'] = df_new['Mileage'].str.extract('(\\d+\\.?\\d*)').astype(float)
        except:
            print("Could not extract mileage values")
    
    # Interaction features
    df_new['Engine_Power_Interaction'] = df_new['Engine'] * df_new['Power']
    df_new['Age_Km_Interaction'] = df_new['Years_used'] * np.log1p(df_new['Kilometers_Driven'])
    
    # Binning features
    df_new['Engine_Category'] = pd.qcut(df_new['Engine'], q=5, labels=['Very Small', 'Small', 'Medium', 'Large', 'Very Large'])
    df_new['Power_Category'] = pd.qcut(df_new['Power'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    return df_new

# Apply feature engineering
df_engineered = engineer_features(df)
print(f"Feature engineering completed. New shape: {df_engineered.shape}")

# Show new features
new_features = list(set(df_engineered.columns) - set(df.columns))
print(f"New features created: {new_features}")

# --------------------- PREPROCESSING ---------------------
print("\nPreprocessing data...")

# Define features and target
X = df_engineered.drop('Price', axis=1)
y = df_engineered['Price']

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"Categorical features: {len(categorical_cols)}")
print(f"Numerical features: {len(numerical_cols)}")

# Split the data with stratified sampling based on price ranges
y_binned = pd.qcut(y, q=5, duplicates='drop')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y_binned)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Create preprocessing pipelines
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# --------------------- MODEL TRAINING AND EVALUATION ---------------------
print("\nSetting up models with advanced hyperparameter tuning...")

# Define models with hyperparameter grids
models = {
    'LinearRegression': {
        'model': LinearRegression(),
        'params': {},
    },
    'Lasso': {
        'model': Lasso(),
        'params': {
            'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
            'max_iter': [1000, 3000]
        }
    },
    'Ridge': {
        'model': Ridge(),
        'params': {
            'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg']
        }
    },
    'ElasticNet': {
        'model': ElasticNet(),
        'params': {
            'alpha': [0.001, 0.01, 0.1, 1, 10],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
            'max_iter': [1000, 3000]
        }
    },
    'RandomForest': {
        'model': RandomForestRegressor(),
        'params': {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False],
            'random_state': [42]
        }
    },
    'ExtraTrees': {
        'model': ExtraTreesRegressor(),
        'params': {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'random_state': [42]
        }
    },
    'GradientBoosting': {
        'model': GradientBoostingRegressor(),
        'params': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'random_state': [42]
        }
    },
    'XGBoost': {
        'model': XGBRegressor(),
        'params': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'random_state': [42]
        }
    },
    'LightGBM': {
        'model': LGBMRegressor(),
        'params': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [31, 50, 100],
            'max_depth': [-1, 10, 20, 30],
            'min_child_samples': [20, 30, 50],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'random_state': [42]
        }
    },
    'CatBoost': {
        'model': CatBoostRegressor(verbose=0),
        'params': {
            'iterations': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'depth': [4, 6, 8, 10],
            'l2_leaf_reg': [1, 3, 5, 7],
            'random_seed': [42]
        }
    }
}

# Function to train and evaluate a model
def train_and_evaluate(name, model, params, X_train, X_test, y_train, y_test, n_iter=10, cv=5):
    print(f"\nTraining {name} model...")
    start_time = time.time()
    
    # Create the full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Use RandomizedSearchCV for efficient hyperparameter tuning
    if params:
        search = RandomizedSearchCV(
            pipeline, 
            param_distributions={f'model__{key}': value for key, value in params.items()},
            n_iter=n_iter,
            scoring='r2',
            cv=cv,
            random_state=42,
            n_jobs=-1
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        best_params = search.best_params_
        print(f"Best parameters: {best_params}")
    else:
        best_model = pipeline
        best_model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    
    # Cross-validation score
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='r2')
    cv_r2 = cv_scores.mean()
    
    # Calculate training time
    training_time = time.time() - start_time
    
    print(f"{name} Model Results:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  Mean Absolute Error: {mae:.4f}")
    print(f"  Root Mean Squared Error: {rmse:.4f}")
    print(f"  CV R² Score: {cv_r2:.4f}")
    print(f"  Training Time: {training_time:.2f} seconds")
    
    # Save the model
    with open(f'../Pickle/{name}_Advanced.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    results = {
        'model_name': name,
        'r2_score': r2,
        'mae': mae,
        'rmse': rmse,
        'cv_r2': cv_r2,
        'training_time': training_time,
        'best_model': best_model
    }
    
    if params:
        results['best_params'] = best_params
    
    return results

# Train and evaluate all models, keeping track of results
results = []

for name, config in models.items():
    try:
        result = train_and_evaluate(
            name=name,
            model=config['model'],
            params=config['params'],
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            n_iter=10 if config['params'] else 1
        )
        results.append(result)
    except Exception as e:
        print(f"Error training {name}: {e}")

# --------------------- RESULTS ANALYSIS ---------------------
print("\nModel Comparison and Analysis:")

# Create a DataFrame with results
results_df = pd.DataFrame([
    {
        'Model': r['model_name'],
        'R² Score': r['r2_score'],
        'MAE': r['mae'],
        'RMSE': r['rmse'],
        'CV R² Score': r['cv_r2'],
        'Training Time (s)': r['training_time']
    }
    for r in results
])

# Sort by R² score (descending)
results_df = results_df.sort_values('R² Score', ascending=False).reset_index(drop=True)
print(results_df)

# Plot results comparison
plt.figure(figsize=(14, 10))

# Plot R² Scores
plt.subplot(2, 1, 1)
sns.barplot(x='Model', y='R² Score', data=results_df, palette='viridis')
plt.title('Model Comparison - R² Score (higher is better)', fontsize=16)
plt.xticks(rotation=45)
plt.ylim(0, 1)

# Plot MAE
plt.subplot(2, 1, 2)
sns.barplot(x='Model', y='MAE', data=results_df, palette='viridis')
plt.title('Model Comparison - Mean Absolute Error (lower is better)', fontsize=16)
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('../Graphs & Visualization/Advanced_Model_Comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# --------------------- FEATURE IMPORTANCE ANALYSIS ---------------------
print("\nFeature importance analysis:")

# Get the best model based on R² Score
best_model_name = results_df.iloc[0]['Model']
best_model_result = next(r for r in results if r['model_name'] == best_model_name)
best_model = best_model_result['best_model']

# Check if the model has feature_importances_
try:
    # For models like RandomForest, XGBoost, etc.
    if hasattr(best_model.named_steps['model'], 'feature_importances_'):
        # Get feature names after preprocessing
        feature_names = []
        
        # Get numerical feature names (they remain the same)
        if numerical_cols:
            feature_names.extend(numerical_cols)
        
        # Get one-hot encoded feature names
        if categorical_cols:
            # Extract one-hot encoder
            ohe = best_model.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot']
            
            # Get all categories 
            all_categories = ohe.get_feature_names_out(categorical_cols)
            feature_names.extend(all_categories)
        
        # Get importance scores
        importances = best_model.named_steps['model'].feature_importances_
        
        # Create dictionary of feature importances
        if len(feature_names) == len(importances):
            feature_importance = dict(zip(feature_names, importances))
            
            # Sort features by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            # Print top 20 features
            print(f"\nTop 20 most important features for {best_model_name}:")
            for i, (feature, importance) in enumerate(sorted_features[:20], 1):
                print(f"{i}. {feature}: {importance:.4f}")
            
            # Plot feature importance
            plt.figure(figsize=(14, 10))
            features = [x[0] for x in sorted_features[:20]]
            scores = [x[1] for x in sorted_features[:20]]
            
            sns.barplot(x=scores, y=features, palette='viridis')
            plt.title(f'Top 20 Feature Importances - {best_model_name}', fontsize=16)
            plt.xlabel('Importance', fontsize=12)
            plt.tight_layout()
            plt.savefig('../Graphs & Visualization/Advanced_Feature_Importance.png', dpi=300, bbox_inches='tight')
            plt.close()
        else:
            print(f"Feature names ({len(feature_names)}) and importances ({len(importances)}) length mismatch")
    else:
        print(f"Model {best_model_name} doesn't have feature_importances_ attribute")
except Exception as e:
    print(f"Error extracting feature importances: {e}")

print("\nAdvanced model development completed!")
print(f"Best model: {best_model_name} with R² Score: {results_df.iloc[0]['R² Score']:.4f}")
print(f"Models saved in the Pickle directory with '_Advanced' suffix")
print(f"Visualizations saved in the Graphs & Visualization directory") 