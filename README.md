# Used Car Price Prediction Project

## Overview
This project aims to predict the prices of used cars using machine learning techniques. It includes data analysis, data preprocessing, model development, visualization, and a web interface for making predictions.

## Key Features
- **Advanced Data Analysis**: In-depth exploratory data analysis with comprehensive visualizations
- **Feature Engineering**: Creating new features that improve model performance
- **Model Development**: Implementation of multiple ML algorithms with hyperparameter tuning
- **Interactive Dashboard**: Dynamic data exploration with Plotly and Dash
- **Web Application**: User-friendly interface for making price predictions

## Project Structure
```
├── Codes/
│   ├── Used Car Price Prediction.ipynb      # Original notebook with analysis
│   ├── Enhanced_Visualizations.py           # Enhanced data visualizations
│   ├── Advanced_Model_Development.py        # Advanced modeling techniques
│   ├── Interactive_Dashboard.py             # Interactive data dashboard
│   └── app.py                               # Flask web application
├── Dataset/
│   └── dataset.csv                          # Used car dataset
├── Graphs & Visualization/
│   └── [Various visualization files]         # Generated visualization outputs
├── Pickle/
│   └── [Various model files]                 # Saved ML models
└── README.md                                # Project documentation
```

## Visualizations
The project includes comprehensive visualizations:
1. **Basic Analysis**: Distribution plots, correlation analysis, and feature relationships
2. **Enhanced Visualizations**: Advanced insights into feature importance and price drivers
3. **Interactive Dashboard**: Dynamic filtering and visualization of data patterns
4. **Model Performance**: Comparison of different algorithms and their effectiveness

## Machine Learning Models
Several regression models are implemented and compared:
- Linear Regression
- Ridge and Lasso Regression
- RandomForest, ExtraTrees 
- Gradient Boosting
- XGBoost, LightGBM, CatBoost

## Feature Engineering
The following features are engineered to improve model performance:
- Price per kilometer (value retention)
- Power to engine ratio
- Age impact (exponential decay)
- Manufacturer prestige score
- Various interaction features

## How to Run

### Prerequisites
- Python 3.8+
- Required packages: Install using `pip install -r requirements.txt`

### Running the Interactive Dashboard
```bash
cd Codes
python Interactive_Dashboard.py
```
Then open your browser to: http://localhost:8050

### Running the Web Application
```bash
cd Codes
python app.py
```
Then open your browser to: http://localhost:5000

### Training Advanced Models
```bash
cd Codes
python Advanced_Model_Development.py
```

## Web Application
The web application provides an intuitive interface to predict used car prices:
- Input car details (manufacturer, age, mileage, etc.)
- Get instant price predictions
- Utilizes the best-performing model

## Dashboard Features
The interactive dashboard allows you to:
- Filter data by various attributes (manufacturer, price range, fuel type, etc.)
- Visualize price distributions and trends
- Compare manufacturers and features
- Analyze relationships between features and prices
- Explore the impact of different factors on car prices

## Future Enhancements
- Time series analysis of price trends
- Geographic price variation analysis
- Integration of additional external factors (economy, fuel prices)
- Model deployment to cloud services

## Contributors
- Project Team

## Acknowledgments
- Data source: Used car dataset
