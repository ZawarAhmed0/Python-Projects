A comprehensive machine learning solution for predicting sales based on TV advertising budget using K-Nearest Neighbors (KNN) regression algorithm. This project demonstrates the complete data science pipeline from data exploration to model deployment.
Overview
The TV Ad Budget Predictor helps businesses optimize their advertising spend by predicting product sales based on TV advertising budget. Using historical advertising data, the system trains multiple KNN models with different parameters and identifies the optimal configuration for accurate predictions.
Key Features

Data-driven insights: Comprehensive exploratory data analysis
Multiple model comparison: Tests various K values for optimal performance
Interactive predictions: Real-time sales forecasting for any TV budget
Professional visualizations: Clear, publication-ready charts and graphs
Business recommendations: Actionable insights for investment decisions

Installation
Prerequisites

Python 3.7 or higher
pip package manager

Setup Instructions

Clone the repository
bashgit clone https://github.com/yourusername/tv-ad-budget-predictor.git
cd tv-ad-budget-predictor

Create virtual environment
bashpython -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

Install dependencies
bashpip install -r requirements.txt


Usage
Quick Start
Run the complete analysis pipeline:
bashpython tv_ad_predictor.py
Custom Usage
pythonfrom tv_ad_predictor import TVAdPredictor

# Initialize with custom data
predictor = TVAdPredictor('your_data.csv')

# Run individual components
predictor.explore_data()
predictor.build_knn_models(k_values=[1, 3, 5, 10])
predictor.visualize_models()
predictor.make_predictions()
predictor.model_comparison()
Data Format
Your CSV file should contain two columns:

TV: TV advertising budget in thousands of dollars
Sales: Product sales in thousands of units

Example:
csvTV,Sales
230.1,22.1
44.5,10.4
17.2,9.3
Analysis Pipeline
1. Exploratory Data Analysis

Dataset overview and statistics
Data quality assessment
Correlation analysis
Distribution visualizations

2. Model Training

K-Nearest Neighbors implementation
Multiple K value testing (1, 3, 5, 8, 10, 15)
Performance metric calculation (MSE, RMSE, R²)

3. Model Visualization

Prediction line comparison
Training data overlay
Performance metric display

4. Interactive Predictions

Real-time sales forecasting
Multiple model consensus
Business recommendation engine

5. Model Comparison

Performance ranking
Best model identification
Bias-variance tradeoff analysis

Sample Output
EXPLORATORY DATA ANALYSIS
============================================================
Dataset Overview:
- Total observations: 30
- Features: ['TV', 'Sales']
- Target variable: Sales

Correlation Analysis:
TV Budget vs Sales correlation: 0.782
Strong positive correlation - Excellent for prediction

MODEL TRAINING AND EVALUATION
============================================================
K= 1: MSE= 0.000 | RMSE= 0.000 | R²= 1.000
K= 3: MSE= 2.145 | RMSE= 1.464 | R²= 0.724
K= 5: MSE= 3.221 | RMSE= 1.795 | R²= 0.586

Best Performing Model: K=3
   • Lowest MSE: 2.145
   • RMSE: 1.464
   • Highest R²: 0.724
Learning Outcomes
This project demonstrates:

Machine Learning Fundamentals: Supervised learning and regression
Data Science Pipeline: Complete workflow from data to insights
Model Evaluation: Performance metrics and model selection
Data Visualization: Effective communication of results
Software Engineering: Clean, modular, and documented code

Project Structure
tv-ad-budget-predictor/
│
├── advertising_data.csv      # Sample dataset
├── tv_ad_predictor.py       # Main application
├── requirements.txt         # Python dependencies
├── README.md               # Project documentation
├── LICENSE                 # License file
└── .gitignore             # Git ignore rules
Technical Details

Algorithm: K-Nearest Neighbors Regression
Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn
Metrics: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), R-squared (R²)
Visualization: matplotlib and seaborn for professional charts

Performance Metrics

MSE (Mean Squared Error): Measures average squared differences between actual and predicted values
RMSE (Root Mean Squared Error): MSE in original units, easier to interpret
R² (R-squared): Proportion of variance explained by the model (0-1 scale)

Contributing
Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

Fork the project
Create your feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request

License
This project is licensed under the MIT License - see the LICENSE file for details.
Contact
Project Link: https://github.com/ZawarAhmed0/Python-Projects/tv-ad-budget-predictor
Acknowledgments

Inspired by the classic advertising dataset used in statistical learning
Built with the excellent Python data science ecosystem
Thanks to the scikit-learn community for providing robust ML tools


If you found this project helpful, please give it a star!