"""
TV Ad Budget Predictor - Machine Learning Project
=================================================

A comprehensive machine learning solution for predicting sales based on TV advertising budget
using K-Nearest Neighbors (KNN) regression algorithm.

Author: Zawar Ahmec Nabeel
Date: 8/3/2025
Version: 1.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
import os
warnings.filterwarnings('ignore')

class TVAdPredictor:
    """
    A machine learning predictor for TV advertising budget optimization.
    
    This class implements K-Nearest Neighbors regression to predict sales
    based on TV advertising budget, with comprehensive analysis and visualization.
    """
    
    def __init__(self, data_path='advertising_data.csv'):
        """
        Initialize the TV Ad Predictor.
        
        Args:
            data_path (str): Path to the CSV file containing advertising data
        """
        self.data_path = data_path
        self.data = None
        self.models = {}
        self.load_data()
        
    def load_data(self):
        """Load advertising data from CSV file."""
        try:
            if os.path.exists(self.data_path):
                self.data = pd.read_csv(self.data_path)
                print(f" Data loaded successfully from {self.data_path}")
            else:
                print(f" Data file not found: {self.data_path}")
                print("Creating sample dataset...")
                self._create_sample_data()
        except Exception as e:
            print(f" Error loading data: {e}")
            print("Creating sample dataset...")
            self._create_sample_data()
    
    def _create_sample_data(self):
        """
        Create sample advertising data if CSV file is not available. If you have your own data you can make 
        changes accordingly
        """
        self.data = pd.DataFrame({
            'TV': [230.1, 44.5, 17.2, 151.5, 180.8, 8.7, 57.5, 120.2, 8.6, 199.8,
                   66.1, 214.7, 23.8, 97.5, 204.1, 195.4, 67.8, 281.4, 69.2, 147.3,
                   218.4, 13.2, 228.3, 62.3, 262.9, 142.9, 240.1, 248.8, 70.6, 292.9],
            'Sales': [22.1, 10.4, 9.3, 18.5, 12.9, 7.2, 11.8, 13.2, 4.8, 10.6,
                     8.6, 17.4, 9.2, 9.7, 19.0, 22.4, 12.5, 24.4, 11.3, 13.6,
                     18.4, 5.8, 17.1, 12.6, 19.4, 15.2, 20.5, 17.9, 11.2, 21.4]
        })
        
    def explore_data(self):
        """
        Perform comprehensive exploratory data analysis.
        
        Analyzes dataset characteristics, statistical properties, and visualizes
        the relationship between TV budget and sales.
        """
        print("="*60)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        # Dataset overview
        print("Dataset Overview:")
        print(f"• Total observations: {len(self.data)}")
        print(f"• Features: {list(self.data.columns)}")
        print(f"• Target variable: Sales")
        print()
        
        # Data preview
        print("Data Preview:")
        print(self.data.head())
        print()
        
        # Statistical summary
        print("Statistical Summary:")
        print(self.data.describe())
        print()
        
        # Data quality check
        print("Data Quality:")
        print(f"• Missing values: {self.data.isnull().sum().sum()}")
        print(f"• Duplicate rows: {self.data.duplicated().sum()}")
        print()
        
        # Visualizations
        self._create_eda_plots()
        
        # Correlation analysis
        correlation = self.data['TV'].corr(self.data['Sales'])
        print(f"Correlation Analysis:")
        print(f"TV Budget vs Sales correlation: {correlation:.3f}")
        
        if correlation > 0.7:
            print("Strong positive correlation - Excellent for prediction")
        elif correlation > 0.5:
            print("Moderate positive correlation - Good for prediction")
        else:
            print("Weak correlation - Prediction may be challenging")
        print()
        
    def _create_eda_plots(self):
        """Create exploratory data analysis visualizations."""
        plt.figure(figsize=(15, 5))
        
        # Scatter plot
        plt.subplot(1, 3, 1)
        plt.scatter(self.data['TV'], self.data['Sales'], alpha=0.7, color='blue', s=50)
        plt.xlabel('TV Budget ($1000s)')
        plt.ylabel('Sales (1000 units)')
        plt.title('TV Budget vs Sales Relationship')
        plt.grid(True, alpha=0.3)
        
        # Distribution plots
        plt.subplot(1, 3, 2)
        plt.hist(self.data['TV'], bins=12, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('TV Budget ($1000s)')
        plt.ylabel('Frequency')
        plt.title('TV Budget Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        plt.hist(self.data['Sales'], bins=12, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.xlabel('Sales (1000 units)')
        plt.ylabel('Frequency')
        plt.title('Sales Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def build_knn_models(self, k_values=[1, 3, 5, 8, 10, 15]):
        """
        Build and train K-Nearest Neighbors models with different K values.
        
        Args:
            k_values (list): List of K values to test for KNN models
        """
        print("="*60)
        print("MODEL TRAINING AND EVALUATION")
        print("="*60)
        
        X = self.data[['TV']]
        y = self.data['Sales']
        
        print("Training KNN models with different K values...")
        print("K-Nearest Neighbors Algorithm Overview:")
        print("• K=1: High sensitivity to individual data points")
        print("• K=3-5: Balanced approach, often optimal")
        print("• K>10: High smoothing, may underfit")
        print()
        
        # Train models
        for k in k_values:
            knn = KNeighborsRegressor(n_neighbors=k)
            knn.fit(X, y)
            
            # Make predictions
            y_pred = knn.predict(X)
            
            # Calculate metrics
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mse)
            
            # Store model and metrics
            self.models[k] = {
                'model': knn,
                'mse': mse,
                'rmse': rmse,
                'r2': r2
            }
            
            print(f"K={k:2d}: MSE={mse:6.3f} | RMSE={rmse:6.3f} | R²={r2:6.3f}")
        
        print(f"\nModel Training Complete! {len(self.models)} models trained.")
        print("Performance Metrics Explained:")
        print("• MSE (Mean Squared Error): Lower values indicate better fit")
        print("• RMSE (Root Mean Squared Error): Error in same units as target")
        print("• R² (R-squared): Proportion of variance explained (0-1, higher is better)")
        print()
        
    def visualize_models(self, k_values_to_plot=[1, 3, 5, 10]):
        """
        Visualize different KNN models and their predictions.
        
        Args:
            k_values_to_plot (list): List of K values to visualize
        """
        print("="*60)
        print("MODEL VISUALIZATION AND COMPARISON")
        print("="*60)
        
        # Create prediction range
        tv_range = np.linspace(0, self.data['TV'].max() * 1.1, 200).reshape(-1, 1)
        
        plt.figure(figsize=(16, 12))
        
        for i, k in enumerate(k_values_to_plot, 1):
            plt.subplot(2, 2, i)
            
            # Plot original data
            plt.scatter(self.data['TV'], self.data['Sales'], 
                       alpha=0.6, color='blue', s=60, label='Training Data', zorder=3)
            
            # Plot predictions if model exists
            if k in self.models:
                predictions = self.models[k]['model'].predict(tv_range)
                plt.plot(tv_range, predictions, color='red', linewidth=3, 
                        label=f'KNN Prediction (K={k})', zorder=2)
                
                # Add performance metrics to title
                mse = self.models[k]['mse']
                r2 = self.models[k]['r2']
                plt.title(f'K={k} | MSE={mse:.2f} | R²={r2:.3f}', fontsize=12, fontweight='bold')
            else:
                plt.title(f'K={k} (Model not trained)', fontsize=12)
            
            plt.xlabel('TV Budget ($1000s)')
            plt.ylabel('Sales (1000 units)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.show()
        
        # Analysis explanation
        print("Model Behavior Analysis:")
        print("• K=1: Highly flexible, may overfit to training data")
        print("• K=3: Good balance between flexibility and stability")
        print("• K=5: Moderate smoothing, robust to outliers")
        print("• K=10: High smoothing, captures general trend")
        print()
        
    def make_predictions(self, interactive=True):
        """
        Make predictions for new TV budget values.
        
        Args:
            interactive (bool): Whether to run in interactive mode
        """
        print("="*60)
        print("PREDICTION ENGINE")
        print("="*60)
        
        if not self.models:
            print("No trained models found. Please run build_knn_models() first.")
            return
        
        if not interactive:
            # Demo predictions
            demo_budgets = [50, 100, 150, 200, 250]
            print("Demo Predictions:")
            for budget in demo_budgets:
                self._predict_single_value(budget)
            return
        
        print("Interactive Prediction Mode")
        print("Enter TV budget values to get sales predictions from all trained models.")
        print()
        
        while True:
            try:
                tv_budget = input("Enter TV budget ($1000s) [or 'quit' to exit]: ")
                
                if tv_budget.lower() in ['quit', 'exit', 'q']:
                    break
                
                tv_budget = float(tv_budget)
                
                if tv_budget < 0:
                    print("Please enter a positive value.")
                    continue
                
                self._predict_single_value(tv_budget)
                
            except ValueError:
                print("Please enter a valid number or 'quit' to exit.")
            except KeyboardInterrupt:
                print("\nPrediction session ended.")
                break
                
    def _predict_single_value(self, tv_budget):
        """Make prediction for a single TV budget value."""
        print(f"\nPredictions for TV Budget: ${tv_budget}k")
        print("-" * 50)
        
        X_new = np.array([[tv_budget]])
        predictions = []
        
        # Get predictions from all models
        for k in sorted(self.models.keys()):
            prediction = self.models[k]['model'].predict(X_new)[0]
            predictions.append(prediction)
            print(f"K={k:2d}: {prediction:6.1f}k units")
        
        # Summary statistics
        avg_prediction = np.mean(predictions)
        std_prediction = np.std(predictions)
        
        print(f"\nPrediction Summary:")
        print(f"Average prediction: {avg_prediction:.1f}k units")
        print(f"Standard deviation: {std_prediction:.1f}k units")
        
        # Business recommendation
        if avg_prediction > 18:
            recommendation = "Excellent ROI expected"
        elif avg_prediction > 12:
            recommendation = "Good investment opportunity"
        elif avg_prediction > 8:
            recommendation = "Moderate returns expected"
        else:
            recommendation = "Low returns - consider alternatives"
            
        print(f"Business recommendation: {recommendation}")
        print()
        
    def model_comparison(self):
        """Compare all trained models and identify the best performer."""
        print("="*60)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*60)
        
        if not self.models:
            print("No trained models found. Please run build_knn_models() first.")
            return
            
        # Create comparison DataFrame
        comparison_data = []
        for k, metrics in self.models.items():
            comparison_data.append({
                'K': k,
                'MSE': metrics['mse'],
                'RMSE': metrics['rmse'],
                'R²': metrics['r2']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('MSE').reset_index(drop=True)
        
        print("Performance Comparison (sorted by MSE):")
        print(comparison_df.to_string(index=False, float_format='%.3f'))
        print()
        
        # Identify best model
        best_model_idx = comparison_df.index[0]
        best_k = comparison_df.iloc[best_model_idx]['K']
        
        print(f"Best Performing Model: K={best_k}")
        print(f"   • Lowest MSE: {comparison_df.iloc[best_model_idx]['MSE']:.3f}")
        print(f"   • RMSE: {comparison_df.iloc[best_model_idx]['RMSE']:.3f}")
        print(f"   • Highest R²: {comparison_df.iloc[best_model_idx]['R²']:.3f}")
        print()
        
        # Create comparison visualizations
        self._create_comparison_plots(comparison_df)
        
        # Insights
        print("Model Selection Insights:")
        print("• Lower MSE/RMSE indicates better prediction accuracy")
        print("• Higher R² indicates better explanation of variance")
        print("• Balance between model complexity and generalization is key")
        print("• Cross-validation would provide more robust model evaluation")
        print()
        
        return best_k
        
    def _create_comparison_plots(self, comparison_df):
        """Create visualization for model comparison."""
        plt.figure(figsize=(15, 5))
        
        # MSE comparison
        plt.subplot(1, 3, 1)
        plt.plot(comparison_df['K'], comparison_df['MSE'], 'bo-', linewidth=2, markersize=8)
        plt.xlabel('K Value')
        plt.ylabel('Mean Squared Error')
        plt.title('MSE vs K Value')
        plt.grid(True, alpha=0.3)
        
        # RMSE comparison
        plt.subplot(1, 3, 2)
        plt.plot(comparison_df['K'], comparison_df['RMSE'], 'go-', linewidth=2, markersize=8)
        plt.xlabel('K Value')
        plt.ylabel('Root Mean Squared Error')
        plt.title('RMSE vs K Value')
        plt.grid(True, alpha=0.3)
        
        # R² comparison
        plt.subplot(1, 3, 3)
        plt.plot(comparison_df['K'], comparison_df['R²'], 'ro-', linewidth=2, markersize=8)
        plt.xlabel('K Value')
        plt.ylabel('R² Score')
        plt.title('R² vs K Value')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def run_complete_analysis(self, k_values=[1, 3, 5, 8, 10, 15]):
        """
        Execute the complete machine learning pipeline.
        
        Args:
            k_values (list): List of K values to test
        """
        print("TV AD BUDGET PREDICTOR - COMPLETE ANALYSIS")
        print("Machine Learning Pipeline for Sales Prediction")
        print("="*60)
        
        try:
            # Phase 1: Data Exploration
            self.explore_data()
            input("\n⏯Press Enter to continue to model training...")
            
            # Phase 2: Model Training
            self.build_knn_models(k_values)
            input("\n⏯Press Enter to continue to visualization...")
            
            # Phase 3: Model Visualization
            self.visualize_models()
            input("\n⏯Press Enter to continue to predictions...")
            
            # Phase 4: Interactive Predictions
            self.make_predictions(interactive=True)
            
            # Phase 5: Model Comparison
            best_k = self.model_comparison()
            
            # Summary
            print("ANALYSIS COMPLETE!")
            print("="*60)
            print("Pipeline Executed Successfully:")
            print("Exploratory Data Analysis")
            print("Model Training and Evaluation")
            print("Model Visualization")
            print("Prediction Generation")
            print("Model Performance Comparison")
            print(f"Best Model Identified: K={best_k}")
            print("\nThis analysis demonstrates:")
            print("• Data-driven decision making")
            print("• Machine learning model development")
            print("• Model evaluation and selection")
            print("• Business insights generation")
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            print("Please check your data and try again.")


def main():
    """Main function to run the TV Ad Budget Predictor."""
    print("TV Ad Budget Predictor")
    print("=====================")
    print("Loading predictor...")
    
    # Initialize predictor
    predictor = TVAdPredictor('advertising_data.csv')
    
    # Run complete analysis
    predictor.run_complete_analysis()


if __name__ == "__main__":
    main()