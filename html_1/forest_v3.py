import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
import itertools
import csv

def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def preprocess_data(train_df, test_df):
    # Reuse the feature list from the original code
    features = [
      # Game context features
        'is_night_game', 'home_team_rest', 'away_team_rest', 
        'home_pitcher_rest', 'away_pitcher_rest', 'season',
        
        # Recent performance features (10-game rolling averages)
        'home_batting_batting_avg_10RA', 'home_batting_onbase_perc_10RA', 
        'home_batting_onbase_plus_slugging_10RA', 'home_batting_RBI_10RA',
        'away_batting_batting_avg_10RA', 'away_batting_onbase_perc_10RA', 
        'away_batting_onbase_plus_slugging_10RA', 'away_batting_RBI_10RA',
        
        # Pitching performance
        'home_pitching_earned_run_avg_10RA', 'home_pitching_SO_batters_faced_10RA',
        'home_pitching_H_batters_faced_10RA', 'home_pitching_BB_batters_faced_10RA',
        'away_pitching_earned_run_avg_10RA', 'away_pitching_SO_batters_faced_10RA',
        'away_pitching_H_batters_faced_10RA', 'away_pitching_BB_batters_faced_10RA',
        
        # Seasonal team performance
        'home_team_wins_mean', 'away_team_wins_mean',
        'home_team_spread_mean', 'away_team_spread_mean',
        'home_team_errors_mean', 'away_team_errors_mean',
        
        # Batting seasonal statistics
        'home_batting_leverage_index_avg_mean', 'away_batting_leverage_index_avg_mean',
        'home_batting_wpa_bat_mean', 'away_batting_wpa_bat_mean',
                # new
        'away_batting_wpa_bat_std', 'away_batting_wpa_bat_skew',
        'away_batting_RBI_mean', 'away_batting_RBI_std', 'away_batting_RBI_skew',
        'home_pitching_earned_run_avg_mean', 'home_pitching_earned_run_avg_std',
        'home_pitching_earned_run_avg_skew', 'home_pitching_SO_batters_faced_mean',
        'home_pitching_SO_batters_faced_std', 'home_pitching_SO_batters_faced_skew',
        'home_pitching_H_batters_faced_mean', 'home_pitching_H_batters_faced_std',
        'home_pitching_H_batters_faced_skew', 'home_pitching_BB_batters_faced_mean',
        'home_pitching_BB_batters_faced_std', 'home_pitching_BB_batters_faced_skew',
        'home_pitching_leverage_index_avg_mean', 'home_pitching_leverage_index_avg_std',
        'home_pitching_leverage_index_avg_skew', 'home_pitching_wpa_def_mean',
        'home_pitching_wpa_def_std', 'home_pitching_wpa_def_skew',
        'away_pitching_earned_run_avg_mean', 'away_pitching_earned_run_avg_std',
        'away_pitching_earned_run_avg_skew', 'away_pitching_SO_batters_faced_mean',
        'away_pitching_SO_batters_faced_std', 'away_pitching_SO_batters_faced_skew',
        'away_pitching_H_batters_faced_mean', 'away_pitching_H_batters_faced_std',
        'away_pitching_H_batters_faced_skew', 'away_pitching_BB_batters_faced_mean',
        'away_pitching_BB_batters_faced_std', 'away_pitching_BB_batters_faced_skew',
        'away_pitching_leverage_index_avg_mean', 'away_pitching_leverage_index_avg_std',
        'away_pitching_leverage_index_avg_skew', 'away_pitching_wpa_def_mean',
        'away_pitching_wpa_def_std', 'away_pitching_wpa_def_skew',
        'home_pitcher_earned_run_avg_mean', 'home_pitcher_earned_run_avg_std',
        'home_pitcher_earned_run_avg_skew', 'home_pitcher_SO_batters_faced_mean',
        'home_pitcher_SO_batters_faced_std', 'home_pitcher_SO_batters_faced_skew',
        'home_pitcher_H_batters_faced_mean', 'home_pitcher_H_batters_faced_std',
        'home_pitcher_H_batters_faced_skew', 'home_pitcher_BB_batters_faced_mean',
        'home_pitcher_BB_batters_faced_std', 'home_pitcher_BB_batters_faced_skew',
        'home_pitcher_leverage_index_avg_mean', 'home_pitcher_leverage_index_avg_std',
        'home_pitcher_leverage_index_avg_skew', 'home_pitcher_wpa_def_mean',
        'home_pitcher_wpa_def_std', 'home_pitcher_wpa_def_skew',
        'away_pitcher_earned_run_avg_mean', 'away_pitcher_earned_run_avg_std',
        'away_pitcher_earned_run_avg_skew', 'away_pitcher_SO_batters_faced_mean',
        'away_pitcher_SO_batters_faced_std', 'away_pitcher_SO_batters_faced_skew',
        'away_pitcher_H_batters_faced_mean', 'away_pitcher_H_batters_faced_std',
        'away_pitcher_H_batters_faced_skew', 'away_pitcher_BB_batters_faced_mean',
        'away_pitcher_BB_batters_faced_std', 'away_pitcher_BB_batters_faced_skew',
        'away_pitcher_leverage_index_avg_mean', 'away_pitcher_leverage_index_avg_std',
        'away_pitcher_leverage_index_avg_skew', 'away_pitcher_wpa_def_mean',
        'away_pitcher_wpa_def_std', 'away_pitcher_wpa_def_skew'
    ]
  
    
    X_train = train_df[features].fillna(train_df[features].mean())
    y_train = train_df['home_team_win']
    return X_train, y_train

def grid_search_random_forest(X_train, y_train):
    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10, 15]
    }
    
    # Prepare to track results
    results = []
    
    # Generate all combinations of hyperparameters
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    # Split data for validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Iterate through hyperparameter combinations
    for params in combinations:
        try:
            # Create and train the model
            model = RandomForestClassifier(
                n_estimators=params['n_estimators'], 
                max_depth=params['max_depth'], 
                min_samples_split=params['min_samples_split'], 
                random_state=42
            )
            
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('feature_selection', SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))),
                ('classifier', model)
            ])
            
            # Fit the model
            pipeline.fit(X_train_split, y_train_split)
            
            # Predict and calculate accuracy
            predictions = pipeline.predict(X_val_split)
            accuracy = accuracy_score(y_val_split, predictions)
            
            # Store results
            result = {
                'n_estimators': params['n_estimators'],
                'max_depth': params['max_depth'],
                'min_samples_split': params['min_samples_split'],
                'accuracy': accuracy
            }
            results.append(result)
            
            print(f"Params: {params}, Accuracy: {accuracy:.4f}")
        
        except Exception as e:
            print(f"Error with params {params}: {e}")
    
    # Find and return the best combination
    best_result = max(results, key=lambda x: x['accuracy'])
    return best_result

def save_results(results, output_path='hyperparameter_tuning_results.csv'):
    # Save all results to CSV
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['n_estimators', 'max_depth', 'min_samples_split', 'accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"All results saved to {output_path}")

def main_hyperparameter_tuning():
    # Load data
    train_df, _ = load_data('train_data.csv', 'same_season_test_data.csv')
    
    # Preprocess data
    X_train, y_train = preprocess_data(train_df, None)
    
    # Perform grid search
    best_result = grid_search_random_forest(X_train, y_train)
    
    # Print and save best result
    print("\nBest Hyperparameters:")
    print(f"n_estimators: {best_result['n_estimators']}")
    print(f"max_depth: {best_result['max_depth']}")
    print(f"min_samples_split: {best_result['min_samples_split']}")
    print(f"Accuracy: {best_result['accuracy']:.4f}")

if __name__ == "__main__":
    main_hyperparameter_tuning()