import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
import optuna

# Load and preprocess data
def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def preprocess_data(train_df, test_df):
    def convert_to_numeric(df):
        if df['is_night_game'].dtype == 'object':
            # If it's an object type, try to convert
            df['is_night_game'] = df['is_night_game'].map({'yes': 1, 'no': 0, 'true': 1, 'false': 0})
        elif pd.api.types.is_numeric_dtype(df['is_night_game']):
            # If it's already numeric, ensure it's 0 or 1
            df['is_night_game'] = df['is_night_game'].astype(int).clip(0, 1)
        else:
            # Last resort conversion
            df['is_night_game'] = df['is_night_game'].astype(int).clip(0, 1)
        return df

    # Apply conversion to both train and test dataframes
    train_df = convert_to_numeric(train_df)
    test_df = convert_to_numeric(test_df)
    features = [
        # Game context features
        'is_night_game', 'home_team_rest', 'away_team_rest',
        'home_pitcher_rest', 'away_pitcher_rest',
        
        # Recent performance features (10-game rolling averages)
        'home_batting_batting_avg_10RA', 'home_batting_onbase_perc_10RA',
        'home_batting_onbase_plus_slugging_10RA', 'home_batting_RBI_10RA',
        'away_batting_batting_avg_10RA', 'away_batting_onbase_perc_10RA',
        'away_batting_onbase_plus_slugging_10RA', 'away_batting_RBI_10RA',
        
        # Pitching performance
        'home_pitching_earned_run_avg_10RA', 'home_pitching_SO_batters_faced_10RA',
        'away_pitching_earned_run_avg_10RA', 'away_pitching_SO_batters_faced_10RA',
        
        # Seasonal team performance
        'home_team_wins_mean', 'away_team_wins_mean',
        'home_team_spread_mean', 'away_team_spread_mean',
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
    X_test = test_df[features].fillna(test_df[features].mean())
    test_ids = test_df['id']
    return X_train, X_test, y_train, test_ids
    
# Train LightGBM with hyperparameter tuning using Optuna
def optimize_lightgbm(trial, X_train, y_train):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0)
    }
    model = LGBMClassifier(**param, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    return np.mean(scores)

# Train the best model and save predictions
def train_best_lightgbm(X_train, X_test, y_train, best_params):
    model = LGBMClassifier(**best_params, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return model, predictions

def save_predictions(test_ids, predictions, output_path):
    output_df = pd.DataFrame({
        'id': test_ids,
        'home_team_win': predictions
    })
    output_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

def main():
    # File paths
    train_path = 'train_data.csv'
    test_path = 'same_season_test_data.csv'
    output_path = 'best_lightgbm_predictions.csv'
    
    # Load and preprocess data
    train_df, test_df = load_data(train_path, test_path)
    X_train, X_test, y_train, test_ids = preprocess_data(train_df, test_df)
    
    # Hyperparameter tuning with Optuna
    print("Tuning LightGBM hyperparameters...")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: optimize_lightgbm(trial, X_train, y_train), n_trials=50)
    
    # Retrieve the best parameters
    print("Best parameters found:", study.best_params)
    best_params = study.best_params

    print(best_params)
    
    # Train the best model and make predictions
    print("Training the best LightGBM model...")
    best_model, predictions = train_best_lightgbm(X_train, X_test, y_train, best_params)
    
    # Save predictions
    save_predictions(test_ids, predictions, output_path)

if __name__ == "__main__":
    main()

