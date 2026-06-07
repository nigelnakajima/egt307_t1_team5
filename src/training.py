from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pandas as pd
import json
import os
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from scipy.stats import randint, uniform
from datetime import datetime
from sklearn.utils.class_weight import compute_sample_weight
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier, log_evaluation, early_stopping
from imblearn.ensemble import BalancedRandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import importlib
import joblib
from config import load_config
config = load_config("config.yaml")
random_state = config["random"]
test_size = config["test_size"]

class ModelTrainer:
    '''
    Handles seperating data into training and validation sets, model training and evaluation
    for multiple machine learning classifiers.
    '''

    def __init__(self, target_col, X_train, y_train, X_test, y_test):
        '''
        Initializes the ModelTrainer class.

        Args:
            target_col (str): The target column to predict
        '''

        # Store target column name
        self.target_col = target_col
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        def get_dist(key, val):
            # Only use randint/uniform if the parameter is a range [low, high]
            # For parameters like max_depth that include 'None', just return the list
            if key in ['n_estimators', 'min_samples_split'] and isinstance(val, list) and len(val) == 2:
                return randint(val[0], val[1])
            if key == 'learning_rate':
                return uniform(val[0], val[1])
            
            # If it's a list with more than 2 elements, return it as-is
            return val
        
        # Define hyperparameter grids
        self.param_distributions = {
            model: {k: get_dist(k, v) for k, v in params.items()}
            for model, params in config['model_params'].items()
        }


        def get_class(module_name, class_name):
            module = importlib.import_module(module_name)
            return getattr(module, class_name)

        def build_models(config, y_train):
            initialised_models = {}
            
            for model_name, cfg in config['models'].items():
                # Look up definition
                definition = config['model_definitions'][model_name]
                
                # Import class dynamically
                model_class = get_class(definition['module'], definition['class'])
                
                # Prepare params
                params = cfg['params'].copy()
                if cfg['name'] == "XGBoost":
                    params['num_class'] = len(np.unique(y_train))
                    
                initialised_models[model_name] = model_class(**params)
                
            return initialised_models
        
        # Define machine learning models
        self.models = build_models(config, self.y_train)

        # Create label encoder for target variable
        self.label_encoder = LabelEncoder()

    def train_models(self):
        '''
        Trains and optimises all configured machine learning models.

        Args:
            X_train: Training features
            y_train: Training labels
        '''
        # Split into train and validation for early stopping
        y_encoded = self.label_encoder.fit_transform(self.y_train)
        X_train_sub, X_val, y_train_sub, y_val = train_test_split(
            self.X_train, y_encoded, 
            test_size=test_size, stratify=y_encoded, random_state=random_state
        )
        # Recalculate weights on this subset
        self.sample_weights_sub = compute_sample_weight(
            class_weight='balanced', 
            y=y_train_sub
        )
        

        self.best_models = {}
        cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
        print("Training machine learning models...")
        
        # Train and store each model
        for name, model in self.models.items():
            print(f"Optimising {name}...")
            fit_params = {}
            if name == "XGBoost":
                model.set_params(early_stopping_rounds=5)
                fit_params = {'eval_set': [(X_val, y_val)], 'verbose': False}

            search = RandomizedSearchCV(
                    model, 
                    param_distributions=self.param_distributions[name],
                    n_iter=25, # Number of parameter settings to try
                    cv=cv_strategy,      # 3-fold cross-validation
                    scoring="f1_weighted",
                    n_jobs=-1, # Use all CPU cores
                    random_state=42
                )

            search.fit(
                X_train_sub, y_train_sub, 
                sample_weight=self.sample_weights_sub,
                **fit_params
            )

            self.best_models[name] = search.best_estimator_
            print(f"Best params for {name}: {search.best_params_}")
            
        print("All models trained successfully")
        # save model weights here
        self.save_models()


    def save_models(self, save_dir="saved_models"):
        """
        Saves trained models (weights) and label encoder.
        """

        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = os.path.join(save_dir, timestamp)
        os.makedirs(folder, exist_ok=True)

        # Save each trained model
        for name, model in self.best_models.items():
            file_path = os.path.join(folder, f"{name}.joblib")
            joblib.dump(model, file_path)

        # Save label encoder
        joblib.dump(self.label_encoder, os.path.join(folder, "label_encoder.joblib"))

        print(f"Models saved successfully in: {folder}")


    def run(self):
        '''
        Executes the complete machine learning workflow.

        Args:
            df (pd.DataFrame): Input dataset
        '''
        
        print("--------------------------")

        # Train machine learning models
        self.train_models()

        print("-------------------------")
        print("Machine learning completed")
        
        return self.best_models, self.label_encoder