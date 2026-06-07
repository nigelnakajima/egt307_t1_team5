'''
main.py: Main file to be ran
Program Flow:
- Load Config
- Load Data
- Clean Data (class)
- Train Model (class)
- Export Results
'''
from config import load_config
from ingestion import load_data_from_db
from cleaning import DataCleaner
from preprocessing import DataPreprocessor
from synthetic_data_generation import generate_data
from training import ModelTrainer
from evaluate import Evaluation
import os

try:
    # Load Data
    df = load_data_from_db()

    # Clean Data
    cleaner = DataCleaner()
    df = cleaner.process(df)
    
    # Split Data
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.process(df)

    # Generate Synthetic Data
    X_train, y_train = generate_data(X_train, y_train)
    
    # Train Models
    trainer = ModelTrainer("activity_level", X_train, y_train, X_test, y_test)
    trainer.run()

    # Load Saved Models
    save_root = "saved_models"
    latest_folder = max(
        [os.path.join(save_root, d) for d in os.listdir(save_root)],
        key=os.path.getctime
    )

    print(f"Using saved models from: {latest_folder}")

    # Evaluate Models 
    evaluator = Evaluation(
        model_path=latest_folder,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test
    )

    evaluator.evaluate_models()

except ValueError as e:
    print(f"Aborting pipeline: {e}")
    exit(1)