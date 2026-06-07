from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import pandas as pd
import os
from datetime import datetime
import joblib


class Evaluation:

    def __init__(self, model_path, X_train, X_test, y_test):
        """
        Initializes Evaluation using SAVED models.

        Args:
            model_path (str): Path to saved model folder (timestamp folder)
            X_train: Training features (for feature importance)
            X_test: Test features
            y_test: True labels
        """

        self.model_path = model_path
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test

    
        # Load all saved models
        self.best_models = {}
        for file in os.listdir(model_path):
            if file.endswith(".joblib") and file != "label_encoder.joblib":
                model_name = file.replace(".joblib", "")
                self.best_models[model_name] = joblib.load(
                    os.path.join(model_path, file)
                )

        # Load label encoder
        self.label_encoder = joblib.load(
            os.path.join(model_path, "label_encoder.joblib")
        )

    def evaluate_models(self):
        """
        Evaluates all loaded models using test data.
        """

        X_test = self.X_test
        y_test = self.label_encoder.transform(self.y_test)

        self.results = {}

        for name, model in self.best_models.items():

            print(f"Evaluating {name}...")

            preds = model.predict(X_test)

            # Metrics
            self.results[name] = {
                "accuracy": accuracy_score(y_test, preds),
                "precision": precision_score(y_test, preds, average='weighted'),
                "recall": recall_score(y_test, preds, average='weighted'),
                "f1_score": f1_score(y_test, preds, average='weighted')
            }

            print(self.results[name])
            print("--------------------------")

            # Classification report
            print(classification_report(
                y_test,
                preds,
                target_names=self.label_encoder.classes_
            ))

            # Confusion matrix
            print(confusion_matrix(y_test, preds))


            # Feature importance (tree models only)
            if hasattr(model, "feature_importances_"):

                importances = model.feature_importances_
                feature_names = self.X_train.columns

                feat_imp = pd.Series(
                    importances,
                    index=feature_names
                ).sort_values(ascending=False)

                print(feat_imp.head(20))

        print("Model evaluation completed")
        return self.results

    def export_results(self, filename="experiment_log.csv"):
        """
        Saves evaluation results to CSV log file.
        """

        row_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_folder": self.model_path
        }

        for name, metrics in self.results.items():
            for metric_name, value in metrics.items():
                row_data[f"{name}_{metric_name}"] = value

        df = pd.DataFrame([row_data])

        file_exists = os.path.isfile(filename)

        df.to_csv(
            filename,
            mode='a',
            index=False,
            header=not file_exists
        )

        print(f"Results appended to '{filename}'.")