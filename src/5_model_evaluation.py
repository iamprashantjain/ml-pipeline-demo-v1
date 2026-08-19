import os
import sys
import json
import pandas as pd
import numpy as np
import joblib
import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report
from mylogging import logging
from myexception import customexception

def load_params(params_path: str):
    """Load parameters from YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.info("Parameters loaded successfully.")
        return params
    except Exception as e:
        logging.info("Error loading params.yaml")
        raise customexception(e, sys)

def load_test_data(file_path: str):
    """Load test data from CSV file."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Test data file not found: {file_path}")
        test_data = pd.read_csv(file_path)
        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values
        logging.info(f"Test data loaded from {file_path}")
        return X_test, y_test
    except Exception as e:
        logging.info("Error loading test data")
        raise customexception(e, sys)

def load_model(model_path: str):
    """Load trained model from joblib file."""
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if os.path.getsize(model_path) == 0:
            raise ValueError(f"Model file is empty: {model_path}")
        
        model = joblib.load(model_path)
        logging.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logging.info("Error loading model")
        raise customexception(e, sys)

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    try:
        y_pred = model.predict(X_test)
        
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        except (AttributeError, IndexError):
            y_pred_proba = None
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
        report = classification_report(y_test, y_pred, output_dict=True)
        
        logging.info(f"Model evaluated with Accuracy: {accuracy:.4f}")
        return accuracy, precision, recall, auc, report, y_pred
    except Exception as e:
        logging.info("Error during model evaluation")
        raise customexception(e, sys)

def save_metrics(accuracy, precision, recall, auc, report, metrics_path: str):
    """Save evaluation metrics to YAML file."""
    try:
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        
        metrics = {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "auc": round(auc, 4) if auc is not None else None,
            "classification_report": {
                "class_0": {
                    "precision": report["0"]["precision"],
                    "recall": report["0"]["recall"],
                    "f1-score": report["0"]["f1-score"]
                },
                "class_1": {
                    "precision": report["1"]["precision"],
                    "recall": report["1"]["recall"],
                    "f1-score": report["1"]["f1-score"]
                }
            }
        }
        
        with open(metrics_path, 'w') as file:
            yaml.dump(metrics, file, default_flow_style=False)
        
        logging.info(f"Metrics saved to {metrics_path}")
        print(f"✓ Metrics saved to: {metrics_path}")
    except Exception as e:
        logging.info("Error saving metrics")
        raise customexception(e, sys)

def save_predictions(y_test, y_pred, output_path: str):
    """Save predictions to CSV file for analysis."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results_df = pd.DataFrame({
            'actual': y_test,
            'predicted': y_pred
        })
        results_df.to_csv(output_path, index=False)
        logging.info(f"Predictions saved to {output_path}")
    except Exception as e:
        logging.info("Error saving predictions")
        raise customexception(e, sys)

# ★★★ ADD THIS FUNCTION ★★★
def save_experiment_info(model_path, metrics_path, output_path: str):
    """Save experiment information to JSON file."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        experiment_info = {
            "model_path": model_path,
            "metrics_path": metrics_path,
            "timestamp": pd.Timestamp.now().isoformat(),
            "status": "completed"
        }
        with open(output_path, 'w') as file:
            json.dump(experiment_info, file, indent=4)
        logging.info(f"Experiment info saved to {output_path}")
        print(f"✓ Experiment info saved to: {output_path}")
    except Exception as e:
        logging.info("Error saving experiment info")
        raise customexception(e, sys)

def main():
    try:
        # Load parameters
        params = load_params("params.yaml")
        eval_params = params.get("model_evaluation", {})
        
        # Get paths from parameters
        model_path = eval_params.get("model_path", "artifacts/model/logistic_regression_model.pkl")
        test_data_path = eval_params.get("input_test", "artifacts/data/vectorized/test_vectorized.csv")
        metrics_path = eval_params.get("metrics_path", "reports/metrics.yaml")
        predictions_path = eval_params.get("predictions_path", "reports/predictions.csv")
        experiment_info_path = eval_params.get("experiment_info_path", "reports/experiment_info.json")  # ← ADD THIS
        
        logging.info("Starting model evaluation pipeline...")
        print("="*50)
        print("MODEL EVALUATION PIPELINE")
        print("="*50)
        
        # Load model
        logging.info(f"Loading model from {model_path}")
        model = load_model(model_path)
        print(f"✓ Model loaded from: {model_path}")
        
        # Load test data
        logging.info(f"Loading test data from {test_data_path}")
        X_test, y_test = load_test_data(test_data_path)
        print(f"✓ Test data loaded: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        
        # Evaluate model
        logging.info("Evaluating model...")
        accuracy, precision, recall, auc, report, y_pred = evaluate_model(model, X_test, y_test)
        
        # Print results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"AUC:       {auc:.4f}" if auc else "AUC:       N/A")
        print("="*50)
        
        # Save metrics
        logging.info("Saving metrics...")
        save_metrics(accuracy, precision, recall, auc, report, metrics_path)
        
        # Save predictions
        logging.info("Saving predictions...")
        save_predictions(y_test, y_pred, predictions_path)
        
        # ★★★ SAVE EXPERIMENT INFO (ADD THIS) ★★★
        logging.info("Saving experiment info...")
        save_experiment_info(model_path, metrics_path, experiment_info_path)
        
        logging.info("Model evaluation pipeline completed successfully!")
        print("\n✓ Model evaluation completed successfully!")
        
    except Exception as e:
        logging.info("Exception in model_evaluation main function.")
        print(f"\n✗ Error: {str(e)}")
        raise customexception(e, sys)

if __name__ == "__main__":
    main()