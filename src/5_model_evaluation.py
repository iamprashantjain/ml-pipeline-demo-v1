import os
import sys
import json

import pandas as pd
import joblib
import yaml

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report
)

from mylogging import logging
from myexception import customexception

import mlflow
import mlflow.sklearn

from dotenv import load_dotenv

load_dotenv()


# ============================================================
# DAGSHUB / MLFLOW CONFIGURATION
# ============================================================

DAGSHUB_USERNAME = "iamprashantjain"
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")
REPO_NAME = "ml-pipeline-demo-v1"

if not DAGSHUB_TOKEN:
    raise ValueError(
        "DAGSHUB_TOKEN is not set in the environment."
    )

mlflow.set_tracking_uri(
    f"https://{DAGSHUB_USERNAME}:{DAGSHUB_TOKEN}"
    f"@dagshub.com/{DAGSHUB_USERNAME}/{REPO_NAME}.mlflow"
)


# ============================================================
# LOAD PARAMETERS
# ============================================================

def load_params(params_path: str):
    """Load parameters from YAML file."""

    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)

        logging.info("Parameters loaded successfully.")

        return params

    except Exception as e:
        logging.error("Error loading params.yaml")
        raise customexception(e, sys)


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_data(file_path: str):
    """Load test data from CSV file."""

    try:

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Test data file not found: {file_path}"
            )

        test_data = pd.read_csv(file_path)

        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        logging.info(
            f"Test data loaded from {file_path}"
        )

        return X_test, y_test

    except Exception as e:

        logging.error(
            "Error loading test data"
        )

        raise customexception(e, sys)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(model_path: str):
    """Load trained model from joblib file."""

    try:

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        if os.path.getsize(model_path) == 0:
            raise ValueError(
                f"Model file is empty: {model_path}"
            )

        model = joblib.load(model_path)

        logging.info(
            f"Model loaded from {model_path}"
        )

        return model

    except Exception as e:

        logging.error(
            "Error loading model"
        )

        raise customexception(e, sys)


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""

    try:

        y_pred = model.predict(X_test)

        # ----------------------------------------------------
        # Prediction probabilities
        # ----------------------------------------------------

        try:

            y_pred_proba = (
                model.predict_proba(X_test)[:, 1]
            )

        except (AttributeError, IndexError):

            y_pred_proba = None

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            average="binary",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            average="binary",
            zero_division=0
        )

        if y_pred_proba is not None:

            auc = roc_auc_score(
                y_test,
                y_pred_proba
            )

        else:

            auc = None

        report = classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0
        )

        logging.info(
            f"Model evaluated - "
            f"Accuracy: {accuracy:.4f}, "
            f"Precision: {precision:.4f}, "
            f"Recall: {recall:.4f}"
        )

        return (
            accuracy,
            precision,
            recall,
            auc,
            report,
            y_pred
        )

    except Exception as e:

        logging.error(
            "Error during model evaluation"
        )

        raise customexception(e, sys)


# ============================================================
# SAVE METRICS
# ============================================================

def save_metrics(
    accuracy,
    precision,
    recall,
    auc,
    report,
    metrics_path: str
):
    """Save evaluation metrics to YAML file."""

    try:

        directory = os.path.dirname(
            metrics_path
        )

        if directory:
            os.makedirs(
                directory,
                exist_ok=True
            )

        metrics = {
            "accuracy": round(
                accuracy,
                4
            ),

            "precision": round(
                precision,
                4
            ),

            "recall": round(
                recall,
                4
            ),

            "auc": (
                round(auc, 4)
                if auc is not None
                else None
            ),

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

        with open(
            metrics_path,
            "w"
        ) as file:

            yaml.dump(
                metrics,
                file,
                default_flow_style=False
            )

        logging.info(
            f"Metrics saved to {metrics_path}"
        )

        print(
            f"✓ Metrics saved to: {metrics_path}"
        )

    except Exception as e:

        logging.error(
            "Error saving metrics"
        )

        raise customexception(e, sys)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

def save_predictions(
    y_test,
    y_pred,
    output_path: str
):
    """Save predictions to CSV file."""

    try:

        directory = os.path.dirname(
            output_path
        )

        if directory:
            os.makedirs(
                directory,
                exist_ok=True
            )

        results_df = pd.DataFrame({
            "actual": y_test,
            "predicted": y_pred
        })

        results_df.to_csv(
            output_path,
            index=False
        )

        logging.info(
            f"Predictions saved to {output_path}"
        )

    except Exception as e:

        logging.error(
            "Error saving predictions"
        )

        raise customexception(e, sys)


# ============================================================
# SAVE EXPERIMENT INFO
# ============================================================

def save_experiment_info(
    model_name,
    model_path,
    mlflow_model_name,
    mlflow_model_id,
    mlflow_model_uri,
    metrics_path,
    run_id,
    output_path: str
):
    """
    Save experiment information.

    model_path:
        Local .pkl model path.

    mlflow_model_name:
        Name used when logging the MLflow model.

    mlflow_model_id:
        MLflow 3 logged-model ID.

    mlflow_model_uri:
        URI returned by MLflow for the logged model.
    """

    try:

        directory = os.path.dirname(
            output_path
        )

        if directory:
            os.makedirs(
                directory,
                exist_ok=True
            )

        experiment_info = {

            # Registered model name
            "model_name": model_name,

            # Local model
            "model_path": model_path,

            # MLflow model name
            "mlflow_model_name": mlflow_model_name,

            # MLflow 3 logged model ID
            "mlflow_model_id": mlflow_model_id,

            # MLflow model URI
            "mlflow_model_uri": mlflow_model_uri,

            # Run information
            "run_id": run_id,

            # Metrics
            "metrics_path": metrics_path,

            "timestamp": (
                pd.Timestamp.now().isoformat()
            ),

            "status": "completed"
        }

        with open(
            output_path,
            "w"
        ) as file:

            json.dump(
                experiment_info,
                file,
                indent=4
            )

        logging.info(
            f"Experiment info saved to {output_path}"
        )

        print(
            f"✓ Experiment info saved to: "
            f"{output_path}"
        )

    except Exception as e:

        logging.error(
            "Error saving experiment info"
        )

        raise customexception(e, sys)


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Set MLflow experiment
    # --------------------------------------------------------

    mlflow.set_experiment(
        "dvc-pipeline"
    )

    with mlflow.start_run() as run:

        try:

            # =================================================
            # LOAD PARAMETERS
            # =================================================

            params = load_params(
                "params.yaml"
            )

            eval_params = params.get(
                "model_evaluation",
                {}
            )

            # =================================================
            # PATHS
            # =================================================

            model_path = eval_params.get(
                "model_path",
                "artifacts/model/logistic_regression_model.pkl"
            )

            model_name = eval_params.get(
                "model_name",
                "logistic_regression"
            )

            test_data_path = eval_params.get(
                "input_test",
                "artifacts/data/vectorized/test_vectorized.csv"
            )

            metrics_path = eval_params.get(
                "metrics_path",
                "reports/metrics.yaml"
            )

            predictions_path = eval_params.get(
                "predictions_path",
                "reports/predictions.csv"
            )

            experiment_info_path = eval_params.get(
                "experiment_info_path",
                "reports/experiment_info.json"
            )

            # ------------------------------------------------
            # MLflow model name
            # ------------------------------------------------

            mlflow_model_name = model_name

            # =================================================
            # START
            # =================================================

            print("=" * 60)
            print("MODEL EVALUATION PIPELINE")
            print("=" * 60)

            # =================================================
            # LOAD MODEL
            # =================================================

            logging.info(
                f"Loading model from {model_path}"
            )

            model = load_model(
                model_path
            )

            print(
                f"✓ Model loaded from: "
                f"{model_path}"
            )

            # =================================================
            # LOG PARAMETERS
            # =================================================

            if hasattr(
                model,
                "get_params"
            ):

                params_dict = (
                    model.get_params()
                )

                for (
                    param_name,
                    param_value
                ) in params_dict.items():

                    if isinstance(
                        param_value,
                        (
                            str,
                            int,
                            float,
                            bool
                        )
                    ):

                        mlflow.log_param(
                            param_name,
                            param_value
                        )

            mlflow.log_param(
                "model_name",
                model_name
            )

            # =================================================
            # LOAD TEST DATA
            # =================================================

            logging.info(
                f"Loading test data from "
                f"{test_data_path}"
            )

            X_test, y_test = load_test_data(
                test_data_path
            )

            print(
                f"✓ Test data loaded: "
                f"{X_test.shape[0]} samples, "
                f"{X_test.shape[1]} features"
            )

            # =================================================
            # EVALUATE MODEL
            # =================================================

            (
                accuracy,
                precision,
                recall,
                auc,
                report,
                y_pred
            ) = evaluate_model(
                model,
                X_test,
                y_test
            )

            # =================================================
            # DISPLAY RESULTS
            # =================================================

            print("\n" + "=" * 60)
            print("EVALUATION RESULTS")
            print("=" * 60)

            print(
                f"Accuracy:  {accuracy:.4f}"
            )

            print(
                f"Precision: {precision:.4f}"
            )

            print(
                f"Recall:    {recall:.4f}"
            )

            if auc is not None:

                print(
                    f"AUC:       {auc:.4f}"
                )

            else:

                print(
                    "AUC:       N/A"
                )

            print("=" * 60)

            # =================================================
            # LOG METRICS
            # =================================================

            mlflow.log_metric(
                "accuracy",
                accuracy
            )

            mlflow.log_metric(
                "precision",
                precision
            )

            mlflow.log_metric(
                "recall",
                recall
            )

            if auc is not None:

                mlflow.log_metric(
                    "auc",
                    auc
                )

            # =================================================
            # SAVE METRICS
            # =================================================

            save_metrics(
                accuracy,
                precision,
                recall,
                auc,
                report,
                metrics_path
            )

            mlflow.log_artifact(
                metrics_path
            )

            # =================================================
            # SAVE PREDICTIONS
            # =================================================

            save_predictions(
                y_test,
                y_pred,
                predictions_path
            )

            mlflow.log_artifact(
                predictions_path
            )

            # =================================================
            # LOG MODEL
            # =================================================
            #
            # IMPORTANT:
            #
            # MLflow 3:
            #
            # Use `name=`
            #
            # Do NOT use:
            #
            # artifact_path=
            #
            # The returned ModelInfo contains:
            #
            # model_id
            # model_uri
            #
            # We save these values and use model_id
            # during registration.
            # =================================================

            print(
                "\nLogging model to MLflow..."
            )

            logged_model = (
                mlflow.sklearn.log_model(
                    sk_model=model,
                    name=mlflow_model_name
                )
            )

            print(
                "✓ Model logged successfully"
            )

            print(
                f"✓ MLflow Model Name: "
                f"{logged_model.name}"
            )

            print(
                f"✓ MLflow Model ID: "
                f"{logged_model.model_id}"
            )

            print(
                f"✓ MLflow Model URI: "
                f"{logged_model.model_uri}"
            )

            # =================================================
            # SAVE EXPERIMENT INFO
            # =================================================

            save_experiment_info(

                model_name=model_name,

                model_path=model_path,

                mlflow_model_name=(
                    logged_model.name
                ),

                mlflow_model_id=(
                    logged_model.model_id
                ),

                mlflow_model_uri=(
                    logged_model.model_uri
                ),

                metrics_path=metrics_path,

                run_id=run.info.run_id,

                output_path=(
                    experiment_info_path
                )
            )

            # =================================================
            # LOG EXPERIMENT INFO
            # =================================================

            mlflow.log_artifact(
                experiment_info_path
            )

            # =================================================
            # COMPLETE
            # =================================================

            logging.info(
                "Model evaluation pipeline "
                "completed successfully!"
            )

            print(
                "\n✓ Model evaluation completed successfully!"
            )

            print(
                f"✓ Model Name: "
                f"{model_name}"
            )

            print(
                f"✓ MLflow Model ID: "
                f"{logged_model.model_id}"
            )

            print(
                f"✓ MLflow Run ID: "
                f"{run.info.run_id}"
            )

            print(
                f"✓ Experiment Info: "
                f"{experiment_info_path}"
            )

        except Exception as e:

            logging.error(
                "Exception in model_evaluation main function."
            )

            print(
                f"\n✗ Error: {str(e)}"
            )

            raise customexception(
                e,
                sys
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()