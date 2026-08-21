import os
import sys
import json

import mlflow
from mlflow.tracking import MlflowClient

from mylogging import logging
from myexception import customexception

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
# LOAD EXPERIMENT INFO
# ============================================================

def load_model_info(file_path: str) -> dict:
    """Load model information from experiment_info.json."""

    try:

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Experiment info file not found: "
                f"{file_path}"
            )

        with open(
            file_path,
            "r"
        ) as file:

            model_info = json.load(file)

        # ----------------------------------------------------
        # Required fields
        # ----------------------------------------------------

        required_fields = [
            "model_name",
            "run_id",
            "mlflow_model_id"
        ]

        missing_fields = [
            field
            for field in required_fields
            if field not in model_info
        ]

        if missing_fields:

            raise KeyError(
                "Missing fields in "
                f"{file_path}: "
                f"{missing_fields}"
            )

        logging.info(
            f"Model info loaded from {file_path}"
        )

        return model_info

    except FileNotFoundError:

        logging.error(
            f"File not found: {file_path}"
        )

        raise

    except Exception as e:

        logging.error(
            f"Error loading model info: {e}"
        )

        raise


# ============================================================
# REGISTER MODEL
# ============================================================

def register_model(
    model_name: str,
    model_info: dict
):
    """Register MLflow 3 logged model."""

    try:

        run_id = model_info[
            "run_id"
        ]

        model_id = model_info[
            "mlflow_model_id"
        ]

        model_uri = (
            f"models:/{model_id}"
        )

        # ----------------------------------------------------
        # Display information
        # ----------------------------------------------------

        print(
            "\n" + "=" * 60
        )

        print(
            "MODEL REGISTRATION"
        )

        print(
            "=" * 60
        )

        print(
            f"Model Name:       {model_name}"
        )

        print(
            f"Run ID:           {run_id}"
        )

        print(
            f"Logged Model ID:  {model_id}"
        )

        print(
            f"Model URI:        {model_uri}"
        )

        print(
            "=" * 60
        )

        logging.info(
            f"Registering logged model "
            f"{model_id}"
        )

        # ----------------------------------------------------
        # Verify MLflow run
        # ----------------------------------------------------

        client = MlflowClient()

        run = client.get_run(
            run_id
        )

        if run is None:

            raise ValueError(
                f"MLflow run not found: "
                f"{run_id}"
            )

        logging.info(
            f"MLflow run found: {run_id}"
        )

        # ----------------------------------------------------
        # Register logged model
        # ----------------------------------------------------
        #
        # MLflow 3 supports:
        #
        # models:/<model_id>
        #
        # This avoids the old:
        #
        # runs:/<run_id>/<artifact_path>
        #
        # resolution problem.
        # ----------------------------------------------------

        model_version = (
            mlflow.register_model(
                model_uri=model_uri,
                name=model_name
            )
        )

        print(
            "\n✓ Model registered successfully!"
        )

        print(
            f"✓ Registered Model: "
            f"{model_name}"
        )

        print(
            f"✓ Model Version: "
            f"{model_version.version}"
        )

        logging.info(
            f"Model {model_name} "
            f"version {model_version.version} "
            f"registered successfully."
        )

        # ----------------------------------------------------
        # Transition to Staging
        # ----------------------------------------------------
        #
        # NOTE:
        # MLflow stages are deprecated in newer MLflow
        # versions. If your DagsHub backend supports them,
        # this will work. Otherwise use aliases instead.
        # ----------------------------------------------------

        try:

            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage="Staging"
            )

            print(
                f"✓ Model version "
                f"{model_version.version} "
                f"transitioned to Staging."
            )

            logging.info(
                f"Model {model_name} "
                f"version {model_version.version} "
                f"transitioned to Staging."
            )

        except Exception as stage_error:

            logging.warning(
                f"Model registered, but "
                f"Staging transition failed: "
                f"{stage_error}"
            )

            print(
                "\n⚠ Model was registered successfully, "
                "but Staging transition failed."
            )

            print(
                f"Reason: {stage_error}"
            )

        print(
            "=" * 60
        )

        return model_version

    except Exception as e:

        logging.error(
            f"Error during model registration: {e}"
        )

        raise


# ============================================================
# MAIN
# ============================================================

def main():

    try:

        # ----------------------------------------------------
        # Experiment information
        # ----------------------------------------------------

        model_info_path = (
            "reports/experiment_info.json"
        )

        # ----------------------------------------------------
        # Load model information
        # ----------------------------------------------------

        model_info = load_model_info(
            model_info_path
        )

        # ----------------------------------------------------
        # Registered model name
        # ----------------------------------------------------

        model_name = model_info[
            "model_name"
        ]

        # ----------------------------------------------------
        # Register model
        # ----------------------------------------------------

        model_version = register_model(
            model_name=model_name,
            model_info=model_info
        )

        # ----------------------------------------------------
        # Success
        # ----------------------------------------------------

        print(
            "\n✓ Model registration completed!"
        )

        print(
            f"✓ Registered Model: "
            f"{model_name}"
        )

        print(
            f"✓ Version: "
            f"{model_version.version}"
        )

    except Exception as e:

        logging.error(
            "Failed to complete model registration: %s",
            e
        )

        print(
            f"\n✗ Error: {e}"
        )

        # Important:
        # DVC must know the stage failed.
        raise


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()