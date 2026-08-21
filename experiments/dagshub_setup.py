import mlflow
import dagshub
from mlflow.tracking import MlflowClient

username = "iamprashantjain"
token = "7bed6b5be2021b1a4eaae221787bcb048ab2bcfd"
repo = "ml-pipeline-demo-v1"

mlflow.set_tracking_uri(f"https://{username}:{token}@dagshub.com/{username}/{repo}.mlflow")
dagshub.init(repo_owner=username, repo_name=repo, mlflow=True)

client = MlflowClient()

# Experiment name
exp_name = "default"

# Check if experiment exists
experiment = client.get_experiment_by_name(exp_name)

if experiment is None:
    # Create new experiment
    exp_id = client.create_experiment(exp_name)
    print(f"New experiment created: {exp_name} (ID: {exp_id})")
else:
    exp_id = experiment.experiment_id
    print(f"Using existing experiment: {exp_name} (ID: {exp_id})")

# AB RUN KARO WITH EXPERIMENT ID
with mlflow.start_run(experiment_id=exp_id, run_name="test_run"):
    mlflow.log_param("param1", "value1")
    mlflow.log_metric("metric1", 0.5)