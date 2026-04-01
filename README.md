# MLOps Credit Default Prediction — Databricks

<p align="center">
<img width="737" alt="cover" src="https://github.com/user-attachments/assets/a1c18fba-9e39-45b5-8fcd-bceb1f5f5af9">
</p>

End-to-end MLOps project for credit card default prediction, built on Databricks with MLflow, Feature Store, LightGBM, and automated CI/CD. Based on the [Kaggle UCI Credit Default Dataset](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset/data).

Developed as part of the [End-to-end MLOps with Databricks](https://maven.com/marvelousmlops/mlops-with-databricks) course. Walk through the implementation in the companion [Medium publication](https://medium.com/@benitomartin/8cd9a85cc3c0).

---

## Tech Stack

![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Databricks](https://img.shields.io/badge/Databricks-FF3621?style=for-the-badge&logo=Databricks&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Spark](https://img.shields.io/badge/Apache_Spark-FFFFFF?style=for-the-badge&logo=apachespark&logoColor=#E35A16)
![MLflow](https://img.shields.io/badge/MLflow-0194E2.svg?style=for-the-badge&logo=MLflow&logoColor=white)
![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture & Data Flow](#2-architecture--data-flow)
3. [Project Structure](#3-project-structure)
4. [Local Environment Setup](#4-local-environment-setup)
5. [Databricks Setup](#5-databricks-setup)
6. [Catalog & Volume Setup](#6-catalog--volume-setup)
7. [Token & Secrets Setup](#7-token--secrets-setup)
8. [Step 1 — Data Loading & Initial Preparation](#8-step-1--data-loading--initial-preparation)
9. [Step 2 — Feature Engineering & Initial Model](#9-step-2--feature-engineering--initial-model)
   - [Option A: Standard Table (no online sync)](#option-a-standard-table-no-online-sync)
   - [Option B: Sync Table (online feature table)](#option-b-sync-table-online-feature-table)
10. [Step 3 — Model Serving](#10-step-3--model-serving)
11. [Step 4 — Deploy the Automated Pipeline](#11-step-4--deploy-the-automated-pipeline)
12. [Step 5 — Monitoring & Alerts](#12-step-5--monitoring--alerts)
13. [CI/CD with GitHub Actions](#13-cicd-with-github-actions)
14. [Running Tests Locally](#14-running-tests-locally)

---

## 1. Project Overview

This project implements a production-grade machine learning pipeline that predicts whether a credit card client will default on their next payment. The pipeline covers:

- **Data ingestion** from Unity Catalog volumes with incremental refresh
- **Data cleaning & preprocessing** using Spark and scikit-learn
- **Feature engineering** stored in Databricks Feature Store
- **Model training** with LightGBM and SMOTE class balancing
- **Experiment tracking** with MLflow
- **Automated model evaluation** — new model only deploys if it beats production AUC
- **Real-time serving** via Databricks Model Serving endpoints
- **Lakehouse monitoring** for data drift and accuracy degradation
- **Automated retraining** on a weekly schedule via Databricks Asset Bundles

---

## 2. Architecture & Data Flow

```
Raw CSV (data.csv)
    │
    ▼
[Data Cleaning]  ←─ data_cleaning_spark.py
    │  rename columns, fix invalid values (Education, Marriage, Pay_X)
    ▼
[Data Preprocessing]  ←─ data_preprocessing_spark.py
    │  RobustScaler on 13 numeric features, 80/20 train-test split
    ├──► train_set (Delta table, CDF enabled)
    └──► test_set  (Delta table, CDF enabled)
              │
              ▼
    [Feature Engineering]
    │  SMOTE balancing → features_balanced (Feature Store table)
    │
    ├─── Option A: Standard table only (offline)
    └─── Option B: + Online Table sync via DLT pipeline
              │
              ▼
    [Model Training]  ←─ LightGBM + FeatureLookup
    │  MLflow experiment tracking, AUC metric
              │
              ▼
    [Model Evaluation]
    │  Compare new AUC vs production AUC
    │  If better → register new version
              │
              ▼
    [Model Deployment]
    │  Update serving endpoint (scale-to-zero)
              │
              ▼
    [Monitoring]
       Payload logs → model_monitoring table → quality monitor → alerts
```

---

## 3. Project Structure

```
mlops-databricks-credit-default/
├── .github/workflows/
│   ├── ci.yml                          # Runs tests and linting on PRs
│   └── cd.yml                          # Deploys bundle to Databricks on merge
├── notebooks/
│   ├── create_source_data/
│   │   └── create_source_data_notebook.py   # Generate synthetic inference data
│   ├── feature_engineering/
│   │   ├── prepare_data_notebook.py         # Load & prepare initial train/test data
│   │   ├── basic_mlflow_experiment_notebook.py      # Baseline model (no Feature Store)
│   │   ├── feature_mlflow_experiment_notebook.py    # Model with Feature Store (SMOTE)
│   │   ├── combined_mlflow_experiment_notebook.py   # PyFunc wrapper model
│   │   └── custom_mlflow_experiment_notebook.py     # Custom PyFunc from existing model
│   ├── model_feature_serving/
│   │   ├── model_serving_notebook.py               # Basic REST endpoint
│   │   ├── feature_serving_notebook.py             # Feature-as-a-service endpoint
│   │   ├── model_serving_feat_lookup_notebook.py   # Serving with feature lookup
│   │   └── AB_test_model_serving_notebbok.py       # A/B test endpoint
│   └── monitoring/
│       ├── lakehouse_monitoring.py      # Set up quality monitor
│       ├── create_inference_data.py     # Generate drift/normal inference data
│       ├── create_alert.py              # Accuracy degradation alert
│       └── send_request_to_endpoint.py # Traffic simulation
├── src/credit_default/
│   ├── data_cleaning.py                # Pandas-based cleaning (local dev)
│   ├── data_cleaning_spark.py          # Spark-based cleaning (Databricks)
│   ├── data_preprocessing.py           # Pandas preprocessing (local dev)
│   ├── data_preprocessing_spark.py     # Spark preprocessing + catalog save
│   └── utils.py                        # Config loading, logging utilities
├── tests/
│   ├── test_data_cleaning.py
│   └── test_data_preprocessor.py
├── workflows/
│   ├── preprocess.py                   # Ingestion + feature table update
│   ├── train_model.py                  # LightGBM training with Feature Store
│   ├── evaluate_model.py               # Compare new vs production model
│   ├── deploy_model.py                 # Update serving endpoint
│   └── refresh_monitor.py             # Process payload logs, refresh monitor
├── project_config.yml                  # Model params, features, catalog settings
├── databricks.yml                      # Asset bundle: jobs, clusters, schedules
├── bundle_monitoring.yml               # Monitoring job bundle config
├── pyproject.toml                      # Dependencies and build config
└── .env.sample                         # Template for local environment variables
```

---

## 4. Local Environment Setup

**Requirements**: Python 3.11, [uv](https://github.com/astral-sh/uv)

```bash
# Clone the repository
git clone https://github.com/benitomartin/mlops-databricks-credit-default.git
cd mlops-databricks-credit-default

# Create virtual environment with Python 3.11
uv venv -p 3.11.0 .venv
source .venv/bin/activate       # Linux/Mac
# .venv\Scripts\activate        # Windows

# Install all dependencies including dev extras
uv pip install -r pyproject.toml --all-extras
uv lock

# Build the wheel package (needed for cluster installation)
uv build
```

Copy and configure the environment file:

```bash
cp .env.sample .env
```

Edit `.env` with your values:

```env
FILEPATH=data/data.csv
FILEPATH_DATABRICKS=dbfs:/Volumes/credit/default/data/data.csv
PROFILE=DEFAULT
CLUSTER_ID=<your-cluster-id>
CLEANING_LOGS=logs/data_cleaning.log
PREPROCESSING_LOGS=logs/data_preprocessing.log
PIPELINE_LOGS=logs/pipeline.log
TRAINING_LOGS=logs/training.log
```

---

## 5. Databricks Setup

### Install Databricks CLI

```bash
curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh
```

### Authenticate

```bash
# Authenticate and configure a default cluster
databricks auth login --configure-cluster --host <your-workspace-url>

# Verify authentication
databricks auth profiles
cat ~/.databrickscfg
```

The CLI saves your credentials to `~/.databrickscfg` under a named profile (default: `DEFAULT`). Update the `PROFILE` value in `.env` to match.

### Install VS Code Extension

Install the **Databricks** extension from the VS Code marketplace, then connect it to your workspace using the same profile.

---

## 6. Catalog & Volume Setup

The project uses **Unity Catalog** with the following hierarchy:

| Level | Name |
|-------|------|
| Catalog | `credit` |
| Schema | `default` |
| Volumes | `data`, `packages` |

```bash
# Create the volumes
databricks volumes create credit default data MANAGED
databricks volumes create credit default packages MANAGED

# Upload the raw dataset
databricks fs cp data/data.csv dbfs:/Volumes/credit/default/data/data.csv

# Upload the wheel package (update version as needed)
databricks fs cp dist/credit_default_databricks-0.0.14-py3-none-any.whl \
    dbfs:/Volumes/credit/default/packages/

# Verify uploads
databricks fs ls dbfs:/Volumes/credit/default/data
databricks fs ls dbfs:/Volumes/credit/default/packages
```

> **Note**: After rebuilding (`uv build`), re-upload the wheel with the new version number.

---

## 7. Token & Secrets Setup

Several notebooks and the monitoring workflow authenticate via a personal access token stored as a Databricks secret.

### Create a Personal Access Token

1. In the Databricks UI: `Settings` → `User` → `Developer` → `Access tokens`
2. Click **Generate new token** and copy the value

### Store the Token as a Secret

```bash
# Create a secret scope
databricks secrets create-scope secret-scope

# Store the token (you'll be prompted to paste it)
databricks secrets put-secret secret-scope databricks-token

# Verify
databricks secrets list-secrets secret-scope
```

### GitHub Actions Secret

For CI/CD to deploy the bundle, add the token as a GitHub repository secret:

- Go to your repo → `Settings` → `Secrets and variables` → `Actions`
- Add a secret named `DATABRICKS_TOKEN` with the token value

---

## 8. Step 1 — Data Loading & Initial Preparation

This step covers the data prep checklist: loading raw data, cleaning it, and creating the initial train/test split in the Databricks catalog.

### What happens

Following data preparation best practices:
- Load raw CSV from Unity Catalog volumes
- Clean column names (rename and capitalize)
- Fix invalid category values:
  - `Education`: values 0, 5, 6 → mapped to 4 (Other)
  - `Marriage`: value 0 → mapped to 3 (Other)
  - `Pay_X`: values -1, -2 → mapped to 0 (paid on time)
- Apply `RobustScaler` to 13 numeric features (Limit_bal, Bill_amt1-6, Pay_amt1-6)
- Split 80% train / 20% test
- Save to Delta tables with Change Data Feed (CDF) enabled

### Run the notebook

Open and run in Databricks:

```
notebooks/feature_engineering/prepare_data_notebook.py
```

This creates two Delta tables in `credit.default`:

| Table | Description |
|-------|-------------|
| `credit.default.train_set` | 80% of cleaned data with `Update_timestamp_utc` |
| `credit.default.test_set` | 20% of cleaned data with `Update_timestamp_utc` |

### Key configuration (`project_config.yml`)

```yaml
catalog_name: credit
schema_name: default

model_parameters:
  learning_rate: 0.05
  random_state: 42
  force_col_wise: true

features:
  clean:
    - Id, Limit_bal, Sex, Education, Marriage, Age
    - Pay_0, Pay_2, Pay_3, Pay_4, Pay_5, Pay_6
    - Bill_amt1 ... Bill_amt6
    - Pay_amt1 ... Pay_amt6
  robust:
    - Limit_bal
    - Bill_amt1 ... Bill_amt6
    - Pay_amt1 ... Pay_amt6

target:
  name: "default.payment.next.month"
  new_name: "Default"
```

---

## 9. Step 2 — Feature Engineering & Initial Model

This step creates the `features_balanced` feature table (using SMOTE oversampling) and trains an initial LightGBM model tracked with MLflow. Choose one of two options depending on whether you need real-time feature lookup (online sync) or batch-only predictions.

---

### Option A: Standard Table (no online sync)

Use this option for **batch predictions** or when you do not need a low-latency online feature table.

**What happens:**
- Creates `credit.default.features_balanced` as a standard Delta table
- Uses offline FeatureLookup during training
- Model is registered as `credit.default.credit_model_feature`
- No DLT pipeline is created; no `pipeline_id` needed in config

**Run the notebook:**

```
notebooks/feature_engineering/feature_mlflow_experiment_notebook.py
```

**In `project_config.yml`**, leave or remove `pipeline_id` (it is only used for online sync):

```yaml
# pipeline_id: not required for Option A
```

**MLflow output:**
- Experiment: `/Shared/credit_default`
- Registered model: `credit.default.credit_model_feature`
- Metrics logged: AUC on test set

---

### Option B: Sync Table (online feature table)

Use this option for **real-time / low-latency serving**. An online table is kept in sync with the offline feature table via a Delta Live Tables (DLT) pipeline. The serving endpoint performs feature lookups directly from the online table.

**What happens:**
- Creates `credit.default.features_balanced` with Change Data Feed enabled
- Creates an **online table** `credit.default.features_balanced_online` backed by a DLT pipeline
- The pipeline ID is stored in `project_config.yml` and used by `preprocess.py` to trigger refreshes after each new data batch
- Uses online FeatureLookup during inference for single-record, low-latency predictions

**Step-by-step:**

**1. Run the feature engineering notebook** to create the offline table and the online table:

```
notebooks/feature_engineering/feature_mlflow_experiment_notebook.py
```

**2. Get the pipeline ID** of the created online table sync pipeline:

In the Databricks UI, go to `Delta Live Tables` → find the pipeline linked to `features_balanced_online` → copy the pipeline ID.

**3. Update `project_config.yml`** with the pipeline ID:

```yaml
pipeline_id: <your-dlt-pipeline-id>
```

**4. Run the serving notebook** to create the feature lookup endpoint:

```
notebooks/model_feature_serving/model_serving_feat_lookup_notebook.py
```

This creates:

| Resource | Name |
|----------|------|
| Online table | `credit.default.features_balanced_online` |
| Feature spec | `credit.default.features_balanced_spec` |
| Serving endpoint | `credit-default-model-serving-feature` |

**5. Test a single prediction:**

```python
import requests, json

token = "<your-databricks-token>"
endpoint_url = "https://<workspace>.azuredatabricks.net/serving-endpoints/credit-default-model-serving-feature/invocations"

payload = {"dataframe_records": [{"Id": 12345}]}  # lookup by primary key

response = requests.post(
    endpoint_url,
    headers={"Authorization": f"Bearer {token}"},
    json=payload
)
print(response.json())
```

---

### Comparing the two options

| | Option A: Standard Table | Option B: Sync Table |
|---|---|---|
| Feature table type | Offline Delta table | Offline + Online (DLT-synced) |
| Inference latency | Batch (seconds–minutes) | Real-time (milliseconds) |
| `pipeline_id` required | No | Yes |
| DLT pipeline created | No | Yes |
| Serving strategy | Batch scoring | Online FeatureLookup |
| Best for | Scheduled batch jobs | REST API real-time predictions |

---

## 10. Step 3 — Model Serving

After training the initial model, set up the serving endpoint before deploying the automated pipeline.

### Basic endpoint (Option A)

```
notebooks/model_feature_serving/model_serving_notebook.py
```

Creates endpoint `credit-default-model-serving` using a standard model version.

### Feature lookup endpoint (Option B)

```
notebooks/model_feature_serving/model_serving_feat_lookup_notebook.py
```

Creates endpoint `credit-default-model-serving-feature` with online feature lookup.

### A/B test endpoint (optional)

```
notebooks/model_feature_serving/AB_test_model_serving_notebbok.py
```

Trains two models (learning_rate A=0.05, B=0.1), wraps them in a routing `CreditDefaultModelWrapper` that routes requests by hashed `Id`, and deploys a single endpoint serving both models.

---

## 11. Step 4 — Deploy the Automated Pipeline

The automated pipeline is packaged as a **Databricks Asset Bundle** and deployed via the Databricks CLI. It runs weekly on Mondays at 06:00 Amsterdam time.

### Pipeline job tasks (in order)

```
preprocess  →  [if refreshed]  →  train_model  →  evaluate_model  →  [if better]  →  deploy_model
```

| Task | File | Condition |
|------|------|-----------|
| `preprocess` | `workflows/preprocess.py` | Always runs |
| `train_model` | `workflows/train_model.py` | Only if `preprocess.refreshed == 1` |
| `evaluate_model` | `workflows/evaluate_model.py` | Only after training |
| `deploy_model` | `workflows/deploy_model.py` | Only if `evaluate_model.model_update == 1` |

### Deploy to development

```bash
# Validate the bundle configuration
databricks bundle validate

# Deploy to dev target
databricks bundle deploy --target dev

# Trigger a manual run
databricks bundle run credit-default --target dev
```

### Deploy to production

```bash
databricks bundle deploy --target prod
```

### Monitor a running job

```bash
# List running jobs
databricks jobs list

# Get run status
databricks runs get --run-id <run-id>
```

### Bundle variables

Defined in `databricks.yml`:

| Variable | Default | Description |
|----------|---------|-------------|
| `root_path` | `/Shared/...` | Workspace path for bundle artifacts |
| `git_sha` | — | Git commit SHA (set automatically by CI/CD) |
| `schedule_pause_status` | `UNPAUSED` | Set to `PAUSED` to disable the schedule |

### Cluster configuration

| Setting | Value |
|---------|-------|
| Spark version | 15.4.x-scala2.12 |
| Node type | Standard_DS3_v2 |
| Mode | SINGLE_USER |
| Workers | 0–2 (autoscaling) |
| Libraries | Wheel from `/Volumes/credit/default/packages/` |

---

## 12. Step 5 — Monitoring & Alerts

After the pipeline is deployed and receiving traffic, set up monitoring to track model performance and data drift.

### Create inference traffic

```
notebooks/monitoring/create_inference_data.py
```

Generates two types of inference records appended to the feature table:
- **Normal**: representative of training distribution
- **Skewed**: 1.5x multiplier on key features (Pay_0, Age, Bill_amt1, Limit_bal) to simulate drift

```
notebooks/monitoring/send_request_to_endpoint.py
```

Sends traffic to the serving endpoint for 35 minutes:
- 10 min normal → 10 min normal → 15 min skewed

### Set up Lakehouse monitoring

```
notebooks/monitoring/lakehouse_monitoring.py
```

Creates a Databricks quality monitor on `credit.default.model_monitoring`:

| Setting | Value |
|---------|-------|
| Problem type | Classification |
| Prediction column | `prediction` |
| Label column | `Default` |
| Granularity | 30 minutes |
| CDF | Enabled |

### Create accuracy alert

```
notebooks/monitoring/create_alert.py
```

Creates a SQL alert that triggers when **accuracy drops below 42%**:
- Query target: `credit.default.model_monitoring_profile_metrics`
- Alert condition: `accuracy < 0.42`

### Monitor refresh job

The monitoring bundle (`bundle_monitoring.yml`) runs a separate job `credit-default-monitor-update-workflow` on the same Monday 06:00 schedule as the main pipeline.

```bash
# Deploy the monitoring bundle
databricks bundle deploy -t dev --config-file bundle_monitoring.yml

# Run manually
databricks bundle run credit-default-monitor-update-workflow \
    --config-file bundle_monitoring.yml
```

**What `refresh_monitor.py` does:**
1. Loads payload logs from the serving endpoint (`model-serving-feature_payload`)
2. Parses request/response JSON
3. Joins with actual labels from train/test sets
4. Appends enriched records to `credit.default.model_monitoring`
5. Triggers the quality monitor refresh

---

## 13. CI/CD with GitHub Actions

### CI Pipeline (`.github/workflows/ci.yml`)

Triggers on **pull requests** to `main`:

```
Lint (ruff) → Unit tests (pytest) → Build wheel
```

### CD Pipeline (`.github/workflows/cd.yml`)

Triggers on **push to `main`**:

```
Build wheel → Upload to DBFS volume → Bundle deploy (prod)
```

Required GitHub secrets:

| Secret | Description |
|--------|-------------|
| `DATABRICKS_HOST` | Workspace URL |
| `DATABRICKS_TOKEN` | Personal access token |

---

## 14. Running Tests Locally

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run a single test file
pytest tests/test_data_cleaning.py -v
```

Lint and format:

```bash
# Check linting
ruff check .

# Auto-fix linting issues
ruff check . --fix

# Format code
ruff format .
```

---

## References

- [Kaggle Dataset](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset/data)
- [Medium Publication](https://medium.com/@benitomartin/8cd9a85cc3c0)
- [Maven Course — End-to-end MLOps with Databricks](https://maven.com/marvelousmlops/mlops-with-databricks)
- [Data Prep Checklist for ML](https://medium.com/learning-data/data-prep-for-machine-learning-checklist-129b46b73782)
- [Databricks Asset Bundles Documentation](https://docs.databricks.com/en/dev-tools/bundles/index.html)
- [Databricks Feature Engineering](https://docs.databricks.com/en/machine-learning/feature-store/index.html)
