import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error

# 1. Retrieve token
p = subprocess.Popen(
    ["git", "credential", "fill"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
)
stdout, _ = p.communicate(input="url=https://github.com/amrkh2004/arabic-sentiment-analysis.git\n\n")

token = ""
for line in stdout.splitlines():
    if line.startswith("password="):
        token = line.split("=", 1)[1].strip()
        break

if not token:
    raise RuntimeError("No token found.")

print(f"Authenticated with GitHub token: {token[:6]}...")

headers = {
    "Authorization": f"Bearer {token}",
    "Accept": "application/vnd.github+json",
    "User-Agent": "ArabicSentiment-MLOps",
    "Content-Type": "application/json",
}

repo = "amrkh2004/arabic-sentiment-analysis"
api_base = f"https://api.github.com/repos/{repo}"

def run_cmd(cmd):
    print(f">> Running: {cmd}")
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"STDERR: {res.stderr}")
        print(f"STDOUT: {res.stdout}")
        raise RuntimeError(f"Command failed: {cmd}")
    return res.stdout.strip()

# 2. Push initial commit to main on portfolio remote
print("\n--- Initializing main branch on portfolio remote ---")
run_cmd("git checkout -B main 31cabdb")
run_cmd("git push -u portfolio main --force")
print("Initial commit pushed to portfolio/main.")

# Module Definitions
modules = [
    {
        "id": 1,
        "branch": "feat/module-1-packaging-core-api",
        "title": "feat(module-1): Packaging, Core Model & FastAPI",
        "body": "### 📦 Module 1: Packaging, Core Model & FastAPI\n- **Packaging:** Configured `pyproject.toml` and structured codebase under `src/`.\n- **Inference & Preprocessing:** OOP-based Arabic text cleaning and sentiment model.\n- **REST API:** FastAPI `/predict` and `/health` with Pydantic validation.\n- **Testing:** Pytest suite with **>= 70% coverage** and Ruff formatting.",
        "checkout_paths": [
            "pyproject.toml", ".gitignore", "configs/", "scripts/prepare_dataset.py",
            "src/arabic_sentiment/", "tests/test_api.py", "tests/test_config.py",
            "tests/test_preprocessor.py", "tests/test_coverage_boost.py",
            ".github/workflows/ci.yml", "README.md"
        ],
        "remove_paths": ["prepare_data.py"],
    },
    {
        "id": 2,
        "branch": "feat/module-2-distillation-mlflow-dvc",
        "title": "feat(module-2): Knowledge Distillation & MLflow Registry",
        "body": "### 🔬 Module 2: Knowledge Distillation & MLflow\n- **Knowledge Distillation:** Student model compressed ~4x from MARBERT teacher.\n- **MLflow Registry:** Tracked experiments and registered `ArabicSentiment-Production` Champion.\n- **DVC Pipeline:** Versioned datasets with MinIO S3 remote storage (`dvcstore`).",
        "checkout_paths": [
            "notebooks/", "scripts/evaluate_baseline.py", "scripts/log_mlflow_experiments.py",
            "dvc.yaml", "dvc.lock", ".dvc/", ".dvcignore", "data/raw/reviews.csv.dvc",
            "data/processed/stats.json", "models/.gitkeep"
        ],
        "remove_paths": ["Arabic_Sentiment_KD_ONNX_TensorRT.ipynb"],
    },
    {
        "id": 3,
        "branch": "feat/module-3-onnx-docker-ghcr",
        "title": "feat(module-3): ONNX INT8 Quantization & Docker GHCR",
        "body": "### ⚡ Module 3: ONNX INT8 & Docker CI/CD\n- **ONNX Quantization:** Exported to INT8 format with ~15ms inference latency.\n- **Parity Tests:** Validated strict numerical parity between PyTorch and ONNX.\n- **Docker & CI/CD:** Hardened multi-stage Dockerfile and GitHub Actions GHCR publishing.",
        "checkout_paths": [
            "tests/test_onnx_parity.py", "tests/test_benchmark_latency.py",
            "scripts/benchmark_latency.py", "metrics.json", "Dockerfile",
            ".dockerignore", ".pre-commit-config.yaml"
        ],
        "remove_paths": [],
    },
    {
        "id": 4,
        "branch": "feat/module-4-serving-canary-api",
        "title": "feat(module-4): BentoML Serving & Canary Deployment",
        "body": "### 🚢 Module 4: Production Serving & Canary\n- **BentoML:** Packaged production service runner with `bentofile.yaml`.\n- **Structured Logging:** Correlation ID middleware for end-to-end request tracing.\n- **Canary & Load Testing:** Nginx 95/5 traffic routing and Locust load tests (p95 < 100ms SLA).",
        "checkout_paths": [
            "bentofile.yaml", "src/arabic_sentiment/serving/", "tests/test_logging_middleware.py",
            "deploy/", "locustfile.py", "scripts/run_load_test.py", "docker-compose.yml"
        ],
        "remove_paths": [],
    },
    {
        "id": 5,
        "branch": "feat/module-5-monitoring-airflow-terraform",
        "title": "feat(module-5): Observability, Airflow, Terraform & Reports",
        "body": "### 📊 Module 5: Observability, Airflow & Terraform\n- **Monitoring:** Prometheus metrics and custom Grafana live dashboard.\n- **Automation:** Evidently data drift detection and Apache Airflow retraining DAG.\n- **Infra & Docs:** Terraform IaC configs, batch scoring pipeline, and complete visual README.",
        "checkout_paths": [
            "monitoring/", "tests/test_monitoring.py", "scripts/monitor_drift.py",
            "artifacts/", "dags/", "terraform/", "tests/test_terraform.py",
            "scripts/batch_scoring.py", "reports/", "docs/screenshots/", "logs/.gitkeep",
            "README.md"
        ],
        "remove_paths": [],
    },
]

for m in modules:
    print(f"\n=======================================================")
    print(f"  PROCESSING MODULE {m['id']}: {m['title']}")
    print(f"=======================================================")

    # 1. Checkout new branch from current main
    run_cmd(f"git checkout -B {m['branch']} main")

    # 2. Checkout paths from backup-perfect-state
    for p_path in m["checkout_paths"]:
        run_cmd(f"git checkout backup-perfect-state -- {p_path}")

    for r_path in m["remove_paths"]:
        if os.path.exists(r_path):
            run_cmd(f"git rm -f {r_path}")

    # Remove any unwanted runtime artifacts
    for ignore_item in [".coverage", "mlflow.db"]:
        run_cmd(f"git rm --cached -f {ignore_item} 2>NUL || rem")

    run_cmd("git add -A")
    run_cmd(f'git commit -m "{m["title"]}"')

    # 3. Push branch to portfolio
    run_cmd(f"git push -u portfolio {m['branch']} --force")
    print(f"Branch {m['branch']} pushed to portfolio.")

    # 4. Open PR
    pr_data = json.dumps({
        "title": m["title"],
        "head": m["branch"],
        "base": "main",
        "body": m["body"],
    }).encode("utf-8")

    req_pr = urllib.request.Request(f"{api_base}/pulls", data=pr_data, headers=headers, method="POST")
    with urllib.request.urlopen(req_pr) as resp:
        pr_res = json.loads(resp.read().decode("utf-8"))
        pr_num = pr_res["number"]
        pr_url = pr_res["html_url"]
        pr_sha = pr_res["head"]["sha"]
        print(f"Opened PR #{pr_num}: {pr_url} (SHA: {pr_sha[:7]})")

    # 5. Wait for CI Check Runs to complete with success
    print(f"Waiting for GitHub Actions checks to pass on PR #{pr_num}...")
    start_time = time.time()
    all_passed = False

    while time.time() - start_time < 600:  # 10 min max timeout
        time.sleep(15)
        req_checks = urllib.request.Request(f"{api_base}/commits/{pr_sha}/check-runs", headers=headers)
        with urllib.request.urlopen(req_checks) as resp:
            checks_data = json.loads(resp.read().decode("utf-8"))
            check_runs = checks_data.get("check_runs", [])

        if not check_runs:
            print("  Check-runs not triggered yet, waiting...")
            continue

        statuses = [c["status"] for c in check_runs]
        conclusions = [c.get("conclusion") for c in check_runs]
        names = [c["name"] for c in check_runs]

        print(f"  Checks: {list(zip(names, statuses, conclusions))}")

        if all(s == "completed" for s in statuses):
            failed = [n for n, c in zip(names, conclusions) if c not in ["success", "neutral", "skipped"]]
            if not failed:
                print(f"ALL CHECKS PASSED (GREEN ✔) on PR #{pr_num}!")
                all_passed = True
                break
            else:
                print(f"ERROR: Check failed: {failed}")
                raise RuntimeError(f"Check failed on PR #{pr_num}: {failed}")

    if not all_passed:
        raise TimeoutError(f"Checks timed out on PR #{pr_num}")

    time.sleep(5)

    # 6. Merge PR
    print(f"Merging PR #{pr_num} into main...")
    merge_data = json.dumps({
        "commit_title": f"Merge pull request #{pr_num} from amrkh2004/{m['branch']}",
        "merge_method": "merge",
    }).encode("utf-8")

    req_merge = urllib.request.Request(f"{api_base}/pulls/{pr_num}/merge", data=merge_data, headers=headers, method="PUT")
    with urllib.request.urlopen(req_merge) as resp:
        m_res = json.loads(resp.read().decode("utf-8"))
        print(f"Successfully Merged PR #{pr_num} (merged: {m_res.get('merged')})")

    # 7. Pull merged main locally
    run_cmd("git checkout main")
    run_cmd("git pull portfolio main")
    print(f"Local main updated with PR #{pr_num} merge.")
    time.sleep(3)

print("\n=======================================================")
print("ALL 5 MODULE PRs SUCCESSFULLY CREATED, VERIFIED 100% GREEN, AND MERGED!")
print("=======================================================")
