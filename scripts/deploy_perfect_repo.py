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
    "User-Agent": "ArabicSentiment-Deployer",
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

def gh_get(url):
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def gh_post(url, data):
    req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def gh_put(url, data):
    req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers, method="PUT")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

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

    # Check if a PR already exists and is already merged
    closed_prs = gh_get(f"{api_base}/pulls?head=amrkh2004:{m['branch']}&state=closed")
    if closed_prs and closed_prs[0].get("merged_at"):
        print(f"Module {m['id']} PR #{closed_prs[0]['number']} is ALREADY MERGED. Skipping.")
        run_cmd("git checkout main")
        run_cmd("git pull portfolio main")
        continue

    # 1. Checkout branch from current main
    branches = run_cmd("git branch --list " + m["branch"])
    if m["branch"] in branches:
        run_cmd(f"git checkout {m['branch']}")
    else:
        run_cmd(f"git checkout -B {m['branch']} main")

    # 2. Sync files from backup-perfect-state
    for p_path in m["checkout_paths"]:
        run_cmd(f"git checkout backup-perfect-state -- {p_path}")

    for r_path in m["remove_paths"]:
        if os.path.exists(r_path):
            run_cmd(f"git rm -f {r_path}")

    # Remove runtime artifacts if any
    for ignore_item in [".coverage", "mlflow.db", "coverage.xml"]:
        try:
            run_cmd(f"git rm --cached -f {ignore_item}")
        except Exception:
            pass

    # Commit if changes exist
    status = run_cmd("git status --porcelain")
    if status.strip():
        run_cmd("git add -A")
        run_cmd(f'git commit -m "{m["title"]}"')
    else:
        print("No new file changes to commit on this branch.")

    # 3. Push branch to portfolio remote
    run_cmd(f"git push -u portfolio {m['branch']} --force")
    print(f"Branch {m['branch']} pushed to portfolio.")

    # 4. Check or Create PR
    open_prs = gh_get(f"{api_base}/pulls?head=amrkh2004:{m['branch']}&state=open")
    if open_prs:
        pr_num = open_prs[0]["number"]
        pr_url = open_prs[0]["html_url"]
        pr_sha = open_prs[0]["head"]["sha"]
        print(f"Using open PR #{pr_num}: {pr_url} (SHA: {pr_sha[:7]})")
    else:
        pr_res = gh_post(f"{api_base}/pulls", {
            "title": m["title"],
            "head": m["branch"],
            "base": "main",
            "body": m["body"],
        })
        pr_num = pr_res["number"]
        pr_url = pr_res["html_url"]
        pr_sha = pr_res["head"]["sha"]
        print(f"Opened new PR #{pr_num}: {pr_url} (SHA: {pr_sha[:7]})")

    # Give GitHub a moment to register push & trigger workflow
    time.sleep(10)
    pr_info = gh_get(f"{api_base}/pulls/{pr_num}")
    pr_sha = pr_info["head"]["sha"]
    print(f"Monitoring CI for commit SHA: {pr_sha[:7]}...")

    # 5. Wait for CI checks to complete with success
    start_time = time.time()
    all_passed = False

    while time.time() - start_time < 600:
        time.sleep(12)
        runs_data = gh_get(f"{api_base}/actions/runs?head_sha={pr_sha}")
        wf_runs = runs_data.get("workflow_runs", [])

        if not wf_runs:
            cr_data = gh_get(f"{api_base}/commits/{pr_sha}/check-runs")
            check_runs = cr_data.get("check_runs", [])
            if not check_runs:
                print("  Waiting for GitHub Actions to trigger...")
                continue
            statuses = [c["status"] for c in check_runs]
            conclusions = [c.get("conclusion") for c in check_runs]
            names = [c["name"] for c in check_runs]
            print(f"  Check-runs: {list(zip(names, statuses, conclusions))}")
            if all(s == "completed" for s in statuses):
                failed = [n for n, c in zip(names, conclusions) if c not in ["success", "neutral", "skipped"]]
                if not failed:
                    print(f"  ALL CHECK-RUNS PASSED (GREEN ✔) on PR #{pr_num}!")
                    all_passed = True
                    break
                else:
                    raise RuntimeError(f"Check-runs failed: {failed}")
            continue

        all_wf_completed = True
        has_failure = False

        for wf in wf_runs:
            w_status = wf.get("status")
            w_conc = wf.get("conclusion")
            run_id = wf.get("id")

            jobs_info = gh_get(f"{api_base}/actions/runs/{run_id}/jobs")
            job_summaries = [f"{j['name']} ({j['status']}/{j.get('conclusion')})" for j in jobs_info.get("jobs", [])]
            print(f"  Run #{run_id} ({wf.get('name')}): status={w_status}, conclusion={w_conc} | Jobs: {', '.join(job_summaries)}")

            if w_status != "completed":
                all_wf_completed = False
            else:
                if w_conc not in ["success", "neutral", "skipped"]:
                    has_failure = True
                    for j in jobs_info.get("jobs", []):
                        if j.get("conclusion") not in ["success", "neutral", "skipped"]:
                            print(f"    FAILED JOB: {j['name']}")
                            for s in j.get("steps", []):
                                if s.get("conclusion") not in ["success", "neutral", "skipped"]:
                                    print(f"      FAILED STEP: {s['name']} -> {s.get('conclusion')}")

        if has_failure:
            raise RuntimeError(f"CI failed on PR #{pr_num} for commit {pr_sha[:7]}")

        if all_wf_completed and len(wf_runs) > 0:
            print(f"\n🎉 ALL GITHUB ACTIONS PASSED (100% GREEN ✔) on PR #{pr_num}!")
            all_passed = True
            break

    if not all_passed:
        raise TimeoutError(f"CI timed out waiting for PR #{pr_num}")

    time.sleep(5)

    # 6. Merge PR
    print(f"Merging PR #{pr_num} into main...")
    merge_res = gh_put(f"{api_base}/pulls/{pr_num}/merge", {
        "commit_title": f"Merge pull request #{pr_num} from amrkh2004/{m['branch']}",
        "merge_method": "merge",
    })
    print(f"PR #{pr_num} merged: {merge_res.get('merged')}")

    # 7. Pull merged main
    time.sleep(3)
    run_cmd("git checkout main")
    run_cmd("git pull portfolio main")
    print(f"Local main updated with PR #{pr_num}.")
    time.sleep(5)

print("\n=======================================================")
print("ALL 5 MODULE PRs SUCCESSFULLY MERGED WITH 100% GREEN ✔ CHECKS!")
print("=======================================================")
