"""
Data and Concept Drift Monitoring Suite for Arabic Sentiment Analysis.
Computes statistical drift between baseline reference data and production traffic.
Calculates Population Stability Index (PSI) for Text Length and Confidence Scores,
along with Kolmogorov-Smirnov (KS) test and Concept Drift analysis.
Triggers automated alerting when PSI exceeds acceptable threshold (PSI > 0.25).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from arabic_sentiment.api.service import SentimentInferenceService


def calculate_categorical_psi(
    expected: np.ndarray, actual: np.ndarray, epsilon: float = 1e-4
) -> float:
    """Calculates Population Stability Index (PSI) between two categorical distributions."""
    expected = np.maximum(expected, epsilon)
    actual = np.maximum(actual, epsilon)
    expected = expected / np.sum(expected)
    actual = actual / np.sum(actual)
    return float(max(0.0, np.sum((actual - expected) * np.log(actual / expected))))


# Backward compatibility alias
calculate_psi = calculate_categorical_psi


def calculate_numerical_psi(
    expected: np.ndarray, actual: np.ndarray, num_bins: int = 10, epsilon: float = 1e-4
) -> float:
    """
    Calculates Population Stability Index (PSI) for continuous numerical features
    (e.g. Text Length and Confidence Scores).
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(expected, quantiles)
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0

    bin_edges[0] -= 1e-5
    bin_edges[-1] += 1e-5

    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    n_bins = len(expected_counts)
    expected_pct = (expected_counts + epsilon) / (len(expected) + epsilon * n_bins)
    actual_pct = (actual_counts + epsilon) / (len(actual) + epsilon * n_bins)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(max(0.0, psi))


def run_drift_analysis(
    reference_path: str = "data/processed/val.csv",
    prod_data_path: str | None = None,
    output_dir: str = "artifacts/monitoring",
    psi_alert_threshold: float = 0.25,
    ks_pvalue_alert_threshold: float = 0.05,
) -> dict:
    """
    Runs comprehensive drift analysis computing PSI on Text Length & Confidence Scores,
    and checks against SLA alert threshold (PSI > 0.25).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("      ARABIC SENTIMENT PIPELINE - DRIFT MONITORING SUITE")
    print("=" * 65)

    ref_file = Path(reference_path)
    service = SentimentInferenceService()

    if not ref_file.exists():
        print(f"Reference file not found at {ref_file}. Creating synthetic baseline...")
        ref_df = pd.DataFrame(
            {
                "clean_text": ["ممتاز جدا ورائع والتوصيل سريع"] * 200
                + ["رديء جدا وسيء للغاية ومكسور"] * 200
                + ["عادي ومقبول بالنسبة لسعره"] * 200,
                "label": ["positive"] * 200 + ["negative"] * 200 + ["neutral"] * 200,
            }
        )
    else:
        ref_df = pd.read_csv(ref_file)

    ref_texts = (
        ref_df["clean_text"].astype(str).tolist()
        if "clean_text" in ref_df.columns
        else ref_df.iloc[:, 0].astype(str).tolist()
    )
    ref_lengths = np.array([len(t) for t in ref_texts], dtype=float)

    # Compute reference predictions if confidence scores not present
    if "confidence" in ref_df.columns:
        ref_confs = ref_df["confidence"].to_numpy(dtype=float)
    else:
        ref_preds = service.predict(ref_texts[:300])  # Sample for speed if large
        ref_confs = np.array([p.confidence for p in ref_preds], dtype=float)

    labels = ["positive", "neutral", "negative"]
    if "label" in ref_df.columns:
        ref_counts = ref_df["label"].value_counts(normalize=True)
        ref_dist = np.array([ref_counts.get(l, 0.333) for l in labels])
    else:
        ref_dist = np.array([0.333, 0.333, 0.334])

    # 2. Load or simulate production batch
    if prod_data_path and Path(prod_data_path).exists():
        print(f"Loading production batch from {prod_data_path}...")
        prod_df = pd.read_csv(prod_data_path)
        text_col = "review" if "review" in prod_df.columns else prod_df.columns[0]
        prod_texts = prod_df[text_col].astype(str).tolist()

        if "text_length" in prod_df.columns:
            prod_lengths = prod_df["text_length"].to_numpy(dtype=float)
        else:
            prod_lengths = np.array([len(t) for t in prod_texts], dtype=float)

        if "confidence_score" in prod_df.columns:
            prod_confs = prod_df["confidence_score"].to_numpy(dtype=float)
            prod_labels = (
                prod_df["predicted_label"].tolist()
                if "predicted_label" in prod_df.columns
                else ["positive"] * len(prod_texts)
            )
        else:
            prod_preds = service.predict(prod_texts)
            prod_confs = np.array([p.confidence for p in prod_preds], dtype=float)
            prod_labels = [p.label for p in prod_preds]
    else:
        print("Simulating incoming production batch (recent customer reviews surge)...")
        simulated_prod_reviews = [
            "التوصيل اتأخر جدا والمنتج مكسور والعلبة مفتوحة حسبي الله",
            "تأخير أسبوع في الشحن وخامة رديئة جدا وغير مطابقة",
            "سيء للغاية ولا أنصح به بتاتا",
            "خدمة عملاء زبالة وما في رد",
            "المنتج عادي بس السعر غالي",
        ] * 60 + [
            "ممتاز ورائع وجودة جيدة جدا",
            "حلو وعجبني وسريع",
        ] * 25
        prod_texts = simulated_prod_reviews
        prod_lengths = np.array([len(t) for t in prod_texts], dtype=float)
        prod_preds = service.predict(prod_texts)
        prod_confs = np.array([p.confidence for p in prod_preds], dtype=float)
        prod_labels = [p.label for p in prod_preds]

    prod_counts = pd.Series(prod_labels).value_counts(normalize=True)
    prod_dist = np.array([prod_counts.get(l, 0.0) for l in labels])

    # 3. Calculate PSI for Text Length and Confidence Scores
    psi_text_length = calculate_numerical_psi(ref_lengths, prod_lengths)
    psi_confidence = calculate_numerical_psi(ref_confs, prod_confs)
    psi_concept = calculate_categorical_psi(ref_dist, prod_dist)

    ks_stat, ks_pvalue = ks_2samp(ref_lengths, prod_lengths)

    # Threshold checks
    length_drift = bool(psi_text_length > psi_alert_threshold)
    confidence_drift = bool(psi_confidence > psi_alert_threshold)
    concept_drift = bool(psi_concept > psi_alert_threshold)
    alert_triggered = bool(length_drift or confidence_drift or concept_drift)

    # 4. Update Prometheus Gauge if available
    try:
        from arabic_sentiment.api.app import PSI_GAUGE

        PSI_GAUGE.labels(feature="text_length").set(round(psi_text_length, 4))
        PSI_GAUGE.labels(feature="confidence_score").set(round(psi_confidence, 4))
    except Exception:  # noqa: BLE001, S110
        pass

    report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "baseline_samples": len(ref_lengths),
        "production_batch_samples": len(prod_lengths),
        "psi_threshold": float(psi_alert_threshold),
        "features": {
            "text_length": {
                "psi": round(float(psi_text_length), 4),
                "drift_detected": length_drift,
                "baseline_mean": round(float(np.mean(ref_lengths)), 1),
                "production_mean": round(float(np.mean(prod_lengths)), 1),
            },
            "confidence_score": {
                "psi": round(float(psi_confidence), 4),
                "drift_detected": confidence_drift,
                "baseline_mean": round(float(np.mean(ref_confs)), 4),
                "production_mean": round(float(np.mean(prod_confs)), 4),
            },
            "concept_sentiment": {
                "psi": round(float(psi_concept), 4),
                "drift_detected": concept_drift,
                "baseline_distribution": {l: round(float(p), 3) for l, p in zip(labels, ref_dist)},
                "production_distribution": {
                    l: round(float(p), 3) for l, p in zip(labels, prod_dist)
                },
            },
        },
        "ks_test": {
            "statistic": round(float(ks_stat), 4),
            "p_value": float(ks_pvalue),
        },
        "alert_triggered": alert_triggered,
    }

    # 5. Optional Evidently HTML report generation
    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        min_len = min(len(ref_lengths), len(ref_confs))
        ref_ev_df = pd.DataFrame(
            {"text_length": ref_lengths[:min_len], "confidence": ref_confs[:min_len]}
        )
        min_p_len = min(len(prod_lengths), len(prod_confs))
        prod_ev_df = pd.DataFrame(
            {"text_length": prod_lengths[:min_p_len], "confidence": prod_confs[:min_p_len]}
        )

        ev_report = Report(metrics=[DataDriftPreset()])
        ev_report.run(reference_data=ref_ev_df, current_data=prod_ev_df)
        ev_file = out_path / "evidently_drift_report.html"
        ev_report.save_html(str(ev_file))
        report["evidently_report_path"] = str(ev_file)
        print(f"[Evidently] Drift report generated: {ev_file}")
    except Exception as ev_err:  # noqa: BLE001
        print(f"[Note] Evidently report generation skipped or not installed: {ev_err}")

    # Save detailed JSON report
    report_file = out_path / "drift_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(
        f"\n[Feature Drift] Text Length PSI:      {psi_text_length:.4f} (Alert Threshold > {psi_alert_threshold})"
    )
    print(
        f"[Feature Drift] Confidence Score PSI: {psi_confidence:.4f} (Alert Threshold > {psi_alert_threshold})"
    )
    print(
        f"[Concept Drift] Class Shift PSI:      {psi_concept:.4f} (Alert Threshold > {psi_alert_threshold})"
    )

    if alert_triggered:
        alert_file = out_path / "drift_alert.json"
        with open(alert_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "status": "CRITICAL_ALERT",
                    "reason": "PSI drift exceeded threshold > 0.25",
                    "psi_text_length": psi_text_length,
                    "psi_confidence": psi_confidence,
                    "psi_concept": psi_concept,
                },
                f,
                indent=2,
            )
        print("\n" + "!" * 65)
        print(f" [ALERT TRIGGERED] CRITICAL: PSI > {psi_alert_threshold} detected!")
        print(f"  - Notification saved to: {alert_file}")
        print("!" * 65 + "\n")
    else:
        print(f"\n[OK] All PSI scores are within safe limits (<= {psi_alert_threshold}).")

    return report


if __name__ == "__main__":
    run_drift_analysis()
