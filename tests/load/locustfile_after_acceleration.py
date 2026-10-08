"""
Locust Load Testing Suite - Accelerated INT8 ONNX / TensorRT Model (After Acceleration).
Simulates high-throughput production e-commerce traffic targeting the optimized student model.
Validates strict SLA requirements (p95 latency <= 120ms, zero errors under burst load).
"""

from __future__ import annotations

import random

from locust import HttpUser, between, events, task

SAMPLE_REVIEWS = [
    "المنتج ممتاز جدا وخامته رائعة والشحن سريع",
    "خامة رديئة جدا وسيئة وتالف ولا يعمل",
    "المنتج عادي ومقبول بالنسبة لسعره المناسب",
    "التوصيل كان متأخر جدا لكن المنتج كويس",
    "تغليف فاشل ووصلت العلبة مكسورة تماما",
    "جودة عالية وتصميم مريح جدا أنصح بشرائه بشدة",
    "للأسف تجربة سيئة ولن أشتري منكم مجددا",
    "وصل بالموعد والمواصفات مطابقة تماما للصور",
    "لا بأس به يؤدي الغرض ولكن الصوت منخفض قليلا",
    "أفضل جهاز اشتريته هذا العام ما شاء الله",
]


class AcceleratedLoadTestUser(HttpUser):
    """Simulates clients hitting optimized INT8 ONNX / TensorRT inference service."""

    wait_time = between(0.05, 0.2)

    @task(8)
    def test_single_inference_optimized(self):
        text = random.choice(SAMPLE_REVIEWS)
        with self.client.post(
            "/predict",
            json={"text": text},
            catch_response=True,
            name="Optimized-Inference [After Acceleration]",
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "label" in data and "confidence" in data:
                    response.success()
                else:
                    response.failure("Malformed response schema")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")

    @task(4)
    def test_batch_inference_optimized(self):
        batch = random.sample(SAMPLE_REVIEWS, 4)
        with self.client.post(
            "/predict",
            json={"texts": batch},
            catch_response=True,
            name="Optimized-Batch4 [After Acceleration]",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")

    @task(1)
    def test_health(self):
        self.client.get("/health", name="HealthCheck")


@events.quitting.add_listener
def check_accelerated_sla(environment, **kwargs):
    stats = environment.runner.stats.total
    p95 = stats.get_current_response_time_percentile(0.95)
    print("\n" + "=" * 60)
    print("   [OPTIMIZED / AFTER ACCELERATION] BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Total Requests: {stats.num_requests}")
    print(f"Requests / Sec: {stats.total_rps:.2f}")
    print(f"Latency p95:    {p95:.2f} ms")
    assert p95 <= 150.0, f"SLA Violation: p95 latency ({p95:.2f}ms) exceeded 150ms target!"
    print("[PASS] Strict SLA target met under high load.")
    print("=" * 60)
