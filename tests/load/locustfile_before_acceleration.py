"""
Locust Load Testing Suite - Baseline Model (Before Acceleration).
Simulates traffic targeting the unoptimized PyTorch FP32 Teacher model (AraBERTv02).
Validates baseline latency baseline before model compression and quantization.
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
]


class BaselineLoadTestUser(HttpUser):
    """Simulates clients hitting unaccelerated baseline inference service."""

    wait_time = between(0.2, 0.8)

    @task(5)
    def test_single_inference_baseline(self):
        text = random.choice(SAMPLE_REVIEWS)
        with self.client.post(
            "/predict",
            json={"text": text},
            catch_response=True,
            name="Baseline-Inference [Before Acceleration]",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")

    @task(1)
    def test_health(self):
        self.client.get("/health", name="HealthCheck")


@events.quitting.add_listener
def check_baseline_stats(environment, **kwargs):
    stats = environment.runner.stats.total
    p95 = stats.get_current_response_time_percentile(0.95)
    print("\n" + "=" * 60)
    print("   [BASELINE / BEFORE ACCELERATION] BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Total Requests: {stats.num_requests}")
    print(f"Requests / Sec: {stats.total_rps:.2f}")
    print(f"Latency p95:    {p95:.2f} ms")
    print("=" * 60)
