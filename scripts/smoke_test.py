"""Post-deploy smoke test: verifies the live deployed app is reachable and
serving predictions correctly. Run manually as:

    python scripts/smoke_test.py https://your-app.onrender.com

or automatically from the CI/CD workflow after each deploy to main.
"""

import json
import os
import sys

import requests

DEMO_ROWS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app", "demo_rows.json"
)


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/smoke_test.py <base_url>")
        sys.exit(2)

    base_url = sys.argv[1].rstrip("/")

    # 1. Health check.
    r = requests.get(f"{base_url}/health", timeout=15)
    assert r.status_code == 200, f"/health returned {r.status_code}"
    assert r.json().get("status") == "ok", f"unexpected /health body: {r.text}"
    print("[OK] /health")

    # 2. Home page loads.
    r = requests.get(f"{base_url}/", timeout=15)
    assert r.status_code == 200, f"/ returned {r.status_code}"
    print("[OK] / (index page)")

    # 3. Prediction API returns a sane response for a known malware sample.
    with open(DEMO_ROWS_PATH) as f:
        demo_rows = json.load(f)

    r = requests.post(f"{base_url}/predict", json=demo_rows["malware"], timeout=15)
    assert r.status_code == 200, f"/predict returned {r.status_code}: {r.text}"
    body = r.json()
    assert body["prediction"] == 1, f"expected malware prediction, got: {body}"
    print("[OK] /predict (malware sample)")

    r = requests.post(f"{base_url}/predict", json=demo_rows["goodware"], timeout=15)
    assert r.status_code == 200, f"/predict returned {r.status_code}: {r.text}"
    body = r.json()
    assert body["prediction"] == 0, f"expected goodware prediction, got: {body}"
    print("[OK] /predict (goodware sample)")

    print("\nAll smoke tests passed against", base_url)


if __name__ == "__main__":
    main()
