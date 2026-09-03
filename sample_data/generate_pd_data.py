"""Generates sample_data/pd_model_data.csv: 1000 synthetic loan applications
for a probability-of-default (PD) model demo. Deterministic (fixed seed) so
it's reproducible; re-run this script to regenerate the CSV.

The default flag isn't random noise -- it's drawn from a logistic function
of the other fields (credit score, DTI, utilization, late payments, loan
size relative to income, employment length), so the dataset actually
supports fitting and validating a toy PD model end to end.
"""

from __future__ import annotations

import csv
import math
import random
from datetime import date, timedelta
from pathlib import Path

random.seed(42)

N_ROWS = 1000

HOME_OWNERSHIP = ["RENT", "MORTGAGE", "OWN", "OTHER"]
HOME_OWNERSHIP_WEIGHTS = [0.40, 0.42, 0.15, 0.03]

PURPOSE = [
    "debt_consolidation",
    "credit_card",
    "home_improvement",
    "major_purchase",
    "medical",
    "small_business",
    "car",
    "vacation",
    "other",
]
PURPOSE_WEIGHTS = [0.32, 0.22, 0.10, 0.08, 0.06, 0.06, 0.08, 0.03, 0.05]

REGION = ["Northeast", "Midwest", "South", "West"]
REGION_WEIGHTS = [0.18, 0.21, 0.38, 0.23]

LOAN_TERM_MONTHS = [36, 60]
LOAN_TERM_WEIGHTS = [0.7, 0.3]


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def gen_row(app_id: int) -> dict:
    age = round(clamp(random.gauss(41, 12), 18, 75))
    employment_years = round(clamp(random.gauss(min(age - 18, 12), 6), 0, age - 16), 1)
    annual_income = round(clamp(random.lognormvariate(math.log(55000), 0.45), 15000, 350000), 2)

    credit_score = round(clamp(random.gauss(690, 80), 300, 850))
    dti = round(clamp(random.gauss(22, 10) + (5 if annual_income < 35000 else 0), 0, 65), 1)
    revolving_utilization = round(clamp(random.betavariate(2, 3) + (300 - min(credit_score, 700)) / 1000, 0, 1), 3)
    num_open_credit_lines = max(0, round(random.gauss(8, 4)))
    num_late_payments_2yr = max(0, round(random.gammavariate(1.2, 0.6) + (0.6 if credit_score < 600 else 0)))

    home_ownership = random.choices(HOME_OWNERSHIP, HOME_OWNERSHIP_WEIGHTS)[0]
    purpose = random.choices(PURPOSE, PURPOSE_WEIGHTS)[0]
    region = random.choices(REGION, REGION_WEIGHTS)[0]
    loan_term_months = random.choices(LOAN_TERM_MONTHS, LOAN_TERM_WEIGHTS)[0]

    loan_amount = round(clamp(random.lognormvariate(math.log(12000), 0.55), 1000, 45000), 2)
    base_rate = 5.0 + (850 - credit_score) / 550 * 20.0
    interest_rate = round(clamp(base_rate + random.gauss(0, 1.2), 5.0, 29.9), 2)

    days_ago = random.randint(0, 365 * 3)
    application_date = date(2026, 9, 3) - timedelta(days=days_ago)

    loan_to_income = loan_amount / max(annual_income, 1.0)

    z = (
        -4.7
        + 3.1 * ((650 - credit_score) / 100.0)
        + 0.03 * dti
        + 0.5 * num_late_payments_2yr
        + 1.4 * revolving_utilization
        + 2.0 * loan_to_income
        - 0.02 * employment_years
        + (0.2 if purpose == "small_business" else 0.0)
        + (-0.15 if home_ownership == "OWN" else 0.0)
        + random.gauss(0, 0.45)
    )
    default_probability = sigmoid(z)
    default_flag = 1 if random.random() < default_probability else 0

    return {
        "application_id": f"APP{app_id:06d}",
        "application_date": application_date.isoformat(),
        "age": age,
        "annual_income": annual_income,
        "employment_years": employment_years,
        "home_ownership": home_ownership,
        "region": region,
        "purpose": purpose,
        "loan_amount": loan_amount,
        "loan_term_months": loan_term_months,
        "interest_rate": interest_rate,
        "credit_score": credit_score,
        "dti": dti,
        "revolving_utilization": revolving_utilization,
        "num_open_credit_lines": num_open_credit_lines,
        "num_late_payments_2yr": num_late_payments_2yr,
        "default_flag": default_flag,
    }


def main() -> None:
    rows = [gen_row(i + 1) for i in range(N_ROWS)]
    out_path = Path(__file__).parent / "pd_model_data.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    default_rate = sum(r["default_flag"] for r in rows) / len(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")
    print(f"Overall default rate: {default_rate:.1%}")


if __name__ == "__main__":
    main()
