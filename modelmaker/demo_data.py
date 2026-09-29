"""Synthetic retail credit-risk dataset for the demo projects: one row per
facility, usable for all three IRB-style parameters.

* PD  -- every row. `default_flag` is whether the facility defaulted within
  12 months of `reference_date` (the cohort start); every other borrower
  and facility field is measured at that date. The flag is drawn from a
  logistic function of those fields, so a PD model has real signal to find.
* CCF -- defaulted credit-card rows (`product_type == "credit_card"`).
  `credit_limit`, `balance_at_reference` and `balance_at_default` are
  exactly what the compute_ccf block takes. Term loans are fully drawn at
  disbursement, so they have no `credit_limit` (empty) and no CCF.
* LGD -- every defaulted row. `balance_at_default` is the EAD,
  `recovery_amount` the recoveries already discounted to the default date,
  `workout_cost` the direct workout costs -- what the compute_lgd block
  takes. Realised LGD comes out bimodal: cures (`cure_flag`) sit near 0,
  unsecured write-offs pile up near 1, and secured facilities
  (`is_secured`, `collateral_value`) land in between depending on
  loan-to-value.

Default-only fields (`default_date`, `balance_at_default`,
`recovery_amount`, `workout_cost`, `workout_months`, `cure_flag`) are empty
on non-defaulted rows.

A small share of observations fall outside [0, 1] on purpose -- a card
balance that fell before default (negative CCF), a limit increase (CCF > 1),
a workout that cost more than it recovered (LGD > 1) -- because real data
does that and compute_lgd/compute_ccf's floor/cap exist for it.

Deterministic: a fixed seed and a private RNG, so the same version of this
module always writes the byte-identical CSV. The CSV is neither committed
nor shipped in the PyPI package: `modelmaker-demo` (see create_workspace)
writes it next to a copy of the demo projects, tests/conftest.py writes it
for the test suite, and it can be written anywhere with:

    python -m modelmaker.demo_data path/to/credit_risk_data.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import shutil
from datetime import date, timedelta
from pathlib import Path

SEED = 42
N_ROWS = 5000
FILENAME = "credit_risk_data.csv"

# Cohort starts span three years up to this date; a 12-month outcome window
# after each one is assumed to be fully observed.
LAST_REFERENCE_DATE = date(2025, 6, 30)

DISCOUNT_RATE = 0.08  # annual, for discounting recoveries to the default date

HOME_OWNERSHIP = ["RENT", "MORTGAGE", "OWN", "OTHER"]
HOME_OWNERSHIP_WEIGHTS = [0.40, 0.42, 0.15, 0.03]

REGION = ["Northeast", "Midwest", "South", "West"]
REGION_WEIGHTS = [0.18, 0.21, 0.38, 0.23]

PRODUCT_TYPE = ["term_loan", "credit_card"]
PRODUCT_TYPE_WEIGHTS = [0.6, 0.4]

TERM_LOAN_PURPOSE = [
    "debt_consolidation",
    "home_improvement",
    "major_purchase",
    "medical",
    "small_business",
    "car",
    "vacation",
    "other",
]
TERM_LOAN_PURPOSE_WEIGHTS = [0.32, 0.16, 0.09, 0.06, 0.06, 0.20, 0.04, 0.07]

LOAN_TERM_MONTHS = [36, 60]
LOAN_TERM_WEIGHTS = [0.7, 0.3]

FIELDS = [
    "application_id",
    "application_date",
    "reference_date",
    "product_type",
    "purpose",
    "age",
    "annual_income",
    "employment_years",
    "home_ownership",
    "region",
    "credit_score",
    "dti",
    "revolving_utilization",
    "num_open_credit_lines",
    "num_late_payments_2yr",
    "loan_amount",
    "loan_term_months",
    "interest_rate",
    "credit_limit",
    "balance_at_reference",
    "is_secured",
    "collateral_type",
    "collateral_value",
    "default_flag",
    "default_date",
    "balance_at_default",
    "recovery_amount",
    "workout_cost",
    "workout_months",
    "cure_flag",
]


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def _add_months(d: date, months: int) -> date:
    # Day 28 at most, so every month has it.
    year, month = divmod(d.month - 1 + months, 12)
    return date(d.year + year, month + 1, min(d.day, 28))


def _amortised_balance(principal: float, annual_rate: float, term: int, months_paid: int) -> float:
    """Outstanding principal on a level-payment loan after `months_paid`."""
    r = annual_rate / 12.0
    if months_paid >= term:
        return 0.0
    growth = (1 + r) ** months_paid
    payment = principal * r / (1 - (1 + r) ** -term)
    return principal * growth - payment * (growth - 1) / r


def _row(rng: random.Random, row_id: int) -> dict:
    # --- borrower, measured at the reference date ------------------------
    age = round(_clamp(rng.gauss(41, 12), 18, 75))
    employment_years = round(_clamp(rng.gauss(min(age - 18, 12), 6), 0, age - 16), 1)
    annual_income = round(_clamp(rng.lognormvariate(math.log(55000), 0.45), 15000, 350000), 2)
    credit_score = round(_clamp(rng.gauss(690, 80), 300, 850))
    dti = round(_clamp(rng.gauss(22, 10) + (5 if annual_income < 35000 else 0), 0, 65), 1)
    revolving_utilization = round(
        _clamp(rng.betavariate(2, 3) + (300 - min(credit_score, 700)) / 1000, 0, 1), 3
    )
    num_open_credit_lines = max(1, round(rng.gauss(8, 4)))
    num_late_payments_2yr = max(
        0, round(rng.gammavariate(1.2, 0.6) + (0.6 if credit_score < 600 else 0))
    )
    home_ownership = rng.choices(HOME_OWNERSHIP, HOME_OWNERSHIP_WEIGHTS)[0]
    region = rng.choices(REGION, REGION_WEIGHTS)[0]

    # --- facility ---------------------------------------------------------
    product_type = rng.choices(PRODUCT_TYPE, PRODUCT_TYPE_WEIGHTS)[0]
    reference_date = LAST_REFERENCE_DATE - timedelta(days=rng.randint(0, 365 * 3))
    base_rate = 5.0 + (850 - credit_score) / 550 * 20.0

    credit_limit = None
    collateral_type = "none"
    collateral_value = 0.0

    if product_type == "term_loan":
        purpose = rng.choices(TERM_LOAN_PURPOSE, TERM_LOAN_PURPOSE_WEIGHTS)[0]
        loan_term_months = rng.choices(LOAN_TERM_MONTHS, LOAN_TERM_WEIGHTS)[0]
        loan_amount = round(_clamp(rng.lognormvariate(math.log(12000), 0.55), 1000, 45000), 2)
        interest_rate = round(_clamp(base_rate + rng.gauss(0, 1.2), 5.0, 29.9), 2)
        months_on_book = rng.randint(0, min(24, loan_term_months - 13))
        application_date = _add_months(reference_date, -months_on_book)
        balance_at_reference = _amortised_balance(
            loan_amount, interest_rate / 100, loan_term_months, months_on_book
        )
        if purpose == "car" and rng.random() < 0.9:
            collateral_type = "vehicle"
            # Bought at roughly 80-110% LTV, depreciating ~15% a year.
            new_value = loan_amount / rng.uniform(0.8, 1.1)
            collateral_value = new_value * 0.85 ** (months_on_book / 12)
        elif home_ownership in ("MORTGAGE", "OWN") and (
            (purpose == "home_improvement" and rng.random() < 0.8)
            or (purpose == "debt_consolidation" and rng.random() < 0.3)
        ):
            collateral_type = "real_estate"
            # A junior lien: the loan is a small slice of the home's equity.
            collateral_value = balance_at_reference / rng.uniform(0.15, 0.7)
    else:
        purpose = "credit_card"
        loan_term_months = None
        interest_rate = round(_clamp(base_rate + 4.0 + rng.gauss(0, 1.5), 9.9, 29.9), 2)
        credit_limit = float(
            round(_clamp(annual_income * rng.uniform(0.08, 0.3) * (credit_score / 700) ** 2, 500, 50000), -2)
        )
        loan_amount = credit_limit  # the commitment
        months_on_book = rng.randint(3, 96)
        application_date = _add_months(reference_date, -months_on_book)
        facility_utilisation = _clamp(
            0.7 * revolving_utilization + 0.3 * rng.betavariate(1.5, 3) + rng.gauss(0, 0.08), 0.02, 1.05
        )
        balance_at_reference = credit_limit * facility_utilisation

    is_secured = 1 if collateral_type != "none" else 0
    loan_to_income = loan_amount / max(annual_income, 1.0)

    # --- PD: default within 12 months of the reference date ---------------
    z = (
        -4.0
        + 2.0 * ((650 - credit_score) / 100.0)
        + 0.03 * dti
        + 0.5 * num_late_payments_2yr
        + 1.4 * revolving_utilization
        + 2.0 * loan_to_income
        - 0.02 * employment_years
        + (0.25 if product_type == "credit_card" else 0.0)
        + (0.2 if purpose == "small_business" else 0.0)
        + (-0.3 if is_secured else 0.0)
        + (-0.15 if home_ownership == "OWN" else 0.0)
        + rng.gauss(0, 0.9)
    )
    default_flag = 1 if rng.random() < _sigmoid(z) else 0

    row = {
        "application_id": f"APP{row_id:06d}",
        "application_date": application_date.isoformat(),
        "reference_date": reference_date.isoformat(),
        "product_type": product_type,
        "purpose": purpose,
        "age": age,
        "annual_income": annual_income,
        "employment_years": employment_years,
        "home_ownership": home_ownership,
        "region": region,
        "credit_score": credit_score,
        "dti": dti,
        "revolving_utilization": revolving_utilization,
        "num_open_credit_lines": num_open_credit_lines,
        "num_late_payments_2yr": num_late_payments_2yr,
        "loan_amount": loan_amount,
        "loan_term_months": loan_term_months,
        "interest_rate": interest_rate,
        "credit_limit": credit_limit,
        "balance_at_reference": round(balance_at_reference, 2),
        "is_secured": is_secured,
        "collateral_type": collateral_type,
        "collateral_value": round(collateral_value, 2),
        "default_flag": default_flag,
        "default_date": None,
        "balance_at_default": None,
        "recovery_amount": None,
        "workout_cost": None,
        "workout_months": None,
        "cure_flag": None,
    }
    if not default_flag:
        return row

    # --- EAD / CCF --------------------------------------------------------
    months_to_default = rng.randint(1, 12)
    default_date = reference_date + timedelta(days=rng.randint(28 * months_to_default - 27, 28 * months_to_default + 2))

    if product_type == "term_loan":
        # Missed payments stop amortisation part-way; arrears interest accrues.
        paid = months_on_book + max(0, months_to_default - rng.randint(2, 4))
        ead = _amortised_balance(loan_amount, interest_rate / 100, loan_term_months, paid)
        ead *= 1 + interest_rate / 100 * rng.uniform(0.1, 0.3)
    else:
        undrawn = max(credit_limit - balance_at_reference, 0.0)
        u = rng.random()
        if u < 0.08 and balance_at_reference > 0.2 * credit_limit:
            # Paid down before defaulting anyway: a negative raw CCF.
            ead = balance_at_reference * rng.uniform(0.7, 0.98)
        elif u < 0.13:
            # Limit raised / over-limit at default: raw CCF above 1.
            ead = balance_at_reference + undrawn * rng.uniform(1.02, 1.4) + credit_limit * rng.uniform(0.0, 0.05)
        else:
            # Weaker borrowers draw more; lines already nearly full have
            # less of their (small) headroom left to use.
            mean = _sigmoid(
                0.6
                + 0.8 * ((650 - credit_score) / 100.0)
                - 1.5 * (balance_at_reference / credit_limit)
                + 0.3 * num_late_payments_2yr
            )
            k = 6.0
            ead = balance_at_reference + undrawn * rng.betavariate(max(mean * k, 0.2), max((1 - mean) * k, 0.2))
    # A defaulted facility always owes something.
    ead = max(ead, 100.0)

    # --- LGD --------------------------------------------------------------
    collateral_at_default = collateral_value * (0.85 ** (months_to_default / 12) if collateral_type == "vehicle" else 1.0)
    cure_z = (
        -1.5
        + 0.35 * ((credit_score - 600) / 100.0)
        + 0.03 * employment_years
        + (0.6 if is_secured else 0.0)
        - (0.4 if product_type == "credit_card" else 0.0)
    )
    cure_flag = 1 if rng.random() < _sigmoid(cure_z) else 0

    if cure_flag:
        workout_months = rng.randint(2, 8)
        recovered_nominal = ead * rng.uniform(0.97, 1.02)
        workout_cost = 50 + ead * rng.uniform(0.0, 0.02)
    elif is_secured:
        workout_months = rng.randint(6, 24)
        haircut = rng.uniform(0.55, 0.8) if collateral_type == "vehicle" else rng.uniform(0.7, 0.9)
        from_collateral = min(ead, collateral_at_default * haircut)
        from_borrower = (ead - from_collateral) * rng.betavariate(1.2, 6)
        recovered_nominal = from_collateral + from_borrower
        workout_cost = 250 + ead * rng.uniform(0.02, 0.06)
    else:
        workout_months = rng.randint(12, 36)
        rate_mean = _clamp(
            0.32 + 0.10 * ((credit_score - 600) / 100.0) + 0.05 * (1 if home_ownership != "RENT" else 0), 0.05, 0.6
        )
        k = 3.0
        recovered_nominal = ead * rng.betavariate(rate_mean * k, (1 - rate_mean) * k)
        workout_cost = 75 + ead * rng.uniform(0.01, 0.05)

    recovery_amount = recovered_nominal / (1 + DISCOUNT_RATE) ** (workout_months / 12)

    row.update(
        default_date=default_date.isoformat(),
        balance_at_default=round(ead, 2),
        recovery_amount=round(recovery_amount, 2),
        workout_cost=round(workout_cost, 2),
        workout_months=workout_months,
        cure_flag=cure_flag,
    )
    return row


def generate(n_rows: int = N_ROWS, seed: int = SEED) -> list[dict]:
    rng = random.Random(seed)
    return [_row(rng, i + 1) for i in range(n_rows)]


def write_csv(path: Path, n_rows: int = N_ROWS, seed: int = SEED) -> Path:
    """Writes the dataset to `path` (parent directories created) and
    returns it. Same arguments, same bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = generate(n_rows, seed)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def bundled_projects_dir() -> Path | None:
    """Folder holding the demo project folders: modelmaker/examples/ in a
    built wheel (copied there from the repo's projects/ by setup.py), or
    the repo's own projects/ for an editable install. None if neither."""
    package_dir = Path(__file__).resolve().parent
    for candidate in (package_dir / "examples", package_dir.parent / "projects"):
        if (candidate / "demo_pd_model" / "model.json").is_file():
            return candidate
    return None


def create_workspace(dest: Path) -> Path:
    """Lays out a runnable copy of the demo in `dest`, mirroring the repo:

        dest/projects/<each demo project>/model.json
        dest/sample_data/credit_risk_data.csv   (generated, not copied)

    Projects save back into their own folder, so they are copied somewhere
    writable instead of being opened inside site-packages. Idempotent: an
    existing project folder is left as is (it may hold the user's edits),
    and the CSV is only written if missing. Start modelmaker-api/-tui from
    `dest` -- the projects read their data by that cwd-relative path."""
    source = bundled_projects_dir()
    if source is None:
        raise FileNotFoundError("no bundled demo projects found in this install")
    dest = Path(dest).resolve()
    for project in sorted(p for p in source.iterdir() if (p / "model.json").is_file()):
        target = dest / "projects" / project.name
        if not target.exists():
            shutil.copytree(project, target)
    data = dest / "sample_data" / FILENAME
    if not data.exists():
        write_csv(data)
    return dest


def workspace_main(argv: list[str] | None = None) -> None:
    """Entry point for the `modelmaker-demo` console script."""
    parser = argparse.ArgumentParser(
        prog="modelmaker-demo",
        description="Set up a folder with the demo projects and their (generated) sample data.",
    )
    parser.add_argument("dest", nargs="?", default="modelmaker-demo", help="folder to create (default: ./modelmaker-demo)")
    args = parser.parse_args(argv)
    try:
        dest = create_workspace(Path(args.dest))
    except FileNotFoundError as exc:
        raise SystemExit(f"modelmaker-demo: {exc}") from exc
    print(f"Demo ready in {dest}")
    print(f"  cd {dest}")
    print("  modelmaker-api                         # then Load projects/demo_pd_model and Run all")
    print("  modelmaker-tui projects/demo_pd_model  # or the terminal UI")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m modelmaker.demo_data", description="Write the synthetic PD/LGD/CCF demo dataset."
    )
    parser.add_argument("out", nargs="?", default=FILENAME, help=f"output CSV path (default: ./{FILENAME})")
    parser.add_argument("--rows", type=int, default=N_ROWS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    out = write_csv(Path(args.out), args.rows, args.seed)
    rows = generate(args.rows, args.seed)
    defaults = [r for r in rows if r["default_flag"]]
    cards = [r for r in defaults if r["product_type"] == "credit_card"]
    print(f"Wrote {len(rows)} rows to {out}")
    print(f"Default rate: {len(defaults) / len(rows):.1%} ({len(defaults)} defaults, {len(cards)} on credit cards)")


if __name__ == "__main__":
    main()
