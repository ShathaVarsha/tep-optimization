import numpy as np
import pandas as pd

from src.data.tep_loader import load_tep_dataset
from src.optimization.cost_model import evaluate_operating_cost_breakdown
from src.optimization.energy_model import evaluate_energy_breakdown


def main():
    data = load_tep_dataset("d00")

    print("=" * 65)
    print("COST AND ENERGY MODEL OUTPUT CHECK")
    print("=" * 65)

    # 1. Current cost calculation on the dataset mean
    mean_cost = evaluate_operating_cost_breakdown(data)

    # 2. Cost calculated for each row, then averaged
    row_costs = pd.DataFrame(
        [
            evaluate_operating_cost_breakdown(row)
            for _, row in data.iterrows()
        ]
    )
    average_row_cost = row_costs.mean().to_dict()

    print("\n1. COST BREAKDOWN: COST AT MEAN STATE")
    for name, value in mean_cost.items():
        print(f"{name}: ${value:.6f}/hr")

    print("\n2. COST BREAKDOWN: AVERAGE OF ROW-WISE COSTS")
    for name, value in average_row_cost.items():
        print(f"{name}: ${value:.6f}/hr")

    print("\n3. DIFFERENCE IN TOTAL COST")
    difference = (
        average_row_cost["total_operating_cost_usd_hr"]
        - mean_cost["total_operating_cost_usd_hr"]
    )
    print(f"Average row-wise minus mean-state: ${difference:.6f}/hr")

    # 4. Energy calculation
    energy = evaluate_energy_breakdown(data)

    print("\n4. ENERGY BREAKDOWN")
    for name, value in energy.items():
        print(f"{name}: {value:.6f} kW")

    # 5. Numerical sanity checks
    print("\n5. SANITY CHECKS")
    print("All cost outputs finite:", np.isfinite(list(mean_cost.values())).all())
    print("All energy outputs finite:", np.isfinite(list(energy.values())).all())
    print("All costs non-negative:", all(v >= 0 for v in mean_cost.values()))

    print("\nDiagnostic completed. These checks do not independently")
    print("validate economic parameters or thermodynamic assumptions.")


if __name__ == "__main__":
    main()
