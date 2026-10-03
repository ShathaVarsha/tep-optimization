from src.data.tep_loader import load_tep_dataset


def main():
    # Load the normal-operation dataset
    data = load_tep_dataset("d00")

    # Variables needed to inspect the cost and energy models
    flow_energy_columns = ["XMEAS_10", "XMEAS_19", "XMEAS_20"]
    composition_columns = [f"XMEAS_{i}" for i in range(29, 37)]
    columns = flow_energy_columns + composition_columns

    print("=" * 70)
    print("TENNESSEE EASTMAN PROCESS - MODEL INPUT VALIDATION")
    print("=" * 70)

    print("\n1. DATASET INFORMATION")
    print("Dataset ID:", data.attrs.get("dataset_id", "Unknown"))
    print("Dataset shape:", data.shape)

    missing = [col for col in columns if col not in data.columns]
    if missing:
        print("\nERROR: Missing expected columns:", missing)
        print("Available columns:", data.columns.tolist())
        return

    selected = data[columns]

    print("\n2. DATA QUALITY")
    print("Missing values:", int(selected.isna().sum().sum()))
    print("Infinite values:", int(selected.isin([float("inf"), float("-inf")]).sum().sum()))

    print("\n3. FLOW AND ENERGY VARIABLES")
    print(selected[flow_energy_columns].describe().round(4).to_string())

    print("\n4. PURGE COMPONENT COMPOSITIONS")
    print(selected[composition_columns].describe().round(4).to_string())

    print("\n5. PURGE COMPOSITION SUM")
    composition_sum = data[composition_columns].sum(axis=1)
    print(composition_sum.describe().round(4).to_string())

    print("\n6. COMPOSITION SCALE INSPECTION")
    maximum = data[composition_columns].max().max()
    minimum = data[composition_columns].min().min()

    print(f"Observed minimum: {minimum:.4f}")
    print(f"Observed maximum: {maximum:.4f}")

    if minimum >= 0 and maximum <= 1:
        print("Observed values fit a 0-1 scale.")
    elif minimum >= 0 and maximum <= 100:
        print("Observed values fit a 0-100 scale.")
    else:
        print("WARNING: Check the composition scale and variable definitions.")

    print("\nNOTE")
    print("This script inspects observed data; it does not validate")
    print("the physical units or economic parameters independently.")

    print("\nInput inspection completed.")


if __name__ == "__main__":
    main()
