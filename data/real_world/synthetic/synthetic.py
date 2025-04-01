import numpy as np
import pandas as pd


def generate_synthetic_high_cardinality_data(
        n_samples=10000,
        n_categories=100,
        random_seed=42,
        sensitive_val=None
):
    np.random.seed(random_seed)

    if sensitive_val is None:
        sensitive_attr = np.random.randint(0, n_categories, size=n_samples)
        sensitive_attr = [f"cat_{val}" for val in sensitive_attr]
    else:
        sensitive_attr = [f"cat_{val}" for val in (np.ones(n_samples)*sensitive_val).astype(int)]

    X1 = np.random.randn(n_samples)
    cat_to_int = lambda c: int(c.split('_')[1])
    sensitive_int = np.array(list(map(cat_to_int, sensitive_attr)))
    X2 = np.random.normal(0, 1, n_samples) + 0.1 * np.sqrt(sensitive_int)
    labels = 0.1 * np.sqrt(sensitive_int) + 0.1 * X1 + 0.1 * X2 + np.random.normal(0, 1, n_samples)

    df = pd.DataFrame({
        'A': sensitive_attr,
        'X1': X1,
        'X2': X2,
        'Y': labels
    })

    return df


if __name__ == "__main__":
    data = generate_synthetic_high_cardinality_data(n_samples=20000, n_categories=100, random_seed=42)
    data.to_csv("data.csv", index=False)
    print(data.head(10))
    print("\nData description:")
    print(data.describe(include='all'))

    data = generate_synthetic_high_cardinality_data(n_samples=1000, n_categories=100, random_seed=42, sensitive_val=0)
    data.to_csv("0.csv", index=False)

    data = generate_synthetic_high_cardinality_data(n_samples=1000, n_categories=100, random_seed=42, sensitive_val=49)
    data.to_csv("49.csv", index=False)

    data = generate_synthetic_high_cardinality_data(n_samples=1000, n_categories=100, random_seed=42, sensitive_val=99)
    data.to_csv("99.csv", index=False)

