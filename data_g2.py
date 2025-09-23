import numpy as np
import pandas as pd

def make_correlated_event_synthetic(config, path="./", padding=True):
    """
    Generate synthetic survival data with strictly ordered correlated event times.
    Each sample's event times are deterministically ordered to produce C-index ~ 1.0
    for intra-sample event rankings.
    """
    num_event = config["num_events"]
    min_time = config.get("min_time", 400)
    max_time = config.get("max_time", 1000)

    np.random.seed(42)
    num_data = 5000 * num_event

    # Shared latent base per sample
    z = np.random.normal(0, 1, size=num_data)

    # Deterministic event ordering: base + increasing offsets
    base = z[:, np.newaxis]  # shape (num_data, 1)
    offsets = np.linspace(0, 1, num_event)  # event-specific fixed offsets
    fix_offset = np.random.normal(0, 0.5)  # small random offsets is for all events, extend it as a vector
    fix_offsets = np.full((num_event,), fix_offset)
    offsets += fix_offsets  # shape (num_event,)
    t = base + offsets  # shape (num_data, num_events)
    # t += np.random.normal(0, 0.0001, size=t.shape)  # small noise

    # --- Rescale t into [min_time, max_time] ---
    t_min, t_max = t.min(), t.max()
    t = (t - t_min) / (t_max - t_min)  # [0,1]
    t = t * (max_time - min_time) + min_time

    labels = np.ones_like(t)

    # Apply right censoring at median time
    horizon = np.percentile(t, 50)
    for i in range(num_event):
        censored = t[:, i] > horizon
        t[censored, i] = horizon
        labels[censored, i] = 0

    # Use z as the feature
    x = z[:, np.newaxis]
    if padding and x.shape[1] < 30:
        pad = np.random.normal(0, 1, size=(x.shape[0], 30 - x.shape[1]))
        x = np.hstack([x, pad])

    # Build DataFrame
    feature_names = [f"x_{i+1}" for i in range(x.shape[1])]
    df = pd.DataFrame(x, columns=feature_names)
    for i in range(num_event):
        df[f"duration{i+1}"] = t[:, i]
    for i in range(num_event):
        df[f"event{i+1}"] = labels[:, i]
    df.insert(0, "id", np.arange(1, num_data + 1))
    filename = f"correlated_linear_data_event{num_event}.csv"
    df.to_csv(path + filename, index=False)
    print(f"Saved to {path}{filename}")

    for i in range(num_event):
        print(f"event {i+1} censor percentage:")
        print(f"  - Censored: {np.sum(labels[:, i] == 0)}")
        print(f"  - Uncensored: {np.sum(labels[:, i] == 1)}")

    return x, t, labels

def compute_intra_sample_event_ranking_accuracy(t):
    """
    Computes the fraction of samples where the predicted event ranking
    matches the true event order (ascending order of event IDs).
    """
    n_samples, n_events = t.shape
    correct = 0
    true_order = np.arange(n_events)

    for i in range(n_samples):
        pred_order = np.argsort(t[i])
        if np.array_equal(pred_order, true_order):
            correct += 1

    return correct / n_samples

if __name__ == "__main__":
    config = {
        "num_events": 8,
        "min_time": 400,
        "max_time": 1000,
    }

    x, t, labels = make_correlated_event_synthetic(config, path="/Users/dingzhu/code/sat/data/hsa-synthetic/", padding=False)
    acc = compute_intra_sample_event_ranking_accuracy(t)
    print(f"\nExact match rate of event ranking within each sample: {acc:.3f}")