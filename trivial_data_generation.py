import numpy as np
import pandas as pd
import copy

# Your original configurations
config_event_2 = {
    'num_events': 2,
    'num_bins': 20,
    'terminal_events': [1],
    'discrete': False,
    'event_ranks': {0:[], 1:[]},
    'event_groups': {0:[0, 1], 1:[0, 1]},
    'min_time': 0,
    'max_time': 20,
    'min_epoch': 50,
}

config_event_3 = {
    'num_events': 3,
    'num_bins': 20,
    'terminal_events': [2],
    'discrete': False,
    'event_ranks': {0:[], 1:[], 2:[]},
    'event_groups': {0:[0, 1], 1:[0, 1, 2], 2:[0, 1, 2]},
    'min_time': 0,
    'max_time': 20,
    'min_epoch': 50,
}

config_event_4 = {
    "num_events": 4,
    "num_bins": 20,
    "terminal_events": [3],
    "discrete": False,
    "event_ranks": {
        "0": [],
        "1": [],
        "2": [],
        "3": []
    },
    "event_groups": {
        "0": [],
        "1": [],
        "2": [],
        "3": []
    },
    "min_time": 0,
    "max_time": 20,
    "min_epoch": 50
}

config_event_6 = {
    "num_events": 6,
    "num_bins": 20,
    "terminal_events": [5],
    "discrete": False,
    "event_ranks": {
        "0": [],
        "1": [],
        "2": [],
        "3": [],
        "4": [],
        "5": []
    },
    "event_groups": {
        "0": [0, 1],
        "1": [0, 1],
        "2": [0, 1, 2],
        "3": [0, 1, 2, 3],
        "4": [0, 1, 2, 3, 4],
        "5": [0, 1, 2, 3, 4]
    },
    "min_time": 0,
    "max_time": 20,
    "min_epoch": 50
}

config_event_8 = {
    "num_events": 8,
    "num_bins": 20,
    "terminal_events": [7],
    "discrete": False,
    "event_ranks": {
        "0": [],
        "1": [],
        "2": [],
        "3": [],
        "4": [],
        "5": [],
        "6": [],
        "7": []
    },
    "event_groups": {
        "0": [0, 1],
        "1": [0, 1],
        "2": [0, 1, 2],
        "3": [0, 1, 2, 3],
        "4": [0, 1, 2, 3, 4],
        "5": [0, 1, 2, 3, 4, 5],
        "6": [0, 1, 2, 3, 4, 5, 6],
        "7": [0, 1, 2, 3, 4, 5, 6]
    },
    "min_time": 0,
    "max_time": 20,
    "min_epoch": 50
}

def make_trivial_synthetic(config, path="/Users/dingzhu/code/sat/data/hsa-synthetic/"):
    """
    Generate trivial synthetic survival data with multiple events.
    Each event is determined by only ONE feature (no complex interactions).
    
    Args:
        config: Configuration dictionary with num_events, etc.
        path: Path to save the generated data
    """
    
    num_event = config["num_events"]
    num_bins = config["num_bins"]
    terminal_events = config["terminal_events"]
    event_ranks = config["event_ranks"]
    event_groups = config["event_groups"]
    min_time = config["min_time"]
    max_time = config["max_time"]
    min_epoch = config["min_epoch"]
    
    print(f"Generating trivial synthetic data with {num_event} events")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate data size proportional to number of events (like original)
    num_data = 5000 * num_event
    
    # Generate features: num_event meaningful features + padding to reach 30 total
    target_feature_size = 30
    x_features = []
    feature_params = []
    
    # Generate meaningful features (one for each event)
    for i in range(num_event):
        # Different distribution parameters for each feature
        mean = np.random.uniform(-1, 1)  # Random mean between -1 and 1
        std = np.random.uniform(0.5, 2.0)  # Random std between 0.5 and 2.0
        feature = np.random.normal(loc=mean, scale=std, size=num_data)
        x_features.append(feature)
        feature_params.append((mean, std))
        print(f"Meaningful Feature {i+1}: mean={mean:.3f}, std={std:.3f}")
    
    # Pad with zeros to reach target feature size
    num_padding = max(0, target_feature_size - num_event)
    for i in range(num_padding):
        padding_feature = np.zeros(num_data)
        x_features.append(padding_feature)
        feature_params.append((0.0, 0.0))
    
    x = np.column_stack(x_features)
    
    print(f"Total features: {x.shape[1]} (Meaningful: {num_event}, Padding: {num_padding})")
    
    # Generate event times - each event determined by its corresponding feature
    # Scale to get durations in range 400-1000
    event_times = []
    base_hazards = np.random.uniform(0.001, 0.003, num_event)  # Much smaller base hazards
    coefficients = np.random.uniform(-0.0005, 0.0005, num_event)  # Smaller coefficients
    base_scale = 700  # Target around 700 as base duration
    
    print(f"\nEvent parameters:")
    for i in range(num_event):
        print(f"Event {i+1}: base_hazard={base_hazards[i]:.6f}, coeff={coefficients[i]:.6f}")
        
        # Calculate hazard for event i using feature i
        log_hazard = base_hazards[i] + coefficients[i] * x[:, i]
        hazard_rates = np.exp(log_hazard)
        
        # Generate event times from exponential distribution and scale
        times = np.random.exponential(scale=1/hazard_rates) * base_scale
        # Ensure times are in desired range
        times = np.clip(times, 0, 1000)
        event_times.append(times)
    
    # Stack event times
    t = np.column_stack(event_times)  # shape: (num_data, num_event)
    labels = np.ones(t.shape)
    
    # Apply event ranking constraints (fixed version)
    if any(len(ranks) > 0 for ranks in event_ranks.values()):
        print("Applying event ranking constraints...")
        t_original = copy.deepcopy(t)
        num_inconsist = 0
        
        for i in range(num_data):
            # Apply ranking constraints
            for e1_str, e2_list in event_ranks.items():
                if len(e2_list) > 0:
                    e1 = int(e1_str)
                    for e2 in e2_list:
                        e2 = int(e2)
                        if e1 < num_event and e2 < num_event:
                            if t_original[i, e1] < t_original[i, e2]:
                                # Add delay to ensure e1 happens before e2
                                delay = np.random.exponential(scale=50)  # Scaled delay
                                t[i, e2] = t_original[i, e2] + delay
                                # FIXED: Ensure we don't exceed 1000
                                t[i, e2] = min(t[i, e2], 1000)
                                if t[i, e2] < t_original[i, e1]:
                                    num_inconsist += 1
        
        print(f"Number of ranking inconsistencies: {num_inconsist}")
    else:
        # Simple random ranking adjustment (fixed version)
        print("Applying simple random ranking adjustments...")
        t_original = copy.deepcopy(t)
        num_inconsist = 0
        
        for i in range(num_data):
            if num_event >= 2:
                # Randomly select two events
                j, k = np.random.choice(num_event, size=2, replace=False)
                if t_original[i, j] < t_original[i, k]:
                    delay = np.random.exponential(scale=30)  # Scaled delay
                    t[i, k] = t_original[i, k] + delay
                    # FIXED: Ensure we don't exceed 1000
                    t[i, k] = min(t[i, k], 1000)
                    if t[i, k] < t_original[i, j]:
                        num_inconsist += 1
                elif t_original[i, k] < t_original[i, j]:
                    delay = np.random.exponential(scale=30)  # Scaled delay
                    t[i, j] = t_original[i, j] + delay
                    # FIXED: Ensure we don't exceed 1000
                    t[i, j] = min(t[i, j], 1000)
                    if t[i, j] < t_original[i, k]:
                        num_inconsist += 1
        
        print(f"Number of simple ranking inconsistencies: {num_inconsist}")
    
    # ADDITIONAL FIX: Ensure all times are within bounds after ranking adjustments
    t = np.clip(t, 0, 1000)
    print(f"After ranking adjustments - Min time: {t.min():.2f}, Max time: {t.max():.2f}")
    
    # Apply censoring (similar to original)
    horizon = np.percentile(t, 70)  # Use 70th percentile for higher censoring threshold
    horizons = np.random.lognormal(mean=np.log(horizon), sigma=0.05, size=num_data)
    horizons = np.clip(horizons, a_min=horizon*0.8, a_max=horizon*1.5)
    
    # FIXED: Ensure horizons don't exceed 1000
    horizons = np.clip(horizons, 0, 1000)
    
    print(f'\nEnforcing prediction horizon at {horizon:.2f}')
    print(f'Horizon range after noise: {horizons.min():.2f} – {horizons.max():.2f}')
    
    # Apply censoring
    for i in range(t.shape[1]):
        censored = t[:, i] > horizons
        t[censored, i] = horizons[censored]
        labels[censored, i] = 0
    
    # FINAL CHECK: Ensure all durations are within [0, 1000]
    t = np.clip(t, 0, 1000)
    print(f"Final check - Duration range: [{t.min():.2f}, {t.max():.2f}]")
    
    # Verify no values exceed 1000
    if np.any(t > 1000):
        print("WARNING: Some durations still exceed 1000!")
        print(f"Max duration: {t.max()}")
        print(f"Number of values > 1000: {np.sum(t > 1000)}")
    else:
        print("✓ All durations are within [0, 1000] range")
    
    # Create DataFrame
    feature_names = [f'x_{i+1}' for i in range(x.shape[1])]
    duration_names = [f'duration{i+1}' for i in range(num_event)]
    event_names = [f'event{i+1}' for i in range(num_event)]
    
    df = pd.DataFrame(x, columns=feature_names)
    
    for i in range(num_event):
        df[duration_names[i]] = t[:, i]
    for i in range(num_event):
        df[event_names[i]] = labels[:, i].astype(float)  # Use float instead of int
    
    df.insert(0, 'id', np.arange(1, x.shape[0] + 1))
    
    # Save to file
    filename = f'trivial_synthetic_data_event{num_event}.csv'
    full_path = path + filename
    df.to_csv(full_path, index=False)
    
    # Print summary statistics (same as original)
    print(f'\nSaved to {full_path}')
    print('Label distribution:\n', np.unique(labels, return_counts=True, axis=0))
    
    numbers = np.sum(labels, axis=0)
    print('Sample number of each event:', numbers)
    print('Sample percentage of each event:', numbers / num_data * 100)
    
    start_times = np.round(np.min(t, axis=0), 2)
    print('Start time of each event:', start_times)
    
    # Calculate longest observed times
    longest_times = []
    for i in range(num_event):
        observed_mask = labels[:, i] == 1
        if np.any(observed_mask):
            longest = np.max(t[observed_mask, i])
        else:
            longest = np.nan
        longest_times.append(longest)
    longest_times = np.round(longest_times, 2)
    print('Longest observed time of each event:', longest_times)
    print('Maximum censored times:', np.round(np.max(t, axis=0), 2))
    
    # Events per sample distribution
    events_per_sample = np.sum(labels, axis=1)
    unique_counts, count_freqs = np.unique(events_per_sample, return_counts=True)
    
    print("\nEvents per sample distribution:")
    for cnt, freq in zip(unique_counts, count_freqs):
        percentage = freq / len(events_per_sample) * 100
        print(f"  {cnt} events: {freq} samples ({percentage:.2f}%)")
    
    avg_events = np.mean(events_per_sample)
    print(f"Average events per sample: {avg_events:.2f}")
    
    # Generate LaTeX table (same as original)
    latex_lines = [
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "\\textbf{Event} & \\textbf{Observed} & \\textbf{Min Time} & \\textbf{Max Time} \\\\",
        "\\midrule"
    ]
    
    for i in range(num_event):
        name = f"Event {i+1}"
        observed = int(numbers[i])
        percentage = observed / num_data * 100
        min_time = start_times[i]
        max_time = longest_times[i]
        latex_lines.append(f"{name} & {observed:,} ({percentage:.2f}\\%) & {min_time:.2f} & {max_time:.2f} \\\\")
    
    latex_lines.extend(["\\bottomrule", "\\end{tabular}"])
    latex_table = "\n".join(latex_lines)
    
    # Save LaTeX table
    # tex_filename = f"{path}trivial_event_summary_event{num_event}.tex"
    # with open(tex_filename, "w") as f:
    #     f.write(latex_table)
    # print(f"\nLaTeX table saved to {tex_filename}")
    
    print(f"\n" + "="*60)
    print("TRIVIAL vs ORIGINAL COMPLEXITY COMPARISON:")
    print("="*60)
    print(f"Trivial version:")
    print(f"  - Features: {x.shape[1]} total ({num_event} meaningful + {num_padding} padding zeros)")
    print(f"  - Events: {num_event}")
    print(f"  - Feature-Event mapping: 1-to-1 (Event i uses Feature i)")
    print(f"  - Interactions: None (each event independent)")
    print(f"  - Segments: None")
    
    print(f"\nOriginal complex version would have:")
    print(f"  - Features: {(num_event+1)*6} ({num_event+1} segments × 6 features)")
    print(f"  - Events: {num_event}")
    print(f"  - Feature-Event mapping: Many-to-many (complex interactions)")
    print(f"  - Interactions: Complex segment-wise power functions")
    print(f"  - Segments: {num_event+1} with different bounds")
    
    return x, t, labels




import numpy as np
import pandas as pd

def make_trivial_linear_synthetic(config, path="./", padding=True):
    """
    Generate trivial synthetic survival data.
    Each event is determined by ONE feature with simple linear mapping.
    
    Args:
        config: dict, must include num_events, min_time, max_time
        path: str, save path
        padding: bool, whether to pad features to 30 dimensions
    """
    num_event = config["num_events"]
    min_time = config["min_time"] if "min_time" in config else 400
    max_time = config["max_time"] if "max_time" in config else 1000
    
    np.random.seed(42)
    num_data = 5000 * num_event
    
    # Features
    features = []
    feature_params = []
    for i in range(num_event):
        mean = np.random.uniform(-1, 1)
        std = np.random.uniform(0.5, 2.0)
        f = np.random.normal(loc=mean, scale=std, size=num_data)
        features.append(f)
        feature_params.append((mean, std))
        print(f"Feature {i+1}: mean={mean:.3f}, std={std:.3f}")
    
    # Pad if required
    if padding:
        num_padding = 30 - num_event
        for _ in range(num_padding):
            features.append(np.zeros(num_data))
    x = np.column_stack(features)
    
    # Durations: linear mapping to [400, 1000]
    durations = []
    for i in range(num_event):
        f = features[i]
        # Normalize feature to [0, 1]
        f_norm = (f - f.min()) / (f.max() - f.min())
        # Scale to [400, 1000]
        t = f_norm * (max_time - min_time) + min_time
        durations.append(t)
    # print(t)
    t = np.column_stack(durations)
    
    # Labels: all observed (no censoring)
    labels = np.ones_like(t)
    
    # Apply fixed horizon noise (optional, here ±50 uniformly)
    horizon = np.percentile(t, 50)
    
    
    for i in range(num_event):
        censored = t[:, i] > horizon
        t[censored, i] = horizon
        labels[censored, i] = 0
    
    # Build DataFrame
    feature_names = [f"x_{i+1}" for i in range(x.shape[1])]
    df = pd.DataFrame(x, columns=feature_names)
    for i in range(num_event):
        df[f"duration{i+1}"] = t[:, i]
    for i in range(num_event):
        df[f"event{i+1}"] = labels[:, i]
    df.insert(0, "id", np.arange(1, num_data + 1))
    
    filename = f"trivial_linear_data_event{num_event}.csv"
    df.to_csv(path + filename, index=False)
    print(f"Saved to {path}{filename}")

    for i in range(num_event):
        print(f"event {i+1} censor percentage:")
        print(f"  - Censored: {np.sum(labels[:, i] == 0)}")
        print(f"  - Uncensored: {np.sum(labels[:, i] == 1)}")

    return x, t, labels


# Example usage
if __name__ == "__main__":
    print("TRIVIAL SYNTHETIC DATA GENERATION - FIXED VERSION")
    print("="*50)
    
    # Test with different configurations
    configs_to_test = [
        ("2-event", config_event_2),
        ("3-event", config_event_3),
        ("4-event", config_event_4)
    ]
    
    path = "/Users/dingzhu/code/sat/data/hsa-synthetic/"
    
    for config_name, config in configs_to_test:
        print(f"\n{'='*20} {config_name.upper()} {'='*20}")
        # x, t, labels = make_trivial_synthetic(config, path)
        x, t, labels = make_trivial_linear_synthetic(config, path, padding=False)
        print(f"Generated data shape: x={x.shape}, t={t.shape}, labels={labels.shape}")
        print(f"Duration range verification: [{t.min():.2f}, {t.max():.2f}]")
        print("-" * 60)