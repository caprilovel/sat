
import numpy as np
import pandas as pd
import copy

config_event_2 = {
    'num_events': 2, \
    'num_bins': 20, \
    'terminal_events': [1], \
    'discrete': False, \
    'event_ranks': {0:[], 1:[]}, \
    'event_groups': {0:[0, 1], 1:[0, 1]}, \
    'min_time': 0, \
    'max_time': 20, \
    'min_epoch': 50, \
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

def make_synthetic_event2(num_event):
    num_data = 5000
    num_feat = 5 #in each segment, total = 15 (5 features x 3 segments)
    
    #construct covariates
    bounds = np.array([-5, -10, 5, 10])
    x_11 = np.random.uniform(bounds[0], bounds[2], size=(num_data//2, num_feat))
    x_12 = np.random.uniform(bounds[0], bounds[2], size=(num_data//2, num_feat))
    x_21 = np.random.uniform(bounds[1], bounds[3], size=(num_data//2, num_feat))
    x_31 = np.random.uniform(bounds[1], bounds[3], size=(num_data//2, num_feat)) 
    x_22 = np.random.uniform(bounds[1], bounds[3], size=(num_data//2, num_feat))
    x_32 = np.random.uniform(bounds[1], bounds[3], size=(num_data//2, num_feat)) 
    
    x1 = np.concatenate((x_11, x_21, x_31), axis=1)
    x2 = np.concatenate((x_12, x_32, x_22), axis=1)
    x = np.concatenate((x1, x2), axis=0)
    
    #construct time to events
    gamma_components = []
    gamma_const = [1, 1, 1]
    for i in range(num_event + 1):
        gamma_components.append(gamma_const[i] * np.ones((num_feat,)))
    gamma_components.append(gamma_const[-1] * np.ones((num_feat,)))

    distr_noise = 0.4 
    distr_noise2 = 0.4 
    
    time2_coeffs = np.array([0, 1, 1]) 
    event_times = [] 
    raw_event_times = []
    raw_event_times2 = []
    for i in range(num_event):
        raw_time = np.power(np.matmul(np.power(np.absolute(x[:, :num_feat]), 1), gamma_components[0]), 2) + \
                   np.power(np.matmul(np.power(np.absolute(x[:, (i + 1)*num_feat:(i+2)*num_feat]), 1), gamma_components[i + 1]), 2)
        raw_event_times.append(raw_time)
        times = np.zeros(raw_time.shape)
        for j in range(raw_time.shape[0]):
            times[j] = np.random.lognormal(mean=np.log(raw_time[j]), sigma=distr_noise)
        event_times.append(times)
        raw_time2 = 1 * (time2_coeffs[2] * np.power(np.matmul(np.absolute(x[:, (0)*num_feat:(1)*num_feat]), gamma_components[2]), 1))
        raw_event_times2.append(raw_time2)
    raw_event_times = np.array(raw_event_times)
    raw_event_times2 = np.array(raw_event_times2)
    print('raw_event_times:', raw_event_times, raw_event_times.shape)
    print('raw_event_times2:', raw_event_times2, raw_event_times2.shape)
    t = np.zeros((num_data, num_event))
    for i in range(num_event):
        t[:, i] = event_times[i]
    labels = np.ones(t.shape)
    
    #time to event for second event (conditional event time)
    t_original = copy.deepcopy(t)
    num_inconsist = 0
    for i in range(num_data):
        if t_original[i, 0] < t_original[i, 1]:
            t[i, 1] = t_original[i, 1] + np.random.lognormal(mean=np.log(raw_event_times2[1][i]), sigma=distr_noise2)
            if t[i, 1] < t_original[i, 0]:
                num_inconsist += 1 
        elif t_original[i, 1] < t_original[i, 0]: 
            t[i, 0] = t_original[i, 0] + np.random.lognormal(mean=np.log(raw_event_times2[1][i]), sigma=distr_noise2)
            if t[i, 0] < t_original[i, 1]:
                num_inconsist += 1  

    #enforce a prediction horizon
    horizon = np.percentile(np.min(t, axis=1), 50) 
    for i in range(t.shape[1]):
        censored = np.where(t[:, i] > horizon)
        t[censored, i] = horizon
        labels[censored, i] = 0
    
    print('label distribution: ', np.unique(labels, return_counts=True, axis=0))
    print('sample number of each event:', np.sum(labels, axis=0))
    print('start time of each event:', np.round(np.min(t, axis=0),
            2))
    print('longest time of each event:', np.round(np.max(t, axis=0), 2))
    
    return x, t, labels


def sample_segment(low, high, size):
    return np.random.uniform(low, high, size=size)

def sample_segment(low, high, size):
    return np.random.uniform(low, high, size=size)

def generate_x_segmentwise_flip(num_data, num_event, num_feat, bound_low=-5, bound_high=5, bound_low2=-10, bound_high2=10):
    assert num_data % 2 == 0, "num_data must be even"
    num_segments = num_event + 1
    half = num_data // 2

    segment_bounds = []
    segment_bounds = [(bound_low, bound_high)] + [(bound_low2, bound_high2)] * (num_segments - 1)
    print(f"segment_bounds: {segment_bounds}")
    segments_front = []
    segments_back = []
    for i in range(num_segments):
        low, high = segment_bounds[i]
        seg_front = sample_segment(low, high, size=(half, num_feat))
        seg_back = sample_segment(low, high, size=(half, num_feat))
        segments_front.append(seg_front)
        segments_back.append(seg_back)

    order_front = list(range(num_segments))
    # here only after first segment would be reverse order
    order_back = [0] + list(reversed(range(1, num_segments)))
    # print(f"order_front: {order_front}, order_back: {order_back}")

    x_front = np.concatenate([segments_front[i] for i in order_front], axis=1)

    x_back = np.concatenate([segments_back[i] for i in order_back], axis=1)

    x = np.concatenate([x_front, x_back], axis=0)
    print(f"Generated synthetic data with shape: {x.shape}, segments: {num_segments}, features per segment: {num_feat}")
    print(f"Segment bounds: {segment_bounds}")
    print(f"Order front: {order_front}, Order back: {order_back}")
    # print(f"First 5 rows of x:\n{x[:5]}")
    # print(f"Last 5 rows of x:\n{x[-5:]}")
    return x

def make_synthetic(config, path):
    import numpy as np
    import pandas as pd
    import copy

    num_event = config["num_events"]
    num_bins = config["num_bins"]
    terminal_events = config["terminal_events"]
    event_ranks = config["event_ranks"]
    event_groups = config["event_groups"]
    min_time = config["min_time"]
    max_time = config["max_time"]
    min_epoch = config["min_epoch"]

    num_data = 5000 * (num_event)
    num_feat = 6
    bounds = np.array([-5, -10, 5, 10])
    num_segments = num_event + 1

    x = generate_x_segmentwise_flip(num_data, num_event, num_feat,
                                     bound_low=bounds[0], bound_high=bounds[2],
                                     bound_low2=bounds[1], bound_high2=bounds[3])

    # generate gamma components
    gamma_components = [np.ones(num_feat) for _ in range(num_segments)]
    distr_noise = 0.4
    distr_noise2 = 0.4
    time2_coeffs = np.array([0, 1, 1])

    event_times = []
    raw_event_times = []
    raw_event_times2 = []

    # sample-wise generation
    for i in range(num_event):
        base_segment = x[:, :num_feat]  # segment 0
        event_segment = x[:, (i + 1)*num_feat:(i + 2)*num_feat]  # segment i+1

        raw_time = np.power(np.dot(np.abs(base_segment), gamma_components[0]), 2) + \
                np.power(np.dot(np.abs(event_segment), gamma_components[i + 1]), 2)

        raw_event_times.append(raw_time)

        # log-normal sampling
        times = np.random.lognormal(mean=np.log(raw_time), sigma=distr_noise)
        event_times.append(times)

        # for rank constraint
        raw_time2 = np.power(np.dot(np.abs(base_segment), gamma_components[0]), 1)
        raw_event_times2.append(raw_time2)
    raw_event_times = np.array(raw_event_times)
    raw_event_times2 = np.array(raw_event_times2)
    print('raw_event_times:', raw_event_times, raw_event_times.shape)
    print('raw_event_times2:', raw_event_times2, raw_event_times2.shape)
    
    t = np.stack(event_times, axis=1) # t's shape is (num_data, num_event)
    
    labels = np.ones(t.shape)

    # apply event_ranks ordering constraints
    t_original = copy.deepcopy(t)
    num_inconsist = 0
    for i in range(num_data):
        # for e1, e2_list in event_ranks.items():
        #     for e2 in e2_list:
        #         e1, e2 = int(e1), int(e2)
        #         if t_original[i, e1] < t_original[i, e2]:
        #             t[i, e2] = t_original[i, e2] + np.random.lognormal(mean=np.log(raw_event_times2[1][i]), sigma=distr_noise2)
        #             if t[i, e2] < t_original[i, e1]:
        #                 num_inconsist += 1
        #         elif t_original[i, e2] < t_original[i, e1]:
        #             t[i, e1] = t_original[i, e1] + np.random.lognormal(mean=np.log(raw_event_times2[1][i]), sigma=distr_noise2)
        #             if t[i, e1] < t_original[i, e2]:
        #                 num_inconsist += 1
        import random
        j, k = random.sample(range(num_event), 2)
        if t_original[i, j] < t_original[i, k]:
            
            t[i, k] = t_original[i, k] + np.random.lognormal(mean=np.log(raw_event_times2[k][i]), sigma=distr_noise2)
            if t[i, k] < t_original[i, j]:
                num_inconsist += 1 
        elif t_original[i, k] < t_original[i, j]: 
            t[i, j] = t_original[i, j] + np.random.lognormal(mean=np.log(raw_event_times2[k][i]), sigma=distr_noise2)
            if t[i, j] < t_original[i, k]:
                num_inconsist += 1  

    # apply right-censoring
    # horizon = np.percentile(np.min(t, axis=1), 50)
    horizon = np.percentile(t, 50)

    # t shape: (num_data, num_event)
    # add noise in each sample's horizon

    horizons = np.random.lognormal(mean=np.log(horizon), sigma=0.01, size=t.shape[0])
    horizons = np.clip(horizons, a_min=horizon*0.5, a_max=horizon*1.5)

    print(f'Enforcing a prediction horizon at {horizon:.2f} days')
    print(f'Horizon range after noise: {horizons.min():.2f} – {horizons.max():.2f}')

    # for i in range(t.shape[1]):
    #     censored = t[:, i] > horizon
    #     t[censored, i] = horizon
    #     labels[censored, i] = 0
    for i in range(t.shape[1]):
        # horizon_i = np.percentile(t[:, i], 50)
        censored = t[:, i] > horizons
        t[censored, i] = horizons[censored]
        labels[censored, i] = 0

    # construct DataFrame
    feature_names = [f'x_{i+1}' for i in range(x.shape[1])]
    duration_names = [f'duration{i+1}' for i in range(num_event)]
    event_names = [f'event{i+1}' for i in range(num_event)]
    df = pd.DataFrame(x, columns=feature_names)

    for i in range(num_event):
        df[duration_names[i]] = t[:, i]
    for i in range(num_event):
        df[event_names[i]] = labels[:, i]

    df.insert(0, 'id', np.arange(1, x.shape[0] + 1))
    filename = f'synthetic_data_event{num_event}.csv'
    filename = path + filename
    df.to_csv(filename, index=False)

    print(f'Saved to {filename}')
    print('Label distribution:\n', np.unique(labels, return_counts=True, axis=0))

    numbers = np.sum(labels, axis=0)
    print('sample number of each event:', numbers)
    print('sample percentage of each event:', numbers / np.sum(numbers))

    start_times = np.min(t, axis=0)
    start_times = np.round(start_times, 2)
    print('start time of each event:', start_times)

    
    longest_times = []
    for i in range(num_event):
        observed_times = t_original[:, i][labels[:, i] == 1]
        longest = np.max(observed_times) if len(observed_times) > 0 else np.nan
        longest_times.append(longest)
    longest_times = np.round(longest_times, 2)
    longest_times = np.round(longest_times, 2)
    print('longest time of each event:', longest_times)
    print('maximum censored times:', np.round(np.max(t, axis=0), 2))
    latex_lines = [
    "\\begin{tabular}{lrrr}",
    "\\toprule",
    "\\textbf{Event} & \\textbf{Observed} & \\textbf{Minimum Time} & \\textbf{Max Time} \\\\",
    "\\midrule"
    ]

    for i in range(num_event):
        name = f"Event {i}"
        observed = int(numbers[i])
        percentage = observed / num_data * 100
        min_time = start_times[i]
        max_time = longest_times[i]
        latex_lines.append(f"{name} & {observed:,} ({percentage:.2f}\%) & {min_time:.2f} & {max_time:.2f} \\\\")

    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")

    latex_table = "\n".join(latex_lines)
    print("\nGenerated LaTeX table:\n")
    print(latex_table)

    # Optional: save to .tex file
    tex_filename = f"{path}event_summary_event{num_event}.tex"
    with open(tex_filename, "w") as f:
        f.write(latex_table)
    print(f"\nLaTeX table saved to {tex_filename}")
    events_per_sample = np.sum(labels, axis=1)  # shape: (num_data,)
    unique_counts, count_freqs = np.unique(events_per_sample, return_counts=True)

    print("Each sample's number of experienced events distribution:")
    for cnt, freq in zip(unique_counts, count_freqs):
        print(f"{cnt} events: {freq} samples ({freq / len(events_per_sample) * 100:.2f}%)")

    avg_events_per_sample = np.mean(events_per_sample)
    print(f"Average events experienced per sample: {avg_events_per_sample:.2f}")
    return x, t, labels




if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser(description='Generate synthetic survival data with competing events.')
    # parser.add_argument('--num_event', type=int, default=2, help='Number of events to simulate')
    # args = parser.parse_args()
    path = "/Users/dingzhu/code/sat/data/hsa-synthetic/"
    make_synthetic(config_event_3, path)

    # make_synthetic_event2(2)
