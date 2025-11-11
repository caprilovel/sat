import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

def make_signal_survival_synthetic_(config, path="./", padding=True):
    """
    Generate synthetic survival data with:
      - Latent risk factor driving hazard
      - Event imbalance
      - Correlation between events
      - Controlled noise for C-index > 0.5
    """
    num_event = config["num_events"]
    min_time = config.get("min_time", 100)
    max_time = config.get("max_time", 2000)
    num_samples = config.get("num_samples", 5000)

    np.random.seed(42)

    # Latent patient risk factor
    z = np.random.normal(0, 1, size=num_samples)  # per patient

    # Event-specific coefficients (correlation to z strength)
    betas = np.random.uniform(0.5, 2.0, size=num_event)  # >0 means risk ↑ with z
    baseline_scale = np.random.uniform(500, 1000, size=num_event)  # baseline scale per event

    # Event imbalance
    event_prevalence = np.random.dirichlet(np.ones(num_event))

    t = np.zeros((num_samples, num_event))
    labels = np.zeros_like(t)

    for i in range(num_event):
        # Exponential-like survival time: smaller hazard → longer time
        hazard = np.exp(-betas[i] * z)  # risk factor effect
        raw_time = baseline_scale[i] * hazard

        # Add controlled noise (not too large, keeps signal intact)
        raw_time += np.random.normal(0, baseline_scale[i] * 0.1, size=num_samples)

        # Ensure positive and within [min_time, max_time]
        raw_time = np.clip(raw_time, min_time, max_time)
        t[:, i] = raw_time

        # Event occurrence (imbalance)
        event_occurs = np.random.binomial(1, event_prevalence[i], size=num_samples)
        labels[:, i] = event_occurs

        # Apply censoring (mild)
        horizon = np.percentile(raw_time, np.random.uniform(85, 90))
        censored = (raw_time > horizon) & (event_occurs == 1)
        t[censored, i] = horizon
        labels[censored, i] = 0

    # Features = latent risk factor + padding
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
    df.insert(0, "id", np.arange(1, num_samples + 1))

    filename = f"signal_survival_data_event{num_event}.csv"
    df.to_csv(path + filename, index=False)
    print(f"Saved to {path}{filename}")

    for i in range(num_event):
        print(f"event {i+1}:")
        print(f"  - Censored: {np.sum(labels[:, i] == 0)}")
        print(f"  - Observed: {np.sum(labels[:, i] == 1)}")

    return x, t, labels


def make_signal_survival_synthetic__(config, path="./", padding=True, censor_q=(0.9, 0.95)):
    """
    Generate synthetic survival data with:
      - Latent risk factor driving hazard
      - Correlation between events
      - Controlled administrative censoring
      - No artificial "never-happen" events (方案A)
    """
    num_event = config["num_events"]
    min_time = config.get("min_time", 100)
    max_time = config.get("max_time", 2000)
    num_samples = config.get("num_samples", 5000)

    np.random.seed(42)

    # Latent patient risk factor
    z = np.random.normal(0, 1, size=num_samples)

    # Event-specific coefficients (risk loading on z)
    betas = np.random.uniform(0.5, 2.0, size=num_event)
    baseline_scale = np.random.uniform(500, 1000, size=num_event)
    shape = np.random.uniform(1.2, 2.0, size=num_event)  # Weibull shape per event

    # True survival times
    t_true = np.zeros((num_samples, num_event))
    for i in range(num_event):
        u = np.random.uniform(0, 1, size=num_samples)
        scale_i = baseline_scale[i] * np.exp(-betas[i] * z)
        t = scale_i * (-np.log(u)) ** (1.0 / shape[i])
        t_true[:, i] = np.clip(t, min_time, max_time)

    # Administrative censoring: per-sample horizon
    pooled = t_true.flatten()
    q = np.random.uniform(censor_q[0], censor_q[1], size=num_samples)  # 每个样本一个分位点
    C_values = np.quantile(pooled, q)  # shape (num_samples,)

    # Observed time & censor indicator
    observed_time = np.minimum(t_true, C_values[:, None])
    observed_flag = (t_true <= C_values[:, None]).astype(int)

    # Features
    x = z[:, None]
    if padding and x.shape[1] < 30:
        pad = np.random.normal(0, 1, size=(num_samples, 30 - x.shape[1]))
        x = np.hstack([x, pad])

    # Build DataFrame
    feature_names = [f"x_{i+1}" for i in range(x.shape[1])]
    df = pd.DataFrame(x, columns=feature_names)
    for i in range(num_event):
        df[f"duration{i+1}"] = observed_time[:, i]
        df[f"event{i+1}"] = observed_flag[:, i]
    df.insert(0, "id", np.arange(1, num_samples + 1))

    filename = f"signal_survival_data_event{num_event}.csv"
    df.to_csv(path + filename, index=False)
    print(f"Saved to {path}{filename}")

    # Summary stats
    for i in range(num_event):
        censored = np.sum(observed_flag[:, i] == 0)
        observed = np.sum(observed_flag[:, i] == 1)
        print(f"event {i+1}:")
        print(f"  - Censored: {censored}")
        print(f"  - Observed: {observed}")

    return x, observed_time, observed_flag, t_true

import numpy as np
import pandas as pd

def make_signal_survival_synthetic(config, path="./", padding=True, censor_q=(0.4, 0.5)):
    """
    Generate synthetic survival data with:
      - Latent risk factor driving hazard
      - Correlation between events
      - Administrative censoring (方案A, guaranteed censor)
    """
    num_event = config["num_events"]
    min_time = config.get("min_time", 100)
    max_time = config.get("max_time", 2000)
    num_samples = config.get("num_samples", 5000)

    np.random.seed(42)

    # Latent patient risk factor
    z = np.random.normal(0, 1, size=num_samples)

    # Event-specific coefficients (risk loading on z)
    betas = np.random.uniform(0.5, 2.0, size=num_event)
    baseline_scale = np.random.uniform(500, 1000, size=num_event)
    shape = np.random.uniform(1.2, 2.0, size=num_event)  # Weibull shape per event

    # True survival times
    t_true = np.zeros((num_samples, num_event))
    for i in range(num_event):
        u = np.random.uniform(0, 1, size=num_samples)
        scale_i = baseline_scale[i] * np.exp(-betas[i] * z)
        t = scale_i * (-np.log(u)) ** (1.0 / shape[i])
        t_true[:, i] = np.clip(t, min_time, max_time)

    # Administrative censoring: choose a global threshold for each sample
    pooled = t_true.flatten()
    q_low, q_high = censor_q
    # 每个样本 horizon 在 [q_low, q_high] 分位数之间
    C_low, C_high = np.quantile(pooled, [q_low, q_high])
    C_values = np.random.uniform(C_low, C_high, size=num_samples)

    # Observed time & censor indicator
    observed_time = np.minimum(t_true, C_values[:, None])
    observed_flag = (t_true <= C_values[:, None]).astype(int)

    # Features
    x = z[:, None]
    if padding and x.shape[1] < 30:
        pad = np.random.normal(0, 1, size=(num_samples, 30 - x.shape[1]))
        x = np.hstack([x, pad])

    # Build DataFrame
    feature_names = [f"x_{i+1}" for i in range(x.shape[1])]
    df = pd.DataFrame(x, columns=feature_names)
    for i in range(num_event):
        df[f"duration{i+1}"] = observed_time[:, i]
    for i in range(num_event):
        df[f"event{i+1}"] = observed_flag[:, i]
    df.insert(0, "id", np.arange(1, num_samples + 1))

    filename = f"signal_survival_data_event{num_event}.csv"
    df.to_csv(path + filename, index=False)
    print(f"Saved to {path}{filename}")

    # Summary stats
    for i in range(num_event):
        censored = np.sum(observed_flag[:, i] == 0)
        observed = np.sum(observed_flag[:, i] == 1)
        print(f"event {i+1}: censored={censored} percentage={censored/num_samples*100:.1f}%, observed={observed} percentage={observed/num_samples*100:.1f}%")
        print(f"  - True time range: {t_true[:, i].min():.1f} to {t_true[:, i].max():.1f}"
              f", Observed time range: {observed_time[:, i].min():.1f} to {observed_time[:, i].max():.1f}")

    return x, observed_time, observed_flag, t_true


def make_coxph_survival_synthetic(config, path="./", padding=True):
    """
    Generate synthetic survival data with Cox PH model:
      - Latent risk factor z
      - Weibull-distributed times
      - Event imbalance
      - Mild censoring
    """
    num_event = config["num_events"]
    min_time = config.get("min_time", 100)
    max_time = config.get("max_time", 2000)
    num_samples = config.get("num_samples", 5000)

    np.random.seed(42)

    # Latent risk factor per patient
    z = np.random.normal(0, 1, size=num_samples)

    # Event-specific parameters
    baseline_scale = np.random.uniform(500, 1000, size=num_event)
    shape = np.random.uniform(1.2, 2.0, size=num_event)   # Weibull shape
    betas = np.random.uniform(0.5, 1.5, size=num_event)   # Cox coefficients
    event_prevalence = np.random.dirichlet(np.ones(num_event))  # imbalance

    t = np.zeros((num_samples, num_event))
    labels = np.zeros_like(t)

    for i in range(num_event):
        # Weibull survival time with PH structure
        u = np.random.uniform(0, 1, size=num_samples)
        scale_i = baseline_scale[i] * np.exp(-betas[i] * z)  # Cox PH scaling
        event_time = scale_i * (-np.log(u)) ** (1.0 / shape[i])

        # Clip into [min_time, max_time]
        event_time = np.clip(event_time, min_time, max_time)
        t[:, i] = event_time

        # Event occurrence (imbalance)
        event_occurs = np.random.binomial(1, event_prevalence[i], size=num_samples)
        labels[:, i] = event_occurs

        # Apply mild censoring
        horizon = np.percentile(event_time, np.random.uniform(70, 85))
        censored = (event_time > horizon) & (event_occurs == 1)
        t[censored, i] = horizon
        labels[censored, i] = 0

    # Features = latent + noise
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
    df.insert(0, "id", np.arange(1, num_samples + 1))

    filename = f"coxph_survival_data_event{num_event}.csv"
    df.to_csv(path + filename, index=False)
    print(f"Saved to {path}{filename}")

    return x, t, labels

def make_signal_survival_synthetic_multi(config, path="./", padding=True, censor_q=(0.4, 0.5)):
    """
    Generate synthetic survival data with more meaningful features
    """
    num_event = config["num_events"]
    min_time = config.get("min_time", 100)
    max_time = config.get("max_time", 2000)
    num_samples = config.get("num_samples", 5000)
    
    np.random.seed(42)
    
    # 1. 主要潜在风险因子 (overall health)
    z_main = np.random.normal(0, 1, size=num_samples)
    
    # 2. 事件特异性风险因子
    z_specific = np.random.normal(0, 0.5, size=(num_samples, num_event))
    
    # 3. 临床相关特征
    age = np.random.normal(65, 15, size=num_samples)  # 年龄
    age = np.clip(age, 18, 95)
    
    gender = np.random.binomial(1, 0.5, size=num_samples)  # 性别
    
    bmi = np.random.normal(25, 5, size=num_samples)  # BMI
    bmi = np.clip(bmi, 15, 45)
    
    smoking = np.random.binomial(1, 0.3, size=num_samples)  # 吸烟史
    
    # 4. 实验室指标 (与主风险因子相关)
    cholesterol = 200 + 30 * z_main + np.random.normal(0, 20, num_samples)
    blood_pressure = 120 + 15 * z_main + 5 * age/65 + np.random.normal(0, 10, num_samples)
    glucose = 90 + 20 * z_main + np.random.normal(0, 15, num_samples)
    
    # 5. 合并症指标
    diabetes = np.random.binomial(1, 0.15 + 0.1 * (z_main > 0), size=num_samples)
    hypertension = np.random.binomial(1, 0.25 + 0.15 * (z_main > 0), size=num_samples)
    
    # 构建特征矩阵
    features = np.column_stack([
        z_main,           # x_1: 主风险因子
        age/100,          # x_2: 标准化年龄
        gender,           # x_3: 性别
        bmi/40,           # x_4: 标准化BMI
        smoking,          # x_5: 吸烟
        cholesterol/300,  # x_6: 标准化胆固醇
        blood_pressure/200, # x_7: 标准化血压
        glucose/200,      # x_8: 标准化血糖
        diabetes,         # x_9: 糖尿病
        hypertension,     # x_10: 高血压
        z_specific        # x_11 到 x_(10+num_event): 事件特异性风险因子
    ])
    
    # 事件特异性系数
    betas = np.random.uniform(0.5, 2.0, size=(num_event, features.shape[1]))
    baseline_scale = np.random.uniform(500, 1000, size=num_event)
    shape = np.random.uniform(1.2, 2.0, size=num_event)
    
    # 生成真实生存时间
    t_true = np.zeros((num_samples, num_event))
    for i in range(num_event):
        u = np.random.uniform(0, 1, size=num_samples)
        linear_pred = np.dot(features, betas[i])
        scale_i = baseline_scale[i] * np.exp(-linear_pred)
        t = scale_i * (-np.log(u)) ** (1.0 / shape[i])
        t_true[:, i] = np.clip(t, min_time, max_time)
    
    # 如果需要padding到30维
    if padding and features.shape[1] < 30:
        remaining = 30 - features.shape[1]
        # 只添加少量噪声特征
        pad = np.random.normal(0, 0.1, size=(num_samples, remaining))
        features = np.hstack([features, pad])
    
    # 行政审查
    pooled = t_true.flatten()
    q_low, q_high = censor_q
    C_low, C_high = np.quantile(pooled, [q_low, q_high])
    C_values = np.random.uniform(C_low, C_high, size=num_samples)
    
    observed_time = np.minimum(t_true, C_values[:, None])
    observed_flag = (t_true <= C_values[:, None]).astype(int)
    
    # 构建DataFrame - 使用统一的特征命名
    feature_names = [f"x_{i+1}" for i in range(features.shape[1])]
    df = pd.DataFrame(features, columns=feature_names)
    
    # 添加生存数据
    for i in range(num_event):
        df[f"duration{i+1}"] = observed_time[:, i]
    for i in range(num_event):
        df[f"event{i+1}"] = observed_flag[:, i]
    df.insert(0, "id", np.arange(1, num_samples + 1))
    
    filename = f"signal_survival_data_event{num_event}.csv"
    df.to_csv(path + filename, index=False)
    print(f"Saved to {path}{filename}")
    
    # 统计信息 - 显示特征含义
    meaningful_features = 10 + num_event  # 基础特征 + 事件特异性特征
    noise_features = features.shape[1] - meaningful_features
    
    print(f"Feature composition:")
    print(f"- x_1: overall_risk (main latent factor)")
    print(f"- x_2: age_normalized")
    print(f"- x_3: gender")
    print(f"- x_4: bmi_normalized") 
    print(f"- x_5: smoking")
    print(f"- x_6: cholesterol_normalized")
    print(f"- x_7: blood_pressure_normalized")
    print(f"- x_8: glucose_normalized")
    print(f"- x_9: diabetes")
    print(f"- x_10: hypertension")
    for i in range(num_event):
        print(f"- x_{11+i}: event{i+1}_specific_risk")
    if noise_features > 0:
        print(f"- x_{meaningful_features+1} to x_{features.shape[1]}: noise features")
    
    print(f"\nTotal: {meaningful_features} meaningful + {noise_features} noise features")
    
    for i in range(num_event):
        censored = np.sum(observed_flag[:, i] == 0)
        observed = np.sum(observed_flag[:, i] == 1)
        print(f"event {i+1}: censored={censored} percentage:{censored/num_samples*100:.1f}%, observed={observed} percentage:{observed/num_samples*100:.1f}%")
        print(f"  - True time range: {t_true[:, i].min():.1f} to {t_true[:, i].max():.1f}"
              f", Observed time range: {observed_time[:, i].min():.1f} to {observed_time[:, i].max():.1f}")
        
    
    return features, observed_time, observed_flag, t_true

if __name__ == "__main__":
    config = {"num_events": 3, "min_time": 100, "max_time": 2000, "num_samples": 15000, }
    # make_realistic_survival_synthetic(config)
    make_signal_survival_synthetic(config, path='/Users/dingzhu/code/sat/data/hsa-synthetic/', padding=False)
    # make_signal_survival_synthetic_multi(config, path='/Users/dingzhu/code/sat/data/hsa-synthetic/', padding=False)
    # make_coxph_survival_synthetic(config)