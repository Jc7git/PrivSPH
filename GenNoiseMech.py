import numpy as np

# 离散高斯噪声生成：根据敏感度、隐私预算和隐私损失生成离散的高斯噪声
def generate_discrete_gaussian_noise(sensitivity, privacy_budget, delta):
    """
    生成一个离散的高斯噪声。
    
    参数：
    sensitivity: 敏感度 Δf，表示函数的最大变化。
    privacy_budget: 隐私预算 ε，控制隐私保护的强度，越小越强。
    privacy_loss: 隐私损失 δ，表示允许的失败概率，通常很小。
    
    返回：
    discrete_noise: 生成的离散高斯噪声（整数值）。
    """
    # 根据给定的敏感度、隐私预算和隐私损失计算高斯机制的标准差 σ
    sigma = (sensitivity / privacy_budget) * np.sqrt(2 * np.log(1.25 / delta))
    
    # 从标准差为σ的正态分布中抽取一个连续的噪声样本
    continuous_noise = np.random.normal(0, sigma)
    
    # 将生成的连续噪声值四舍五入，并转换为整数（离散化）
    discrete_noise = np.round(continuous_noise).astype(int)
    
    # 返回离散化后的噪声
    return discrete_noise

# 计算放大后的隐私预算
def calculate_amplified_privacy_budget(original_privacy_budget, node_size, max_triangle_count):
    """
    计算通过放大机制后的隐私预算 ε'。
    
    参数：
    original_privacy_budget: 原始隐私预算 ε。
    node_size: 数据集节点的数量，影响放大的结果。
    max_triangle_count: 一条边参与三角形数量的最大值。
    
    返回：
    amplified_privacy_budget: 放大后的隐私预算 ε'。
    """
    # 计算通过放大得到的概率 p，影响隐私预算的放大效果
    amplification_probability = ((2 * max_triangle_count + 1) * (node_size - 3) - max_triangle_count * (max_triangle_count - 1)) / ((node_size - 1) * (node_size - 3))
    
    # 使用放大的概率 p 来计算放大后的隐私预算 ε'
    amplified_privacy_budget = np.log((np.exp(original_privacy_budget) - 1) / amplification_probability + 1)
    
    # 返回放大后的隐私预算
    return amplified_privacy_budget

# 结合放大后的隐私预算计算离散的高斯噪声
def generate_discrete_gaussian_after_amplification(sensitivity, original_privacy_budget, delta, node_size, max_triangle_count,noAm = 0):
    """
    结合放大后的隐私预算计算离散的高斯噪声。
    
    参数：
    sensitivity: 敏感度 Δf，表示函数的最大变化。
    original_privacy_budget: 原始隐私预算 ε。
    privacy_loss: 隐私损失 δ。
    node_size: 数据集节点的数量，影响放大的结果。
    max_triangle_count: 一条边参与三角形数量的最大值。
    
    返回：
    discrete_noise: 生成的离散高斯噪声（整数值）。
    """
    # 计算放大后的隐私预算 ε'
    if noAm == 0:
        amplified_privacy_budget = calculate_amplified_privacy_budget(original_privacy_budget, node_size, max_triangle_count)
    else:
        amplified_privacy_budget = original_privacy_budget
    
    # 使用放大后的隐私预算 ε' 来生成离散的高斯噪声
    discrete_noise = generate_discrete_gaussian_noise(sensitivity, amplified_privacy_budget, delta)
    
    # 返回离散化后的噪声
    return discrete_noise

def generate_gaussian_vector_after_amplification(sensitivity, original_privacy_budget, delta, node_size, max_triangle_count,noAm = 0):
    """
    结合放大后的隐私预算计算离散的高斯噪声。
    
    参数：
    sensitivity: 敏感度 Δf，表示函数的最大变化。
    original_privacy_budget: 原始隐私预算 ε。
    privacy_loss: 隐私损失 δ。
    node_size: 数据集节点的数量，影响放大的结果。
    max_triangle_count: 一条边参与三角形数量的最大值。
    
    返回：
    discrete_noise: 生成的离散高斯噪声（整数值）。
    """
    # 计算放大后的隐私预算 ε'
    if noAm == 0:
        amplified_privacy_budget = calculate_amplified_privacy_budget(original_privacy_budget, node_size, max_triangle_count)
    else:
        amplified_privacy_budget = original_privacy_budget

    # 使用放大后的隐私预算 ε' 来生成离散的高斯噪声
    sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / amplified_privacy_budget
    
    # 向每个元素添加独立的高斯噪声
    noise = np.random.normal(0, sigma, max_triangle_count+1)
    
    return noise

def generate_laplace_vector_after_amplification(sensitivity, original_privacy_budget, node_size, max_triangle_count,noAm = 0):
    """
    结合放大后的隐私预算计算离散的高斯噪声。
    
    参数：
    sensitivity: 敏感度 Δf，表示函数的最大变化。
    original_privacy_budget: 原始隐私预算 ε。
    privacy_loss: 隐私损失 δ。
    node_size: 数据集节点的数量，影响放大的结果。
    max_triangle_count: 一条边参与三角形数量的最大值。
    
    返回：
    discrete_noise: 生成的离散高斯噪声（整数值）。
    """
    # 计算放大后的隐私预算 ε'
    if noAm == 0:
        amplified_privacy_budget = calculate_amplified_privacy_budget(original_privacy_budget, node_size, max_triangle_count)
    else:
        amplified_privacy_budget = original_privacy_budget
    
    # 使用放大后的隐私预算 ε' 
    scale = sensitivity / amplified_privacy_budget
    
    # 向每个元素添加独立的拉普拉斯噪声
    noise = np.random.laplace(0, scale,max_triangle_count+1)
    
    # return np.round(noise)
    return noise

def discrete_laplace_mechanism(epsilon, sensitivity):
    """
    根据隐私预算 (ε) 和敏感度 (Δf)，给定真实值 true_value，生成离散拉普拉斯噪声，并返回带噪声的查询结果。

    参数:
    epsilon -- 隐私预算 (ε)
    delta_f -- 敏感度 (Δf)
    true_value -- 真实值（查询的原始结果）

    返回:
    含噪声的查询结果
    """
    # 计算拉普拉斯分布的尺度
    scale = sensitivity / epsilon
    
    # 生成拉普拉斯噪声
    noise = np.random.laplace(0, scale)
    
    # 离散化噪声为整数
    discrete_noise = int(round(noise))
    
    # 返回带噪声的查询结果
    return discrete_noise

def generate_discrete_laplace_after_amplification(sensitivity, original_privacy_budget, node_size, max_triangle_count,noAm = 0):
    """
    结合放大后的隐私预算计算离散的高斯噪声。
    
    参数：
    sensitivity: 敏感度 Δf，表示函数的最大变化。
    original_privacy_budget: 原始隐私预算 ε。
    privacy_loss: 隐私损失 δ。
    node_size: 数据集节点的数量，影响放大的结果。
    max_triangle_count: 一条边参与三角形数量的最大值。
    
    返回：
    discrete_noise: 生成的离散高斯噪声（整数值）。
    """
    # 计算放大后的隐私预算 ε'
    if noAm == 0:
        amplified_privacy_budget = calculate_amplified_privacy_budget(original_privacy_budget, node_size, max_triangle_count)
    else:
        amplified_privacy_budget = original_privacy_budget
    
    # 使用放大后的隐私预算 ε' 来生成离散的高斯噪声
    discrete_noise = discrete_laplace_mechanism(amplified_privacy_budget,sensitivity)
    
    # 返回离散化后的噪声
    return discrete_noise

