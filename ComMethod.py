import math
import random
import DataProcess as dp
import numpy as np
from collections import defaultdict


def calc_loc_tri_arr(a_mat, node_num, eps_t, alg_prm=0.01):
    """
    在本地模型中使用非对称随机响应(ARR)计算三角形数量
    Args:
        a_mat (dict): 原始邻接列表，格式为 {节点i: [邻居节点列表]}
        node_num (int): 总节点数
        eps_t (float): 隐私预算ε
        alg_prm (float): 算法参数，用于调整采样率
    Returns:
        float: 估计的三角形数量
    """
    # === 初始化噪声邻接列表 ===
    # 使用字典存储噪声后的邻接关系，格式为 {节点i: [邻居节点列表]}
    a_mat_ns = defaultdict(list)

    # === 参数计算 ===
    # 计算随机响应概率p1
    p1 = math.exp(eps_t) / (math.exp(eps_t) + 1.0)
    # 计算采样率p2：p2^3 = (1/n)^(1/3) * 采样权重   为什么要加一个采样率呢？
    p2 = math.pow(1.0 / node_num, 1.0 / 3) * alg_prm

    # 计算ARR参数
    mu = p1 * p2          # 保持边存在的概率
    murho = mu / math.exp(eps_t)  # 翻转边不存在的概率

    # === 生成噪声邻接列表 ===
    for i in range(node_num):
        for j in range(i + 1, node_num):
            rnd = random.random()
            # 原始边不存在的情况（翻转添加边）
            if rnd < murho and j not in a_mat[i]:
                a_mat_ns[i].append(j)
                a_mat_ns[j].append(i)
            # 原始边存在的情况（保持边）
            elif rnd < mu and j in a_mat[i]:
                a_mat_ns[i].append(j)
                a_mat_ns[j].append(i)

    # === 计算噪声度数 ===
    deg_ns = [0] * node_num
    for i in range(node_num):
        deg_ns[i] = len(a_mat_ns[i])

    # === 计算总边数 ===
    tot_edge_num_ns = sum(deg_ns) // 2

    # === 统计噪声三角形数量 ===
    tri_num = 0
    for i in a_mat_ns:
        neighbors = a_mat_ns[i]
        for idx_j, j in enumerate(neighbors):
            if j <= i: continue  # 避免重复计算
            for k in neighbors[idx_j + 1:]:
                if k <= j: continue
                if k in a_mat_ns[j]:
                    tri_num += 1

    # === 计算二阶星数量 ===
    st2_num = 0
    for d in deg_ns:
        st2_num += d * (d - 1) // 2

    # === 估计原始图参数 ===
    ed2_num = st2_num - 3 * tri_num  # 二阶边数量
    ed1_num = tot_edge_num_ns * (node_num - 2) - 2 * ed2_num - 3 * tri_num  # 一阶边数量

    # 逆向计算采样前的统计量
    q2 = 1.0 - p2
    tri_num_bs = tri_num / (p2 ** 3)  # 原始三角形数估计
    ed2_num_bs = ed2_num / (p2 ** 2) - 3 * q2 * tri_num_bs
    ed1_num_bs = ed1_num / p2 - 2 * q2 * ed2_num_bs - 3 * (q2 ** 2) * tri_num_bs

    # 计算无效边组合数
    non_num_bs = node_num * (node_num - 1) * (node_num - 2) // 6 - tri_num_bs - ed2_num_bs - ed1_num_bs

    # === 隐私逆变换计算 ===
    alp = math.exp(eps_t)
    alp_1_3 = (alp - 1) ** 3
    # 计算四个逆变换系数
    q_inv = [
        (alp ** 3) / alp_1_3,  # q_inv_11
        -(alp ** 2) / alp_1_3,  # q_inv_21
        alp / alp_1_3,  # q_inv_31
        -1 / alp_1_3  # q_inv_41
    ]

    # 最终三角形数量估计
    tri_num_ns = (tri_num_bs * q_inv[0] +
                 ed2_num_bs * q_inv[1] +
                 ed1_num_bs * q_inv[2] +
                 non_num_bs * q_inv[3])

    return tri_num_ns

def read_numerical_bound(n, eps, delta_s):
    # 初始化变量
    epsl = 0.0
    eps_tmp = 0.0

    # 构建文件路径
    path = "./data/IMDB/"  # 需要预先计算出来

    infile = f"{path}numerical-bound_n{n}_d10-{delta_s}.csv"

    # 打开文件
    try:
        with open(infile, "r") as fp:
            # 跳过标题行
            next(fp)
            # 逐行读取文件
            for line in fp:
                # 分割每一行的数据
                tokens = line.strip().split(" ")
                # 提取 epsL 和 eps_upper
                epsl = float(tokens[0])
                eps_tmp = float(tokens[2])
                # 如果满足条件，则退出循环
                if eps_tmp <= eps:
                    break
    except FileNotFoundError:
        print(f"Cannot open {infile}")
        exit(-1)

    # 返回满足条件的 epsl
    return epsl

def calc_eps_l(n, eps, delta = 8):
    # 初始化变量
    epsl_min = 0.001
    epsl_max = 100.0
    epsl_sht = 0.001
    epsl = 0.0
    alpha = 0.0
    x1, x2, x3 = 0.0, 0.0, 0.0
    eps_tmp = 0.0

    # 计算 epsl_max
    epsl_max = math.log(n / (16.0 * math.log(2.0 / math.pow(10, -delta))))

    # 迭代搜索合适的 epsl
    epsl = epsl_max
    while epsl >= epsl_min:
        alpha = math.exp(epsl)
        x1 = (alpha - 1.0) / (alpha + 1.0)
        x2 = 8.0 * math.sqrt(alpha * math.log(4.0 / math.pow(10, -delta))) / math.sqrt(n)
        x3 = 8.0 * alpha / n
        eps_tmp = math.log(1.0 + x1 * (x2 + x3))
        
        # 如果找到满足条件的 epsl，则退出循环
        if eps_tmp <= eps:
            break
        
        # 减小 epsl
        epsl -= epsl_sht

    # 边界检查
    if epsl > epsl_max:
        epsl = epsl_max

    # 返回计算得到的 epsl
    return epsl

def calc_shuffle_tri(a_mat, deg, alg, node_num, eps_t, eps_l, eps_d, alg_prm=1, pair_num=-1):
    """
    在shuffle/local模型下基于差分隐私计算三角形数量
    Args:
        a_mat (dict): 邻接表 {节点: {邻居集合}}
        deg (list): 节点度数列表
        alg (int): 算法类型 (2: shuffle wedge, 3: shuffle+degree threshold, 4: shuffle+noisy degree threshold, 5: local)
        node_num (int): 总节点数
        eps_t (float): 隐私预算
        eps_l (float): shuffle放大后的隐私预算，和上一个对应
        eps_d (float): 度估计隐私预算（仅alg=4时使用）
        alg_prm (float): 算法参数 c
        pair_num (int): 处理的节点对数（-1表示全部）
    Returns:
        float: 估计的三角形数量
    """
    # ====== 初始化部分 ======
    tri_num_ns = 0.0  # 噪声三角形数量估计值
    deg_ns = np.zeros(node_num) if alg ==4 else None  # 噪声度数
    node_order = list(range(node_num))  # 节点顺序

    # ====== 算法参数设置 ======
    # 计算度数阈值（仅alg3/4需要）
    deg_thr = 0.0
    if alg ==3:
        # 计算平均度数
        avg_deg = sum(deg) / node_num
        deg_thr = avg_deg * alg_prm
    elif alg ==4:
        # 添加拉普拉斯噪声到度数
        for i in range(node_num):
            deg_ns[i] = deg[i] + np.random.laplace(0, 1/eps_d)
        avg_deg_ns = np.mean(deg_ns)
        deg_thr = avg_deg_ns * alg_prm

    # 设置wedge翻转概率
    if alg in {2,3,4}:  # Shuffle模型
        q1w = 1/(math.exp(eps_l)+1)
    elif alg ==5:       # Local模型
        q1w = 1/(math.exp(eps_t)+1)
    p1w = 1 - q1w

    # 设置边翻转概率
    q2 = 1/(math.exp(eps_t)+1)
    p2 = 1 - q2

    # ====== 生成随机节点排列 ======
    random.shuffle(node_order) 

    # ====== 处理节点对 ======
    processed_pairs = 0
    for i in range(0, node_num, 2):
        # 获取节点对
        if i+1 >= node_num: break
        node1 = node_order[i]
        node2 = node_order[i+1]

        # ====== 边噪声处理 ======
        # 生成node1->node2的噪声边
        edge12_ns = 0
        rand1 = random.random()
        if rand1 < q2 and node2 not in a_mat[node1]:
            edge12_ns = 1
        elif rand1 >= q2 and node2 in a_mat[node1]:
            edge12_ns = 1

        # 生成node2->node1的噪声边（对称处理）
        edge21_ns = 0
        rand2 = random.random()
        if rand2 < q2 and node1 not in a_mat[node2]:
            edge21_ns = 1
        elif rand2 >= q2 and node1 in a_mat[node2]:
            edge21_ns = 1

        # ====== 构建邻接向量 ======
        # 节点1的邻接向量（存在边为1，否则0）
        a_mat_node1 = [0]*node_num
        for neighbor in a_mat[node1]:
            a_mat_node1[neighbor] = 1
        
        # 节点2的邻接向量
        a_mat_node2 = [0]*node_num
        for neighbor in a_mat[node2]:
            a_mat_node2[neighbor] = 1

        # ====== 楔形(wedge)统计 ======
        wedge_num = 0
        for j in range(node_num):
            if j == node1 or j == node2: continue
            
            # 原始wedge存在性
            wedge = 1 if a_mat_node1[j] and a_mat_node2[j] else 0
            
            # 添加wedge噪声
            rand3 = random.random()
            if rand3 < q1w and wedge ==0:
                wedge_ns = 1
            elif rand3 >= q1w and wedge ==1:
                wedge_ns = 1
            else:
                wedge_ns = 0
            
            if wedge_ns:
                wedge_num +=1

        # ====== 无偏估计计算 ======
        numerator1 = (edge12_ns + edge21_ns) - 2*q2
        numerator2 = wedge_num - (node_num-2)*q1w
        denominator = 2*(2*p2-1)*(2*p1w-1)
        c1 = numerator1 * numerator2 / denominator

        # ====== 根据算法类型累加估计值 ======
        if alg in {2,5}:  # 基本shuffle/local模型
            tri_num_ns += c1
            processed_pairs +=1
        elif alg ==3:     # 度数阈值过滤
            if deg[node1] > deg_thr and deg[node2] > deg_thr:
                tri_num_ns += c1
                processed_pairs +=1
        elif alg ==4:     # 噪声度数阈值过滤
            if deg_ns[node1] > deg_thr and deg_ns[node2] > deg_thr:
                tri_num_ns += c1
                processed_pairs +=1

        # 处理对数控制
       
        if pair_num != -1 and processed_pairs >= pair_num:
            break

    # ====== 最终校正 ======
    tri_num_ns *= node_num*(node_num-1)/(6*processed_pairs)
    
    return tri_num_ns

def calc_shuffle_cy4(adj_list: list, node_num: int,alg: int, eps_t: float, eps_l: float, pair_num: int = -1) -> float:
    """
    在Shuffle/Local模型下基于差分隐私计算四环数量
    Args:
        adj_list: 邻接表，列表元素为集合，索引代表节点ID
        node_num: 节点总数
        alg: 算法类型 (8: shuffle, 9: local)
        eps_t: 未shuffle时的隐私预算
        eps_l: shuffle后放大的隐私预算
        pair_num: 处理的节点对数（-1表示全部）
    Returns:
        估计的四环数量
    """

    # ====== 初始化 ======
    cy4_num_ns = 0.0
    node_order = list(range(node_num))
    random.shuffle(node_order)  # 随机节点排列

    # ====== 隐私参数设置 ======
    q1w = 1/(math.exp(eps_l)+1) if alg ==8 else 1/(math.exp(eps_t)+1)
    p1w = 1 - q1w

    # ====== 处理节点对 ======
    processed_pairs = 0
    for i in range(0, node_num, 2):
        # 终止条件检查
        if pair_num != -1 and processed_pairs >= pair_num:
            break
        if i+1 >= node_num:
            break

        # 获取当前节点对
        node1 = node_order[i]
        node2 = node_order[i+1]

        # ====== 构建邻接向量 ======
        # 将集合转换为二进制向量（优化内存访问）
        vec1 = [1 if j in adj_list[node1] else 0 for j in range(node_num)]
        vec2 = [1 if j in adj_list[node2] else 0 for j in range(node_num)]

        # ====== Wedge统计 ======
        c1 = 0.0
        for j in range(node_num):
            if j == node1 or j == node2:
                continue
                
            # 原始wedge存在性检查
            wedge = 1 if vec1[j] and vec2[j] else 0
            
            # 差分隐私扰动
            rnd = random.random()
            wedge_ns = 0
            if wedge == 0 and rnd < q1w:    # 翻转0->1
                wedge_ns = 1
            elif wedge == 1 and rnd >= q1w: # 保持1->1
                wedge_ns = 1

            # 无偏估计量累计
            c1 += (wedge_ns - (1 - p1w)) / (2*p1w - 1)

        # ====== 四环估计计算 ======
        term1 = c1 * (c1 - 1) / 2
        term2 = (node_num-2)*p1w*(1-p1w) / (2*(2*p1w-1)**2)
        cy4_num_ns += (term1 - term2)
        processed_pairs += 1

    # ====== 最终结果校正 ======
    return cy4_num_ns * node_num*(node_num-1) / (4*processed_pairs)


def calc_shuffle_tri2(adj_list: list, node_num: int,alg: int, eps_t: float, eps_l: float, pair_num: int = -1) -> float:
    """
    在Shuffle/Local模型下基于差分隐私计算四环数量
    Args:
        adj_list: 邻接表，列表元素为集合，索引代表节点ID
        node_num: 节点总数
        alg: 算法类型 (8: shuffle, 9: local)
        eps_t: 未shuffle时的隐私预算
        eps_l: shuffle后放大的隐私预算
        pair_num: 处理的节点对数（-1表示全部）
    Returns:
        估计的四环数量
    """

    # ====== 初始化 ======
    tri2_num_ns = 0.0
    node_order = list(range(node_num))
    random.shuffle(node_order)  # 随机节点排列

    # ====== 隐私参数设置 ======
    q1w = 1/(math.exp(eps_l)+1) if alg ==8 else 1/(math.exp(eps_t)+1)
    p1w = 1 - q1w

      # 设置边翻转概率
    q2 = 1/(math.exp(eps_t)+1)
    p2 = 1 - q2

    # ====== 处理节点对 ======
    processed_pairs = 0
    for i in range(0, node_num, 2):
        # 终止条件检查
        if pair_num != -1 and processed_pairs >= pair_num:
            break
        if i+1 >= node_num:
            break

        # 获取当前节点对
        node1 = node_order[i]
        node2 = node_order[i+1]

         # ====== 边噪声处理 ======
        # 生成node1->node2的噪声边
        edge12_ns = 0
        rand1 = random.random()
        if rand1 < q2 and node2 not in adj_list[node1]:
            edge12_ns = 1
        elif rand1 >= q2 and node2 in adj_list[node1]:
            edge12_ns = 1

        # 生成node2->node1的噪声边（对称处理）
        edge21_ns = 0
        rand2 = random.random()
        if rand2 < q2 and node1 not in adj_list[node2]:
            edge21_ns = 1
        elif rand2 >= q2 and node1 in adj_list[node2]:
            edge21_ns = 1

        # ====== 构建邻接向量 ======
        # 将集合转换为二进制向量（优化内存访问）
        vec1 = [1 if j in adj_list[node1] else 0 for j in range(node_num)]
        vec2 = [1 if j in adj_list[node2] else 0 for j in range(node_num)]

        # ====== Wedge统计 ======
        c1 = 0.0
        for j in range(node_num):
            if j == node1 or j == node2:
                continue
                
            # 原始wedge存在性检查
            wedge = 1 if vec1[j] and vec2[j] else 0
            
            # 差分隐私扰动
            rnd = random.random()
            wedge_ns = 0
            if wedge == 0 and rnd < q1w:    # 翻转0->1
                wedge_ns = 1
            elif wedge == 1 and rnd >= q1w: # 保持1->1
                wedge_ns = 1

            # 无偏估计量累计
            c1 += (wedge_ns - (1 - p1w)) / (2*p1w - 1)

        # ====== 四环估计计算 ======
        term1 = c1 * (c1 - 1) / 2
        term2 = (node_num-2)*p1w*(1-p1w) / (2*(2*p1w-1)**2)

        q2ind = q2*q2/(q2*q2+(1-q2)**2)
        match edge12_ns+edge21_ns:
            case 0:
                weight = q2ind
            case 1:
                weight = 0.5
            case 2:
                weight = 1-q2ind
        tri2_num_ns += (term1 - term2)*weight
        processed_pairs += 1

    # ====== 最终结果校正 ======
    return tri2_num_ns * node_num*(node_num-1) / (2*processed_pairs)

def calc_local_tri_his(adj_list,node_num,tri_max,epsilon):
    # === 初始化噪声邻接列表 ===
    # 使用字典存储噪声后的邻接关系，格式为 {节点i: [邻居节点列表]}
    a_mat_ns = defaultdict(set)

    # === 参数计算 ===

    # 计算随机响应概率p1
    mu = math.exp(epsilon) / (math.exp(epsilon) + 1.0)
    murho = mu / math.exp(epsilon)  # 翻转边不存在的概率

    # === 生成噪声邻接列表 ===
    for i in range(node_num):
        for j in range(i + 1, node_num):
            rnd = random.random()
            # 原始边不存在的情况（翻转添加边）
            if rnd < murho and j not in adj_list[i]:
                a_mat_ns[i].add(j)
                a_mat_ns[j].add(i)
            # 原始边存在的情况（保持边）
            elif rnd < mu and j in adj_list[i]:
                a_mat_ns[i].add(j)
                a_mat_ns[j].add(i)


    # 创建一个直方图
    histogram = [0]*(tri_max+1)

    # 遍历每个节点 u
    for u in range(node_num):
        # 遍历节点 u 的所有邻居 v
        for v in a_mat_ns[u]:
            if v > u:  # 确保每对节点只处理一次
                # 使用集合交集操作找出 u 和 v 的共同邻居
                com_ner = len(a_mat_ns[u].intersection(a_mat_ns[v]))
                com_ner = com_ner if com_ner < tri_max else tri_max
                histogram[com_ner]+=1
    
    histogram[0] += node_num*(node_num-1)//2 - sum(histogram)

    return histogram


def calc_local_wedge_his(adj_list,node_num,wedge_max,epsilon):
    # === 初始化噪声邻接列表 ===
    # 使用字典存储噪声后的邻接关系，格式为 {节点i: [邻居节点列表]}
    a_mat_ns = defaultdict(set)

    # === 参数计算 ===

    # 计算随机响应概率p1
    mu = math.exp(epsilon) / (math.exp(epsilon) + 1.0)

    murho = mu / math.exp(epsilon)  # 翻转边不存在的概率

    # === 生成噪声邻接列表 ===
    for i in range(node_num):
        for j in range(i + 1, node_num):
            rnd = random.random()
            # 原始边不存在的情况（翻转添加边）
            if rnd < murho and j not in adj_list[i]:
                a_mat_ns[i].add(j)
                a_mat_ns[j].add(i)
            # 原始边存在的情况（保持边）
            elif rnd < mu and j in adj_list[i]:
                a_mat_ns[i].add(j)
                a_mat_ns[j].add(i)


    # 创建一个直方图
    histogram = [0]*(wedge_max+1)

    # 遍历每个节点 u
    for u in range(node_num):
        # 遍历节点 u 的所有邻居 v
        for v in range(u + 1, node_num):
            com_ner = len(a_mat_ns[u].intersection(a_mat_ns[v]))
            com_ner = com_ner if com_ner < wedge_max else wedge_max
            histogram[com_ner]+=1

    return histogram

def calc_shuffle_tri_his(a_mat,node_num,tri_max,epsilon):
    """
    Args:
        a_mat (dict): 邻接表 {节点: {邻居集合}}
        node_num (int): 总节点数
        tri_max (int): 边相关的最大三角形数量
        eps_t (float): 隐私预算
        eps_l (float): shuffle放大后的隐私预算，和上一个对应
    Returns:

    """
    # ====== 初始化部分 ======
    node_order = list(range(node_num))  # 节点顺序
    random.shuffle(node_order)  #随机化节点序列

    # 设置wedge翻转概率
    eps_l = calc_eps_l(node_num,epsilon)
    q1w = 1/(math.exp(eps_l)+1)

    # 设置边翻转概率
    q2 = 1/(math.exp(epsilon)+1)

    # ====== 处理节点对 ======
    histogram = [0]*(tri_max+1)
    for i in range(0, node_num, 2):
        # 获取节点对
        if i+1 >= node_num: break
        node1 = node_order[i]
        node2 = node_order[i+1]

        # ====== 边噪声处理 ======
        # 生成node1->node2的噪声边
        edge12_ns = 0
        rand1 = random.random()
        if rand1 < q2 and node2 not in a_mat[node1]:
            edge12_ns = 1
        elif rand1 >= q2 and node2 in a_mat[node1]:
            edge12_ns = 1

        # 生成node2->node1的噪声边（对称处理）
        edge21_ns = 0
        rand2 = random.random()
        if rand2 < q2 and node1 not in a_mat[node2]:
            edge21_ns = 1
        elif rand2 >= q2 and node1 in a_mat[node2]:
            edge21_ns = 1

        # ====== 构建邻接向量 ======
        # 节点1的邻接向量（存在边为1，否则0）
        a_mat_node1 = [0]*node_num
        for neighbor in a_mat[node1]:
            a_mat_node1[neighbor] = 1
        
        # 节点2的邻接向量
        a_mat_node2 = [0]*node_num
        for neighbor in a_mat[node2]:
            a_mat_node2[neighbor] = 1

        # ====== 楔形(wedge)统计 ======
        wedge_num = 0
        for j in range(node_num):
            if j == node1 or j == node2: continue
            
            # 原始wedge存在性
            wedge = 1 if a_mat_node1[j] and a_mat_node2[j] else 0
            
            # 添加wedge噪声
            rand3 = random.random()
            if rand3 < q1w and wedge ==0:
                wedge_ns = 1
            elif rand3 >= q1w and wedge ==1:
                wedge_ns = 1
            else:
                wedge_ns = 0
            
            if wedge_ns:
                wedge_num +=1
        
        # ===== 直方图统计 =======
        index = (wedge_num-(node_num-2)*q1w)/(1-2*q1w)
        decimal_part = index - math.floor(index)
        # 根据小数部分p，随机决定是向下取整还是向上取整
        if random.random() < decimal_part:  # 向下取整的概率为p，向上取整的概率为1-p
            index =  math.floor(index)  # 向下取整
        else:
            index =  math.ceil(index)  # 向上取整
        index = max(0, min(index, tri_max))


        q2ind = q2*q2/(q2*q2+(1-q2)**2)
        match edge12_ns+edge21_ns:
            case 0:
                histogram[0] += 1-q2ind
                histogram[index] += q2ind
            case 1:
                histogram[0] += 0.5
                histogram[index] += 0.5
            case 2:
                histogram[index] += 1-q2ind
                histogram[0] += q2ind

    for i in range(tri_max+1):
        histogram[i] *= (node_num-1)

    return histogram

def calc_shuffle_wedge_his(a_mat,node_num,deg_max,epsilon):
    """
    Args:
        a_mat (dict): 邻接表 {节点: {邻居集合}}
        node_num (int): 总节点数
        tri_max (int): 边相关的最大三角形数量
        eps_t (float): 隐私预算
        eps_l (float): shuffle放大后的隐私预算，和上一个对应
    Returns:

    """
    # ====== 初始化部分 ======
    node_order = list(range(node_num))  # 节点顺序
    random.shuffle(node_order) #随机化节点序列

    # 设置wedge翻转概率
    eps_l = calc_eps_l(node_num,epsilon)
    q1w = 1/(math.exp(eps_l)+1)
  
    # ====== 处理节点对 ======
    histogram = [0]*(deg_max+1)
    for i in range(0, node_num, 2):
        # 获取节点对
        if i+1 >= node_num: break
        node1 = node_order[i]
        node2 = node_order[i+1]

        # ====== 构建邻接向量 ======
        # 节点1的邻接向量（存在边为1，否则0）
        a_mat_node1 = [0]*node_num
        for neighbor in a_mat[node1]:
            a_mat_node1[neighbor] = 1
        # 节点2的邻接向量
        a_mat_node2 = [0]*node_num
        for neighbor in a_mat[node2]:
            a_mat_node2[neighbor] = 1

        # ====== 楔形(wedge)统计 ======
        wedge_num = 0
        for j in range(node_num):
            if j == node1 or j == node2: continue
            # 原始wedge存在性
            wedge = 1 if a_mat_node1[j] and a_mat_node2[j] else 0
            # 添加wedge噪声
            rand3 = random.random()
            if rand3 < q1w and wedge ==0:
                wedge_ns = 1
            elif rand3 >= q1w and wedge ==1:
                wedge_ns = 1
            else:
                wedge_ns = 0
            
            if wedge_ns:
                wedge_num +=1
        
        # ===== 直方图统计 =======
        index = (wedge_num-(node_num-2)*q1w)/(1-2*q1w)
        decimal_part = index - math.floor(index)
        # 根据小数部分p，随机决定是向下取整还是向上取整
        if random.random() < decimal_part:  # 向下取整的概率为p，向上取整的概率为1-p
            index =  math.floor(index)  # 向下取整
        else:
            index =  math.ceil(index)  # 向上取整

        index = max(0, min(index, deg_max))

        histogram[index]+=1

    for i in range(deg_max+1):
        histogram[i] *= (node_num-1)

    return histogram

def count_triangles(adj_list):
    triangle_count = 0

    # 遍历每个节点 u
    for u in range(len(adj_list)):
        # 遍历节点 u 的所有邻居 v
        for v in adj_list[u]:
            if v > u:  # 确保只处理每对节点一次
                # 使用集合交集操作找出 u 和 v 的共同邻居
                common_neighbors = adj_list[u] & adj_list[v]
                
                # 对于每个共同邻居，形成一个三角形
                for w in common_neighbors:
                    if w > v:  # 确保每个三角形只计算一次
                        triangle_count += 1

    return triangle_count

def MSE(true_value,est_values):
    sorted_data = sorted(est_values)
    trimmed_data = sorted_data[2:-2]
    n = len(trimmed_data)
    mse = 0
    for value in trimmed_data:
        mse+=(true_value-value)**2
    return mse/n

def RE(true_value,est_values):
    sorted_data = sorted(est_values)
    trimmed_data = sorted_data[2:-2]
    n = len(trimmed_data)
    mse = 0
    for value in trimmed_data:
        mse += abs(value-true_value)
    return mse/n

def tri_arr(epsilon,filename,node_num,fre = 20):
    adj_list = dp.load_adj_list_from_csv(filename)

    triangles = []
    for i in range(fre):
        triangles.append(calc_loc_tri_arr(adj_list,node_num,epsilon))

    return triangles

def tri_shuffle(epsilon,filename,node_num,fre =20,alg = 2):

    adj_list = dp.load_adj_list_from_csv(filename)
    triangles = []

    deg = np.zeros(node_num)
    for index, neighbors in enumerate(adj_list):
        deg[index] = len(neighbors)
    eps_t = epsilon/2
    if alg == 4:
        eps_d = 0.1*eps_t
        eps_t = eps_t - eps_d
    else:
        eps_d = 200
    eps_l = calc_eps_l(node_num-2,eps_t)
    if (eps_l < eps_t):
        eps_l = eps_t

    for i in range(fre):
        triangles.append(calc_shuffle_tri(adj_list,deg,alg,node_num,eps_t,eps_l,eps_d))

    return triangles


def cy4_shuffle(epsilon,filename, node_num,fre =20, alg = 8):

    adj_list = dp.load_adj_list_from_csv(filename)
    cy4_arr = []
    eps_t = epsilon/2
    eps_l = calc_eps_l(node_num-2,eps_t)
    if (eps_l < eps_t):
        eps_l = eps_t

    for i in range(fre):
        cy4_arr.append(calc_shuffle_cy4(adj_list, node_num,alg, eps_t, eps_l))
        # print(cy4_arr[-1])

    return cy4_arr

def tri2_shuffle(epsilon,filename, node_num, fre =20,alg = 8):

    adj_list = dp.load_adj_list_from_csv(filename)
    tri2_arr = []
    eps_t = epsilon/2
    eps_l = calc_eps_l(node_num-2,eps_t)
    if (eps_l < eps_t):
        eps_l = eps_t

    for i in range(fre):
        tri2_arr.append(calc_shuffle_tri2(adj_list, node_num,alg, eps_t, eps_l))
        # print(cy4_arr[-1])

    return tri2_arr

