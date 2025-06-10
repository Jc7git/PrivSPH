import random
import math
import time
import os
from collections import defaultdict
import DataProcess as dp
import GenNoiseMech as gm
import numpy as np
import ComMethod as cm

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

def count_triangle_by_wedge(adj_list,wedge_list):
    
    triangle_count = 0

    for u, v in wedge_list:
        if v in adj_list[u]:
            triangle_count +=  len(adj_list[u].intersection(adj_list[v]))

    return triangle_count

def count_cy4s(adj_list):
    cy4_count = 0
    node_num = len(adj_list)

    # 遍历每个节点可能的节点对 u,v
    for u in range(node_num):
        for v in range(u+1,node_num):
            # 使用集合交集操作找出 u 和 v 的共同邻居
            common_neighbors = len(adj_list[u] & adj_list[v])
            if(common_neighbors>1):
                cy4_count+= common_neighbors*(common_neighbors-1)//2

    return cy4_count//2

def count_cy4_by_wedge(adj_list,wedge_list):

    cy4_count = 0
    for u, v in wedge_list:
            com_ner =  len(adj_list[u].intersection(adj_list[v]))
            if (com_ner>1):
                cy4_count += com_ner*(com_ner-1)//2

    return cy4_count


def count_tri2(adj_list):
    tri2_count = 0
    node_num = len(adj_list)
    # 遍历每个节点可能的节点对 u,v
    for u in range(node_num):
        for v in range(u+1,node_num):
            # 使用集合交集操作找出 u 和 v 的共同邻居
            if v in adj_list[u]:
                common_neighbors = len(adj_list[u] & adj_list[v])
                if(common_neighbors>1):
                    tri2_count+= common_neighbors*(common_neighbors-1)//2

    return tri2_count

def count_tri2_by_wedge(adj_list,wedge_list):
    tri2_count = 0
    for u, v in wedge_list:
            if v in adj_list[u]:
                com_ner =  len(adj_list[u].intersection(adj_list[v]))
                if (com_ner>1):
                    tri2_count += com_ner*(com_ner-1)//2
    return tri2_count


def count_triangle_histogram(adj_list,tri_max):

    # 记录每对节点 (u, v) 参与的三角形数量
    triangle_count_per_pair = defaultdict(int)
    n = len(adj_list)
    # 遍历每个节点 u
    for u in range(len(adj_list)):
        # 遍历节点 u 的所有邻居 v
        for v in adj_list[u]:
            if v > u:  # 确保每对节点只处理一次
                # 使用集合交集操作找出 u 和 v 的共同邻居
                com_ner = len(adj_list[u].intersection(adj_list[v]))
                triangle_count_per_pair[(u, v)] = com_ner if com_ner < tri_max else tri_max

    # 创建一个直方图
    histogram = [0]*(tri_max+1)
    
    # 统计每对节点参与三角形的数量频率
    for count in triangle_count_per_pair.values():
        histogram[count] += 1
    histogram[0] += n*(n-1)//2 - sum(histogram)

    return histogram

def count_triangle_histogram_by_wedge(adj_list,wedge_list,tri_max):
    # 创建一个直方图
    histogram = [0]*(tri_max+1)
    # 遍历wedge_list
    for u, v in wedge_list:
        if v in adj_list[u]:
            com_ner = len(adj_list[u].intersection(adj_list[v]))
            com_ner = com_ner if com_ner < tri_max else tri_max 
        else:
            com_ner = 0
        histogram[com_ner]+=1

    return histogram

def count_triangle_histogram_by_wedge_noise(adj_list,wedge_list,tri_max,epsilon):
    # 创建一个直方图
    histogram = [0]*(tri_max+1)
    # 遍历wedge_list
    for u, v in wedge_list:
        if v in adj_list[u]:
            com_ner = len(adj_list[u].intersection(adj_list[v]))
        else:
            com_ner = 0

        com_ner += gm.discrete_laplace_mechanism(epsilon,tri_max)
        com_ner = max(min(tri_max,com_ner),0)
        histogram[com_ner]+=1

    return histogram


def count_wedge_histogram(adj_list,deg_max):
    
    node_num = len(adj_list)
    
    his = [0]*(deg_max+1)
    # 遍历每个节点可能的节点对 u,v
    for u in range(node_num):
        for v in range(u+1,node_num):
            com_ner = len(adj_list[u].intersection(adj_list[v]))
            if com_ner<0 or com_ner>deg_max:
                print(com_ner)
            his[com_ner]+=1
    
    return his

def count_wedge_histogram_by_wedge(adj_list,wedge_list,deg_max):
    histogram = [0]*(deg_max+1)
    # 遍历wedge_list
    for u, v in wedge_list:
        com_ner = len(adj_list[u].intersection(adj_list[v]))
        com_ner = com_ner if com_ner < deg_max else deg_max 
        histogram[com_ner]+=1
    return histogram

def count_wedge_histogram_by_wedge_noise(adj_list,wedge_list,deg_max,epsilon):
    histogram = [0]*(deg_max+1)
    # 遍历wedge_list
    for u, v in wedge_list:
        com_ner = len(adj_list[u].intersection(adj_list[v]))
        com_ner += gm.discrete_laplace_mechanism(epsilon,1)
        com_ner = max(min(deg_max,com_ner),0)
        histogram[com_ner]+=1
    return histogram


def count_triangle_under_DP_by_novel_sampling(adj_list, node_num, deg_max, epsilon, privacy_loss,k,noAm = 0):
    triangles = 0
    node_wedge = list(range(node_num))

    for _ in range(k):
        discrete_noise1 = gm.generate_discrete_laplace_after_amplification((deg_max-1)*3,epsilon/k,node_num,deg_max-1,noAm)
        random.shuffle(node_wedge)
        wedge_list = [(node_wedge[i], node_wedge[i+1]) for i in range(0, len(node_wedge)-1, 2)]
        wedge_triangles = count_triangle_by_wedge(adj_list,wedge_list)
        noi_tri = wedge_triangles+discrete_noise1
        triangles+=noi_tri*(node_num-1)/3
   
    return round(triangles/k)  

def count_tri_his_under_DP_by_novel_sampling(adj_list, node_num, deg_max, epsilon, delta,k,noAm = 0):
    est_his = [0]*(deg_max)

    node_wedge = list(range(node_num))
    for _ in range(k):
        random.shuffle(node_wedge)
        wedge_list = [(node_wedge[i], node_wedge[i+1]) for i in range(0, len(node_wedge)-1, 2)]
        his = count_triangle_histogram_by_wedge(adj_list,wedge_list,deg_max-1)
        noise = gm.generate_laplace_vector_after_amplification(6,epsilon/k,node_num,deg_max-1,noAm)
        for i in range(deg_max):
            est_his[i]+= (his[i] + noise[i])
  
    for i in range(deg_max):
        est_his[i] = est_his[i]/k*(node_num-1)

    return est_his

def count_tri_his_under_element_DP_by_novel_sampling(adj_list, node_num, deg_max, epsilon, delta,k,noAm = 0):
    est_his = [0]*(deg_max)

    node_wedge = list(range(node_num))
    for _ in range(k):
        random.shuffle(node_wedge)
        wedge_list = [(node_wedge[i], node_wedge[i+1]) for i in range(0, len(node_wedge)-1, 2)]
        his = count_triangle_histogram_by_wedge_noise(adj_list,wedge_list,deg_max-1,epsilon)
        for i in range(deg_max):
            est_his[i]+= his[i]
  
    for i in range(deg_max):
        est_his[i] = est_his[i]/k*(node_num-1)

    return est_his

def count_wedge_his_under_DP_by_novel_sampling(adj_list, node_num, deg_max, epsilon, delta,k,noAm = 0):
    est_his = [0]*(deg_max+1)

    node_wedge = list(range(node_num))

    for _ in range(k):
        random.shuffle(node_wedge)
        wedge_list = [(node_wedge[i], node_wedge[i+1]) for i in range(0, len(node_wedge)-1, 2)]
        his = count_wedge_histogram_by_wedge(adj_list,wedge_list,deg_max)
        noise = gm.generate_laplace_vector_after_amplification(6,epsilon/k,node_num,2*deg_max,noAm)
        
        for i in range(deg_max+1):
            est_his[i]+= (his[i] + noise[i])*(node_num-1)
  
    for i in range(deg_max+1):
        est_his[i] = est_his[i]/k

    return est_his

def count_wedge_his_under_element_DP_by_novel_sampling(adj_list, node_num, deg_max, epsilon, delta,k,noAm = 0):
    est_his = [0]*(deg_max+1)

    node_wedge = list(range(node_num))

    for _ in range(k):
        random.shuffle(node_wedge)
        wedge_list = [(node_wedge[i], node_wedge[i+1]) for i in range(0, len(node_wedge)-1, 2)]
        his = count_wedge_histogram_by_wedge_noise(adj_list,wedge_list,deg_max,epsilon)
        
        for i in range(deg_max+1):
            est_his[i]+= his[i] *(node_num-1)
  
    for i in range(deg_max+1):
        est_his[i] = est_his[i]/k

    return est_his
  
def count_tri2_under_DP_by_novel_sampling(adj_list, node_num, deg_max, epsilon, privacy_loss,k,noAm = 0):
    tri2s = 0
    node_wedge = list(range(node_num))

    for _ in range(k):
        discrete_noise1 = gm.generate_discrete_laplace_after_amplification((deg_max-1)*(deg_max-2)*3,epsilon/k,node_num,deg_max-1,noAm)
        random.shuffle(node_wedge)
        wedge_list = [(node_wedge[i], node_wedge[i+1]) for i in range(0, len(node_wedge)-1, 2)]
        wedge_tri2 = count_tri2_by_wedge(adj_list,wedge_list)
        tri2s+=(wedge_tri2+discrete_noise1)*(node_num-1)
    
    return round(tri2s/k)

def count_cyc4_under_DP_by_novel_sampling(adj_list, node_num, deg_max, epsilon, privacy_loss,k,noAm = 0):
    cy4s = 0
    node_wedge = list(range(node_num))

    for _ in range(k):
        discrete_noise1 = gm.generate_discrete_laplace_after_amplification((deg_max-1)*(deg_max)*3,epsilon/k,node_num,2*deg_max,noAm)
        random.shuffle(node_wedge)
        wedge_list = [(node_wedge[i], node_wedge[i+1]) for i in range(0, len(node_wedge)-1, 2)]
        wedge_cy4 = count_cy4_by_wedge(adj_list,wedge_list)
        cy4s+=(wedge_cy4+discrete_noise1)*(node_num-1)//2
    
    return round(cy4s/k)


def count_tri_under_CDP(filename,deg_max,epsilon):
    if os.path.exists(f"./midData/triangle/triangle_count_{filename}.pkl"):
        triangles = dp.load_object_from_file(f"./midData/triangle/triangle_count_{filename}.pkl")
    else:
        adj_list = dp.load_adj_list_from_csv(f"./graphData/{filename}.csv")
        triangles = count_triangles(adj_list)
        # 保存列表到文件
        dp.save_object_to_file(triangles, f"./midData/triangle/triangle_count_{filename}.pkl")
  
    noise = gm.discrete_laplace_mechanism(epsilon,deg_max-1)

    return triangles+noise

def count_cyc4_under_CDP(filename,deg_max,epsilon):
    if os.path.exists(f"./midData/cycle4/cycle4_count_{filename}.pkl"):
        num_cy4s = dp.load_object_from_file(f"./midData/cycle4/cycle4_count_{filename}.pkl")
    else:
        adj_list = dp.load_adj_list_from_csv(f"./graphData/{filename}.csv")
        num_cy4s = count_cy4s(adj_list)
        # 保存列表到文件
        dp.save_object_to_file(num_cy4s, f"./midData/cycle4/cycle4_count_{filename}.pkl")
    noise = gm.discrete_laplace_mechanism(epsilon,(deg_max-1)**2)
    return num_cy4s+noise

def count_tri2_under_CDP(filename,deg_max,epsilon):
    if os.path.exists(f"./midData/triangle2/triangle2_count_{filename}.pkl"):
        tri2s = dp.load_object_from_file(f"./midData/triangle2/triangle2_count_{filename}.pkl")
    else:
        adj_list = dp.load_adj_list_from_csv(f"./graphData/{filename}.csv")
        tri2s = count_tri2(adj_list)
        # 保存列表到文件
        dp.save_object_to_file(tri2s, f"./midData/triangle2/triangle2_count_{filename}.pkl")
    noise = gm.discrete_laplace_mechanism(epsilon,3*(deg_max-1)*(deg_max-2))
    
    return tri2s+noise

def count_tri_his_under_CDP(filename,deg_max,epsilon,delta):
    if os.path.exists(f"./midData/triangle/triangle_his_{filename}.pkl"):
        tri_his = dp.load_object_from_file(f"./midData/triangle/triangle_his_{filename}.pkl")
    else:
        adj_list = dp.load_adj_list_from_csv(f"./graphData/{filename}.csv")
        tri_his = count_triangle_histogram(adj_list,deg_max-1)
        # 保存列表到文件
        dp.save_object_to_file(tri_his, f"./midData/triangle/triangle_his_{filename}.pkl")

    # 拉普拉斯机制
    scale = (2*deg_max-1)*2 / epsilon
    noise = np.random.laplace(0, scale, deg_max)

    noise_tri_his = [0]*deg_max
    for i in range(deg_max):
        noise_tri_his[i] = tri_his[i]+noise[i]

    return noise_tri_his

def count_wedge_his_under_CDP(filename,deg_max,epsilon,delta):

    if os.path.exists(f"./midData/wedge/wedge_his_{filename}.pkl"):
        wedge_his = dp.load_object_from_file(f"./midData/wedge/wedge_his_{filename}.pkl")
    else:
        adj_list = dp.load_adj_list_from_csv(f"./graphData/{filename}.csv")
        wedge_his = count_wedge_histogram(adj_list,deg_max)
        # 保存列表到文件
        dp.save_object_to_file(wedge_his, f"./midData/wedge/wedge_his_{filename}.pkl")
   

    # 拉普拉斯机制
    scale = (2*deg_max-2)*2 / epsilon
    noise = np.random.laplace(0, scale, deg_max+1)

    noise_wedge_his = [0]*(deg_max+1)
    for i in range(deg_max+1):
        noise_wedge_his[i] = wedge_his[i]+noise[i]

    return noise_wedge_his


def adjust_list(lst, target_sum):
    # 1. 将所有小于0的元素变为0
    lst = [max(x, 0) for x in lst]
    
    # 2. 计算修改后的元素的和
    current_sum = sum(lst)
    
    # 如果当前和已经是目标和，则不需要做任何修改
    if current_sum == target_sum:
        return lst
    
    # 3. 计算放缩系数
    if current_sum == 0:
        # 如果原和为0，直接将所有元素设为0
        return [0] * len(lst)
    
    scale_factor = target_sum / current_sum
    
    # 4. 对所有元素进行等系数放缩
    scaled_lst = [x * scale_factor for x in lst]
    
    # 5. 让所有元素成为整数
    scaled_lst = [math.floor(x) for x in scaled_lst]
    
    # 6. 调整第一个元素使得总和为 target_sum
    scaled_lst[0] = target_sum - sum(scaled_lst[1:])
    
    return scaled_lst

def L2loss(true_value,est_values):
    # sorted_data = sorted(est_values)
    # trimmed_data = sorted_data[2:-2]
    trimmed_data = est_values
    n = len(trimmed_data)
    mse = 0
    for value in trimmed_data:
        mse+=(true_value-value)**2
    return mse/n

def RE(true_value,est_values):
    # sorted_data = sorted(est_values)
    # trimmed_data = sorted_data[2:-2]
    trimmed_data = est_values
    n = len(trimmed_data)
    mse = 0
    for value in trimmed_data:
        mse += abs(value-true_value)/true_value
    return mse/n


def exp_tri(epsilon,delta,filename,node_num,deg_max,fre,alg,k):
    if os.path.exists(f"./midData/triangle/triangle_count_{filename}.pkl"):
        num_triangles = dp.load_object_from_file(f"./midData/triangle/triangle_count_{filename}.pkl")
    else:
        adj_list = dp.load_adj_list_from_csv(f"./graphData/{filename}.csv")
        num_triangles = count_triangles(adj_list)
        # 保存列表到文件
        dp.save_object_to_file(num_triangles, f"./midData/triangle/triangle_count_{filename}.pkl")
    print("真实值",num_triangles)

    start_time = time.time()
    triangles = []
    match alg:
        case "DDP":
            adj_list = dp.load_adj_list_from_csv(f"./graphData/{filename}.csv")
            for _ in range(fre):
          
                triangles.append(count_triangle_under_DP_by_novel_sampling(adj_list,node_num,deg_max,epsilon,delta,k))
        case "DDP_noAm":
            adj_list = dp.load_adj_list_from_csv(f"./graphData/{filename}.csv")
            for _ in range(fre):
         
                triangles.append(count_triangle_under_DP_by_novel_sampling(adj_list,node_num,deg_max,epsilon,delta,1,1))
        case "DDP_1":
            adj_list = dp.load_adj_list_from_csv(f"./graphData/{filename}.csv")
            for _ in range(fre):
              
                triangles.append(count_triangle_under_DP_by_novel_sampling(adj_list,node_num,deg_max,epsilon,delta,1))
        case "CDP":
            for _ in range(fre):
                triangles.append(count_tri_under_CDP(filename,deg_max,epsilon))
        case "shuffle":
            triangles = cm.tri_shuffle(epsilon,f"./graphData/{filename}.csv",node_num,fre)
        case "ARR":
            triangles = cm.tri_arr(epsilon,f"./graphData/{filename}.csv",node_num,fre)
    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算运行时间（秒）

    with open('./results/triangle_count_log.txt', 'a') as f:
        f.write("####### Parameter Setting #######\n")
        f.write("task_type:subgraph_count\n")
        f.write("task_name:triangle_count\n")
        f.write("Privacy Budget:"+str(epsilon)+"\n")
        f.write("Number of nodes:"+str(node_num)+"\n")
        f.write("Maximum Degree:"+str(deg_max)+"\n")
        f.write("####### Results #######\n")
        f.write("Estimated count:"+str(round(np.mean(triangles)))+"\n")
        f.write("MSE:"+str(L2loss(num_triangles,triangles))+"\n")
        f.write("RE:"+str(RE(num_triangles,triangles))+"\n")
        f.write( f"Times: {elapsed_time:.6f} s")
        # f.write("true count:"+str(num_triangles)+"\n")
    # print("####### Parameter Setting #######\n")
    # print("task_type:subgraph_count\n")
    # print("task_name:triangle_count\n")
    # print("Privacy Budget:"+str(epsilon)+"\n")
    # print("Number of nodes:"+str(node_num)+"\n")
    # print("Maximum Degree:"+str(deg_max)+"\n")
    # print("####### Results #######\n")
    # print("Estimated count:"+str(round(np.mean(triangles)))+"\n")
    # print("MSE:"+str(L2loss(num_triangles,triangles))+"\n")
    # print("RE:"+str(RE(num_triangles,triangles))+"\n")
    # print( f"Times: {elapsed_time:.6f} s")
    

    print(f"triangle {alg} 均值:{round(np.mean(triangles))}")
    print(f"triangle {alg} l2 loss:{L2loss(num_triangles,triangles)}")
    print(f"triangle {alg} 相对误差:{RE(num_triangles,triangles)}")

    tri_L2 = []
    tri_RE = []
    for est_num in triangles:
        tri_L2.append((num_triangles-est_num)**2)
        tri_RE.append(abs(num_triangles-est_num)/num_triangles)
    return triangles, tri_L2, tri_RE

def exp_tri2(epsilon,delta,filename,node_num,deg_max,fre,alg,k):
    if os.path.exists(f"./midData/triangle2/triangle2_count_{filename}.pkl"):
        num_tri2s = dp.load_object_from_file(f"./midData/triangle2/triangle2_count_{filename}.pkl")
    else:
        adj_list = dp.load_adj_list_from_csv(f"./graphData/{filename}.csv")
        num_tri2s = count_tri2(adj_list)
        # 保存列表到文件
        dp.save_object_to_file(num_tri2s, f"./midData/triangle2/triangle2_count_{filename}.pkl")
   
    print("真实值",num_tri2s)
    start_time = time.time()
    tri2s = []
    match alg:
        case "DDP":
            adj_list = dp.load_adj_list_from_csv(f"./graphData/{filename}.csv")
            for _ in range(fre):
              
                tri2s.append(count_tri2_under_DP_by_novel_sampling(adj_list,node_num,deg_max,epsilon,delta,k))
        case "DDP_noAm":
            adj_list = dp.load_adj_list_from_csv(f"./graphData/{filename}.csv")
            for _ in range(fre):
                
                tri2s.append(count_tri2_under_DP_by_novel_sampling(adj_list,node_num,deg_max,epsilon,delta,1,1))
        case "DDP_1":
            adj_list = dp.load_adj_list_from_csv(f"./graphData/{filename}.csv")
            for _ in range(fre):
        
                tri2s.append(count_tri2_under_DP_by_novel_sampling(adj_list,node_num,deg_max,epsilon,delta,1))
        case "CDP":
            for _ in range(fre):
                tri2s.append(count_tri2_under_CDP(filename,deg_max,epsilon))
        case "shuffle":
            tri2s = cm.tri2_shuffle(epsilon,f"./graphData/{filename}.csv",node_num,fre)
    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算运行时间（秒）

    with open('./results/2-triangle_count_log.txt', 'a') as f:
        f.write("####### Parameter Setting #######\n")
        f.write("task_type:subgraph_count\n")
        f.write("task_name:2-triangle_count\n")
        f.write("Privacy Budget:"+str(epsilon)+"\n")
        f.write("Number of nodes:"+str(node_num)+"\n")
        f.write("Maximum Degree:"+str(deg_max)+"\n")
        f.write("####### Results #######\n")
        f.write("Estimated count:"+str(round(np.mean(tri2s)))+"\n")
        f.write("MSE:"+str(L2loss(num_tri2s,tri2s))+"\n")
        f.write("RE:"+str(RE(num_tri2s,tri2s))+"\n")
        f.write( f"Times: {elapsed_time:.6f} s")
        # f.write("true count:"+str(num_triangles)+"\n")

    print(f"triangle2 {alg} 均值:{round(np.mean(tri2s))}")
    print(f"triangle2 {alg} 均方差:{L2loss(num_tri2s,tri2s)}")
    print(f"triangle2 {alg} 相对误差:{RE(num_tri2s,tri2s)}")

    tri2_L2 = []
    tri2_RE = []
    for est_num in tri2s:
        tri2_L2.append((num_tri2s-est_num)**2)
        tri2_RE.append(abs(num_tri2s-est_num)/num_tri2s)
    return tri2s, tri2_L2, tri2_RE

def exp_cyc4(epsilon,delta,filename,node_num,deg_max,fre,alg,k):

    if os.path.exists(f"./midData/cycle4/cycle4_count_{filename}.pkl"):
        num_cy4s = dp.load_object_from_file(f"./midData/cycle4/cycle4_count_{filename}.pkl")
    else:
        adj_list = dp.load_adj_list_from_csv(f"./graphData/{filename}.csv")
        num_cy4s = count_cy4s(adj_list)
        # 保存列表到文件
        dp.save_object_to_file(num_cy4s, f"./midData/cycle4/cycle4_count_{filename}.pkl")
    print("真实值",num_cy4s)

    cy4s = []
    start_time = time.time()
    match alg:
        case "DDP":
            adj_list = dp.load_adj_list_from_csv(f"./graphData/{filename}.csv")
            for _ in range(fre):
            
                cy4s.append(count_cyc4_under_DP_by_novel_sampling(adj_list,node_num,deg_max,epsilon,delta,k))
        case "DDP_noAm":
            adj_list = dp.load_adj_list_from_csv(f"./graphData/{filename}.csv")
            for _ in range(fre):
           
                cy4s.append(count_cyc4_under_DP_by_novel_sampling(adj_list,node_num,deg_max,epsilon,delta,1,1))
        case "DDP_1":
            adj_list = dp.load_adj_list_from_csv(f"./graphData/{filename}.csv")
            for _ in range(fre):
          
                cy4s.append(count_cyc4_under_DP_by_novel_sampling(adj_list,node_num,deg_max,epsilon,delta,1))
        case "CDP":
            for _ in range(fre):
                cy4s.append(count_cyc4_under_CDP(filename,deg_max,epsilon))
        case "shuffle":
            cy4s = cm.cy4_shuffle(epsilon,f"./graphData/{filename}.csv",node_num,fre)

    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算运行时间（秒）

    with open('./results/4-cycle_count_log.txt', 'a') as f:
        f.write("####### Parameter Setting #######\n")
        f.write("task_type:subgraph_count\n")
        f.write("task_name:4-cycle_count\n")
        f.write("Privacy Budget:"+str(epsilon)+"\n")
        f.write("Number of nodes:"+str(node_num)+"\n")
        f.write("Maximum Degree:"+str(deg_max)+"\n")
        f.write("####### Results #######\n")
        f.write("Estimated count:"+str(round(np.mean(cy4s)))+"\n")
        f.write("MSE:"+str(L2loss(num_cy4s,cy4s))+"\n")
        f.write("RE:"+str(RE(num_cy4s,cy4s))+"\n")
        f.write( f"Times: {elapsed_time:.6f} s")
        # f.write("true count:"+str(num_triangles)+"\n")

    print(f"cycle4 {alg} 均值:{round(np.mean(cy4s))}")
    print(f"cycle4 {alg}  均方差{L2loss(num_cy4s,cy4s)}:")
    print(f"cycle4 {alg}  相对误差:{RE(num_cy4s,cy4s)}")

    cyc4_L2 = []
    cyc4_RE = []
    for est_num in cy4s:
        cyc4_L2.append((num_cy4s-est_num)**2)
        cyc4_RE.append(abs(num_cy4s-est_num)/num_cy4s)
    return cy4s, cyc4_L2, cyc4_RE

def exp_triangle_his(epsilon,delta,filename,node_num,deg_max,fre,alg,k=1):
    if os.path.exists(f"./midData/triangle/triangle_his_{filename}.pkl"):
        true_his = dp.load_object_from_file(f"./midData/triangle/triangle_his_{filename}.pkl")
    else:
        adj_list = dp.load_adj_list_from_csv(f"./graphData/{filename}.csv")
        true_his = count_triangle_histogram(adj_list,deg_max-1)
        # 保存列表到文件
        dp.save_object_to_file(true_his, f"./midData/triangle/triangle_his_{filename}.pkl")

    est_his = []
    tri_mse = []
    tri_nl1 = []
    adj_list = dp.load_adj_list_from_csv(f"./graphData/{filename}.csv")
    EMS = 0
    RE = 0
    EMD = 0
    start_time = time.time()
    for i in range(fre):
        match alg:
            case "DDP":
              
                his = count_tri_his_under_DP_by_novel_sampling(adj_list, node_num,deg_max,epsilon,delta,k)
                his = adjust_list(his,node_num*(node_num-1)//2)
            case "Intial":
              
                his = count_tri_his_under_element_DP_by_novel_sampling(adj_list, node_num,deg_max,epsilon,delta,k)
                his = adjust_list(his,node_num*(node_num-1)//2)
            case "DDP_noAm":
               
                his = count_tri_his_under_DP_by_novel_sampling(adj_list, node_num,deg_max,epsilon,delta,1,1)
                his = adjust_list(his,node_num*(node_num-1)//2)
            case "DDP_1":
         
                his = count_tri_his_under_DP_by_novel_sampling(adj_list, node_num,deg_max,epsilon,delta,1)
                his = adjust_list(his,node_num*(node_num-1)//2)
            case "CDP":
                his = count_tri_his_under_CDP(filename,deg_max,epsilon,delta)
            case "shuffle":
                his = cm.calc_shuffle_tri_his(adj_list,node_num,deg_max-1,epsilon/2)
            case "LDP":
                his = cm.calc_local_tri_his(adj_list,node_num,deg_max-1,epsilon)
        est_his.append(his)
      
        ems = 0
        re = 0
        for i in range(deg_max):
            ems += (his[i]-true_his[i])**2
            re += abs(his[i]-true_his[i])
        EMS += ems/deg_max
        RE += re*2/(node_num*(node_num-1))

        tri_mse.append(ems/deg_max)
        tri_nl1.append(re*2/(node_num*(node_num-1)))

     
        
    
    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算运行时间（秒）
    with open('./results/triangle_histogram_count_log.txt', 'a') as f:
        f.write("####### Parameter Setting #######\n")
        f.write("task_type:subgraph_histogram\n")
        f.write("task_name:triangle_histogram\n")
        f.write("Privacy Budget:"+str(epsilon)+"\n")
        f.write("Number of nodes:"+str(node_num)+"\n")
        f.write("Maximum Degree:"+str(deg_max)+"\n")
        f.write("####### Results #######\n")
        f.write("Estimated histogram:"+str(est_his[0])+"\n")
        f.write("MSE:"+str(EMS/fre)+"\n")
        f.write("RE:"+str(RE/fre)+"\n")
        f.write( f"Times: {elapsed_time:.6f} s")
        # f.write("true count:"+str(num_triangles)+"\n")

    print(f"三角直方图 {alg} 均方差：{EMS/fre}")
    print(f"三角直方图 {alg} 相对误差：{RE/fre}")
    
    return est_his,tri_mse,tri_nl1
  
def exp_wedge_his(epsilon,delta,filename,node_num,deg_max,fre,alg,k=1):
    if os.path.exists(f"./midData/wedge/wedge_his_{filename}.pkl"):
        true_his = dp.load_object_from_file(f"./midData/wedge/wedge_his_{filename}.pkl")
    else:
        adj_list = dp.load_adj_list_from_csv(f"./graphData/{filename}.csv")
        true_his = count_wedge_histogram(adj_list,deg_max)
        # 保存列表到文件
        dp.save_object_to_file(true_his, f"./midData/wedge/wedge_his_{filename}.pkl")

    est_his = []
    wedge_mse = []
    wedge_nl1 = []
    EMS = 0
    RE = 0
    adj_list = dp.load_adj_list_from_csv(f"./graphData/{filename}.csv")
    start_time = time.time()
    for i in range(fre):
        match alg:
            case "DDP":
              
                ori_his = count_wedge_his_under_DP_by_novel_sampling(adj_list, node_num,deg_max,epsilon,delta,k)
                his = adjust_list(ori_his,(node_num-1)*node_num//2)
            case "Intial":
                
                ori_his = count_wedge_his_under_element_DP_by_novel_sampling(adj_list, node_num,deg_max,epsilon,delta,k)
                his = adjust_list(ori_his,(node_num-1)*node_num//2)
            case "DDP_noAm":
               
                ori_his = count_wedge_his_under_DP_by_novel_sampling(adj_list, node_num,deg_max,epsilon,delta,1,1)
                his = adjust_list(ori_his,(node_num-1)*node_num//2)
            case "DDP_1":
                
                ori_his = count_wedge_his_under_DP_by_novel_sampling(adj_list, node_num,deg_max,epsilon,delta,1)
                his = adjust_list(ori_his,(node_num-1)*node_num//2)
            case "CDP":
                his = count_wedge_his_under_CDP(filename,deg_max,epsilon,delta)
            case "shuffle":
                his = cm.calc_shuffle_wedge_his(adj_list,node_num,deg_max,epsilon/2)
            case "LDP":
                his = cm.calc_local_wedge_his(adj_list,node_num,deg_max,epsilon)
        est_his.append(his)
      
        ems = 0
        re = 0
        for i in range(deg_max+1):
            ems += (his[i]-true_his[i])**2
            re += abs(his[i]-true_his[i])

        EMS += ems/(deg_max+1)
        RE += re*2/(node_num*(node_num-1))
        wedge_mse.append(ems/(deg_max+1))
        wedge_nl1.append(re*2/(node_num*(node_num-1)))

    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算运行时间（秒）
    with open('./results/wedge_histogram_log.txt', 'a') as f:
        f.write("####### Parameter Setting #######\n")
        f.write("task_type:subgraph_histogram\n")
        f.write("task_name:wedge_histogram\n")
        f.write("Privacy Budget:"+str(epsilon)+"\n")
        f.write("Number of nodes:"+str(node_num)+"\n")
        f.write("Maximum Degree:"+str(deg_max)+"\n")
        f.write("####### Results #######\n")
        f.write("Estimated histogram:"+str(est_his[0])+"\n")
        f.write("MSE:"+str(EMS/fre)+"\n")
        f.write("RE:"+str(RE/fre)+"\n")
        f.write( f"Times: {elapsed_time:.6f} s")
        # f.write("true count:"+str(num_triangles)+"\n")
   
    print(f"wedge直方图 {alg} 均方差：{EMS/fre}")
    print(f"wedge直方图 {alg} 相对误差：{RE/fre}")

    return est_his,wedge_mse,wedge_nl1



def triangle_count_epsilon(filename,node_num,deg_max,k,epsilons):
    with open('./results/' +filename+'_triangele_count_epsilon.txt', 'a') as f:
        f.write(f"data_name:{filename}\n")
        f.write("task_type:subgraph_count\n")
        f.write("task_name:triangle_count\n")
        f.write("x_name:epsilon\n")
        f.write("x_axis:"+str(epsilons)+"\n")

        # method_list = ["CDP","DDP","DDP_noAm","DDP_1","shuffle","ARR"]
        # fres = [50,200,200,200,50,50]
        method_list = ["shuffle","ARR"]
        fres = [30,30]
    
     
        for i in range(len(method_list)):
            for j, epsilon in enumerate(epsilons) :
                print("epsilon:",epsilon)
                f.write(f"epsilon:{epsilon}\n")
                tri, L2, RE = exp_tri(epsilon,1e-5,filename,node_num,deg_max,fres[i],method_list[i],k[j])
                f.write(f"{method_list[i]} L2Loss:"+str(L2)+"\n")
                f.write(f"{method_list[i]} RE:"+str(RE)+"\n")


def triangle_count_k(filename,node_num,deg_max,ks,epsilons):
    with open('./results/' +filename+'_triangle_count_k.txt', 'a') as f:
        f.write(f"data_name:{filename}\n")
        f.write("task_type:subgraph_count\n")
        f.write("task_name:triangle_count\n")
        f.write("x_name:k\n")
        f.write("x_axis:"+str(ks)+"\n")

        # method_list = ["CDP","DDP","DDP_noAm","DDP_1","shuffle","ARR"]
        # fres = [50,200,200,200,50,50]
        method_list = ["DDP"]
        fres = [200]
    
     
        for i in range(len(epsilons)):
            for j, k in enumerate(ks) :
                print("k:",k)
                f.write(f"k:{k}\n")
                tri, L2, RE = exp_tri(epsilons[i],1e-5,filename,node_num,deg_max,fres[0],method_list[0],k)
                f.write(f"{epsilons[i]} L2Loss:"+str(L2)+"\n")
                f.write(f"{epsilons[i]} RE:"+str(RE)+"\n")


def triangle_2_count_epsilon(filename,node_num,deg_max,k,epsilons):
    with open('./results/' +filename+'_2-triangele_count_epsilon.txt', 'a') as f:
        f.write(f"data_name:{filename}\n")
        f.write("task_type:subgraph_count\n")
        f.write("task_name:triangle_2_count\n")
        f.write("x_name:epsilon\n")
        f.write("x_axis:"+str(epsilons)+"\n")

        # method_list = ["CDP","DDP","DDP_noAm","DDP_1"]
        # fres = [50,300,300,300]
        method_list = ["shuffle"]
        fres = [30]
       
       
        for i in range(len(method_list)):
            for j, epsilon in enumerate(epsilons) :
                print("epsilon:",epsilon)
                f.write(f"epsilon:{epsilon}\n")
                tri2, L2, RE = exp_tri2(epsilon,1e-5,filename,node_num,deg_max,fres[i],method_list[i],k[j])
                f.write(f"{method_list[i]} L2Loss:"+str(L2)+"\n")
                f.write(f"{method_list[i]} RE:"+str(RE)+"\n")

def triangle_2_count_k(filename,node_num,deg_max,ks,epsilons):
    with open('./results/' +filename+'_2-triangele_count_k.txt', 'a') as f:
        f.write(f"data_name:{filename}\n")
        f.write("task_type:subgraph_count\n")
        f.write("task_name:triangle_2_count\n")
        f.write("x_name:k\n")
        f.write("x_axis:"+str(ks)+"\n")

        
        method_list = ["DDP"]
        fres = [200]
       
       
        for i in range(len(epsilons)):
            for j, k in enumerate(ks) :
                print("k:",k)
                f.write(f"k:{k}\n")
                tri2, L2, RE = exp_tri2(epsilons[i],1e-5,filename,node_num,deg_max,fres[0],method_list[0],k)
                f.write(f"{epsilons[i]} L2Loss:"+str(L2)+"\n")
                f.write(f"{epsilons[i]} RE:"+str(RE)+"\n")

def cycle_4_count_epsilon(filename,node_num,deg_max,k,epsilons):
    with open('./results/' +filename+'_cycle_4_count_epsilon.txt', 'a') as f:
        f.write(f"data_name:{filename}\n")
        f.write("task_type:subgraph_count\n")
        f.write("task_name:cycle_4_count\n")
        f.write("x_name:epsilon\n")
        f.write("x_axis:"+str(epsilons)+"\n")

        # method_list = ["CDP","DDP","DDP_noAm","DDP_1"]
        # fres = [50,300,300,300]
        method_list = ["shuffle"]
        fres = [30]
     
        
       

        for i in range(len(method_list)):
            for j, epsilon in enumerate(epsilons) :
                print("epsilon:",epsilon)
                f.write(f"epsilon:{epsilon}\n")
                cyc4, L2, RE = exp_cyc4(epsilon,1e-5,filename,node_num,deg_max,fres[i],method_list[i],k[j])
                f.write(f"{method_list[i]} L2Loss:"+str(L2)+"\n")
                f.write(f"{method_list[i]} RE:"+str(RE)+"\n")

def cycle_4_count_k(filename,node_num,deg_max,ks,epsilons):
    with open('./results/' +filename+'_cycle_4_count_k.txt', 'a') as f:
        f.write(f"data_name:{filename}\n")
        f.write("task_type:subgraph_count\n")
        f.write("task_name:cycle_4_count\n")
        f.write("x_name:k\n")
        f.write("x_axis:"+str(ks)+"\n")

        method_list = ["DDP"]
        fres = [200]
     
        
       

        for i in range(len(epsilons)):
            for j,k in enumerate(ks) :
                print("k:",k)
                f.write(f"k:{k}\n")
                cyc4, L2, RE = exp_cyc4(epsilons[i],1e-5,filename,node_num,deg_max,fres[0],method_list[0],k)
                f.write(f"{epsilons[i]} L2Loss:"+str(L2)+"\n")
                f.write(f"{epsilons[i]} RE:"+str(RE)+"\n")


def triangle_histogram_epsilon(filename,node_num,deg_max,k,epsilons):
    with open('./results/' +filename+'_triangle_histogram_epsilon.txt', 'a') as f:
        f.write(f"data_name:{filename}\n")
        f.write("task_type:subgraph_histogram\n")
        f.write("task_name:triangle_histogram\n")
        f.write("x_name:epsilon\n")
        f.write("x_axis:"+str(epsilons)+"\n")

        # method_list = ["CDP","DDP","DDP_noAm","DDP_1"]
        # fres = [50,200,200,200]
        method_list = ["Intial"]
        fres = [50]
       

        for i in range(len(method_list)):
            for j, epsilon in enumerate(epsilons) :
                print("epsilon:",epsilon)
                f.write(f"epsilon:{epsilon}\n")
                his, mse, nl1 = exp_triangle_his(epsilon,1e-5,filename,node_num,deg_max,fres[i],method_list[i],k[j])
                f.write(f"{method_list[i]} MSE:"+str(mse)+"\n")
                f.write(f"{method_list[i]} NL1:"+str(nl1)+"\n")

def triangle_histogram_k(filename,node_num,deg_max,ks,epsilons):
    with open('./results/' +filename+'_triangle_histogram_k.txt', 'a') as f:
        f.write(f"data_name:{filename}\n")
        f.write("task_type:subgraph_histogram\n")
        f.write("task_name:triangle_histogram\n")
        f.write("x_name:k\n")
        f.write("x_axis:"+str(ks)+"\n")

        method_list = ["DDP"]
        fres = [100]
       

        for i in range(len(epsilons)):
            for j, k in enumerate(ks) :
                print("k:",k)
                f.write(f"k:{k}\n")
                his, mse, nl1 = exp_triangle_his(epsilons[i],1e-5,filename,node_num,deg_max,fres[0],method_list[0],k)
                f.write(f"{epsilons[i]} MSE:"+str(mse)+"\n")
                f.write(f"{epsilons[i]} NL1:"+str(nl1)+"\n")

def wedge_histogram_epsilon(filename,node_num,deg_max,k,epsilons):
    with open('./results/' +filename+'_wedge_histogram_epsilon.txt', 'a') as f:
        f.write(f"data_name:{filename}\n")
        f.write("task_type:subgraph_histogram\n")
        f.write("task_name:wedge_histogram\n")
        f.write("x_name:epsilon\n")
        f.write("x_axis:"+str(epsilons)+"\n")

        # method_list = ["CDP","DDP","DDP_noAm","DDP_1"]
        # fres = [50,200,200,200]
        method_list = ["Intial"]
        fres = [50]
  


        for i in range(len(method_list)):
            for j, epsilon in enumerate(epsilons) :
                print("epsilon:",epsilon)
                f.write(f"epsilon:{epsilon}\n")
                his, mse, nl1 = exp_wedge_his(epsilon,1e-5,filename,node_num,deg_max,fres[i],method_list[i],k[j])
                f.write(f"{method_list[i]} MSE:"+str(mse)+"\n")
                f.write(f"{method_list[i]} NL1:"+str(nl1)+"\n")

def wedge_histogram_k(filename,node_num,deg_max,ks,epsilons):
    with open('./results/' +filename+'_wedge_histogram_k.txt', 'a') as f:
        f.write(f"data_name:{filename}\n")
        f.write("task_type:subgraph_histogram\n")
        f.write("task_name:wedge_histogram\n")
        f.write("x_name:k\n")
        f.write("x_axis:"+str(ks)+"\n")

        method_list = ["DDP"]
        fres = [100]
       

        for i in range(len(epsilons)):
            for j, k in enumerate(ks) :
                print("k:",k)
                f.write(f"k:{k}\n")
                his, mse, nl1 = exp_wedge_his(epsilons[i],1e-5,filename,node_num,deg_max,fres[0],method_list[0],k)
                f.write(f"{epsilons[i]} MSE:"+str(mse)+"\n")
                f.write(f"{epsilons[i]} NL1:"+str(nl1)+"\n")

def main_exp(filename,node_num,deg_max,ks,epsilons):
    
    # triangle_count_epsilon(filename,node_num,deg_max,ks["tc"],epsilons)
    # triangle_2_count_epsilon(filename,node_num,deg_max,ks["t2c"],epsilons)
    # cycle_4_count_epsilon(filename,node_num,deg_max,ks["c4c"],epsilons)
    triangle_histogram_epsilon(filename,node_num,deg_max,ks["th"],epsilons)
    wedge_histogram_epsilon(filename,node_num,deg_max,ks["wh"],epsilons)
    

def main_k(filename,node_num,deg_max,ks,epsilons):
    # 
    # triangle_histogram_k(filename,node_num,deg_max,ks,epsilons)
    # wedge_histogram_k(filename,node_num,deg_max,ks,epsilons)
    # triangle_count_k(filename,node_num,deg_max,ks,epsilons)
    cycle_4_count_k(filename,node_num,deg_max,ks,epsilons)
    triangle_2_count_k(filename,node_num,deg_max,ks,epsilons)
    

if __name__ == "__main__":

    
  
    # epsilons = [0.8,1.0,1.2,1.4,1.6,1.8,2]
    epsilons = [0.4,0.6,0.8,1.0]
    ks = [i for i in range(1,31)]

    
    

    main_k("graph_Gplus",31196,300,ks,epsilons)
    main_k("soc-twitter-higgs",40986,330,ks,epsilons)
    main_k("graph_crankseg",29214,270,ks,epsilons)


    # ## graph_crankseg50_270
    # ks={"tc":[20,20,20,20,20,20,20],
    #     "t2c":[6,7,8,8,9,9,9],
    #     "c4c":[3,3,6,7,7,7,8],    
    #     "th":[5,7,7,8,8,13,13], 
    #     "wh":[3,4,5,5,6,7,7] 
    #     }
    # main_exp("graph_crankseg",29214,270,ks,epsilons)
    

    # ## graph_Gplus50_300
    # ks={"tc":[14,14,15,15,15,15,15], 
    #     "t2c":[7,7,8,10,10,10,11], 
    #     "c4c":[2,3,3,4,6,7,8],  
    #     "th":[5,6,6,7,9,10,12], 
    #     "wh":[3,5,5,6,6,7,8]  
    #     }
    # main_exp("graph_Gplus",31196,300,ks,epsilons)



    # ## soc-twitter-higgs50-330 
    # ks={"tc":[9,9,14,14,15,17,17],  
    #     "t2c":[5,5,6,7,12,14,17], 
    #     "c4c":[4,4,4,7,8,8,8], 
    #     "th":[6,7,8,10,11,12,12],  
    #     "wh":[4,5,6,8,8,9,9]
    #     }
    # main_exp("soc-twitter-higgs",40986,330,ks,epsilons)
