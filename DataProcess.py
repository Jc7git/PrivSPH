import csv
import pickle
import random
import numpy as np

def save_adj_list_to_csv(adj_list, filename):
    # 打开 CSV 文件准备写入
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # 遍历每个节点及其邻接节点
        for node, neighbors in enumerate(adj_list):
            # 写入节点编号及其邻接节点（集合转换为列表）
            writer.writerow([node] + list(neighbors))

def load_adj_list_from_csv(filename):
    adj_list = []

    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        
        for row in reader:
            node = int(row[0])  # 第一列是节点编号
            neighbors = set(map(int, row[1:]))  # 其余列是邻接节点
            # 确保邻接列表的长度足够
            while len(adj_list) <= node:
                adj_list.append(set())
            adj_list[node] = neighbors
    
    return adj_list

def read_edges(adj_list,edge_file):

    with open(edge_file, 'r') as fp:
     
        for _ in range(3):
            fp.readline()


        for line in fp:
            parts = line.strip().split(',')
            node1 = int(parts[0])
            node2 = int(parts[1])

            if node1 == node2:
                continue

            adj_list[node1].add(node2)
            adj_list[node2].add(node1)



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


# 将列表保存到文件中
def save_object_to_file(lst, filename):
    with open(filename, 'wb') as file:
        pickle.dump(lst, file)
    print(f"列表已保存到文件 {filename}")

# 从文件中读取列表
def load_object_from_file(filename):
    with open(filename, 'rb') as file:
        lst = pickle.load(file)
    return lst