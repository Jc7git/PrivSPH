import matplotlib.pyplot as plt
import numpy as np
import datetime
from matplotlib import rcParams, ticker, font_manager
from pylab import mpl
from matplotlib.ticker import FixedLocator



# 加载自定义字体文件 'times.ttf'，用于绘图中的字体设置
timesnr = font_manager.FontProperties(fname='./times.ttf')

# 设置图例的字体属性，包括字体家族、样式和大小
legendpro = font_manager.FontProperties(family='Times New Roman',
                                        style='oblique',
                                        size=28) #19

# 配置绘图中的字体参数、Y轴刻度等
config = {
    "font.family": 'serif',  # 设置字体家族为 serif
    "font.size": 24, #24  # 设置字体大小为 24
    "mathtext.fontset": 'stix',  # 设置数学文本字体为 STIX
    "font.serif": ['Times New Roman'],  # 设置 serif 字体为 Times New Roman
    # "font.style":'oblique',  # 注释掉的字体样式设置
    "figure.figsize": (6.5, 5.0),  # 设置图形的大小为 6.5x5.0 英寸
    # "ytick.major.width": 0.1,  # 注释掉的 Y 轴主刻度线宽度设置
    # "ytick.minor.width": 0.09,  # 注释掉的 Y 轴次刻度线宽度设置
    # "ytick.major.size": 2,  # 注释掉的 Y 轴主刻度线长度设置
}

# 更新全局的 rcParams 配置，应用上述设置
rcParams.update(config)

# 设置 matplotlib 的 rcParams，确保负号正常显示
mpl.rcParams['axes.unicode_minus'] = False

# 设置默认文本的字体大小
plt.rc('font', size=34)  # 控制默认文本大小为 34
plt.rc('axes', titlesize=34)  # 设置坐标轴标题的字体大小为 34
plt.rc('axes', labelsize=24)  # 设置 X 和 Y 轴标签的字体大小为 24
plt.rc('xtick', labelsize=24)  # 设置 X 轴刻度标签的字体大小为 24
plt.rc('ytick', labelsize=24)  # 设置 Y 轴刻度标签的字体大小为 24
plt.rc('legend', fontsize=18)  # 设置图例的字体大小为 18
plt.rc('figure', titlesize=15)  # 设置图形标题的字体大小为 15

# 注释掉的代码，用于设置字体家族为 Times New Roman 或黑体
# plt.rc('font', family='Times New Roman')
# mpl.rcParams['font.sans-serif'] = ['SimHei']  # 黑体

# 定义一个绘图类
class Plot:
    def __init__(self, file_path):
        self.data_name = ""  # 数据名称
        self.task_type = ""  # 任务类型
        self.task_name ="" # 任务名称
        self.x_name = ""  # X 轴名称
        self.x_axis = []  # X 轴数据
        self.res_dict = {}  # 存储结果的字典

        # 定义线条样式和颜色
        self.line_style = ['-', '--', '-.', 'solid', 'dashed', 'dashdot', 'dotted']
        # self.line_style = ['-','-','-','-','-','-','-','-','-']
        # self.colors = ['#1f77b4', '#d62728', '#9467bd', '#8c564b',  '#7f7f7f', '#bcbd22', '#17becf']
        self.colors = plt.cm.tab10.colors[:7]  # 取前 7 种颜色
        self.markers = ['o', 's','D','*','x','p','h'] 

        # 从文件中读取数据并解析
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                key = line.strip().split(':')[0]
                if key == "x_axis":
                    self.x_axis = eval(line.strip().split(':')[1])
                    continue
                elif key == "task_type":
                    self.task_type = line.strip().split(':')[1]
                    continue
                elif key == "task_name":
                    self.task_name = line.strip().split(':')[1]
                    continue
                elif key == "data_name":
                    self.data_name = line.strip().split(':')[1]
                    continue
                elif key == "x_name":
                    self.x_name = line.strip().split(':')[1]
                    continue
                if "[" in line:
                    if key not in self.res_dict.keys():
                        self.res_dict[key] = []
                        self.res_dict[key].append(eval(line.strip().split(':')[1]))
                    else:
                        self.res_dict[key].append((eval(line.strip().split(':')[1])))

    # 绘制实验结果的函数
    def plot_exp(self, method_list):
        res_mean = {}  # 存储均值结果
        res_error = {}  # 存储误差结果
        res_length = len(self.x_axis)  # X 轴数据的长度

        # 计算每个键对应的均值和误差
        for key in self.res_dict.keys():
            res_mean[key] = [0] * res_length
            res_error[key] = [0] * res_length
            res_key = self.res_dict[key]
            for i in range(res_length):
                t = res_key[i]
                res_error[key][i] = np.std(t)  # 计算标准差
                res_mean[key][i] = np.mean(t)  # 计算均值

        res_mean1 = {}  # 存储均值结果
        res_error1 = {}  # 存储误差结果
        xs = []
        for i in range(0,res_length,1): xs.append(self.x_axis[i])

        for key in self.res_dict.keys():
            res_mean1[key] = []
            res_error1[key] = [] 
            res_key = self.res_dict[key]
            for i in range(0,res_length,1):
                t = res_key[i]
                res_error1[key].append(np.std(t)*100 )  # 计算标准差
                res_mean1[key].append( np.mean(t)*100)  # 计算均值
                # res_error1[key].append(np.std(t))  # 计算标准差
                # res_mean1[key].append( np.mean(t))  # 计算均值
        self.x_axis=xs
        res_mean = res_mean1
        res_error = res_error1

        # 根据任务类型绘制不同的指标
        match self.task_type:
            case "subgraph_count":
                self.exp_which_metric('L2Loss', res_mean, res_error, method_list)
                # self.exp_which_metric('RE', res_mean, res_error, method_list)
                # self.exp_legend("L2Loss",method_list)
            case "subgraph_histogram":
                # self.exp_which_metric('MSE', res_mean, res_error, method_list)
                self.exp_which_metric('NL1', res_mean, res_error, method_list)
                # self.exp_legend("NL1",method_list)

    # 绘制特定指标的图形
    def exp_which_metric(self, metric, res_mean, res_error, method_list):
        _, ax = plt.subplots(constrained_layout=True)  # 创建子图

        # 遍历结果字典，绘制对应的曲线
        # 正无穷大
        y_min = float('inf')
        # 负无穷小
        y_max = float('-inf')

        for key in res_mean.keys():
            if metric == key.split()[1]:
                method_name = key.split()[0]
                if method_name not in method_list:
                    continue
                if self.x_name == "epsilon":
                # # 根据方法名称设置线条样式和标签
                    if method_name == 'Intial':
                        line_style_index = 1
                        label = "Intial"
                        if self.task_name =="wedge_histogram":
                            label = r'$Intial_{\wedge}$'
                        else:
                            label = r'$Intial_{\Delta}$'
                    elif method_name == 'shuffle':
                        line_style_index = 0
                        label = "WShuffle"
                        if self.task_name =="wedge_histogram":
                            label = r'$WShuffle_{\wedge}$'
                        else:
                            label = r'$WShuffle_{\Delta}$'
                    elif method_name == 'DDP':
                        line_style_index = 3
                        if self.task_name =="wedge_histogram":
                            label = r'$PrivSPH_{\wedge}$'
                        else:
                            label = r'$PrivSPH_{\Delta}$'
                    elif method_name == 'DDP_noAm':
                        line_style_index = 4
                        if self.task_name =="wedge_histogram":
                            label = r'$PrivSPH_{\wedge}^*$'
                        else:
                            label = r'$PrivSPH_{\Delta}^*$'
                    elif method_name == 'DDP_1':
                        line_style_index = 2
                        if self.task_name =="wedge_histogram":
                            label = r'$PrivSPH1_{\wedge}$'
                        else:
                            label = r'$PrivSPH1_{\Delta}$'
                    elif method_name == 'ARR':
                        line_style_index = 5
                        label = "ARR"
                    else:
                        line_style_index = 6
                        label = method_name.upper()

                # elif self.x_name == "k":
                #     if method_name == '0.8':
                #         line_style_index = 1
                #     elif method_name == '1.0':
                #         line_style_index = 2
                #     elif method_name == '1.2':
                #         line_style_index = 3
                #     elif method_name == '1.4':
                #         line_style_index = 4
                        
                elif self.x_name == "k":
                    if method_name == '0.8':
                        line_style_index = 1
                    elif method_name == '1.0':
                        line_style_index = 2
                    elif method_name == '1.2':
                        line_style_index = 3
                    elif method_name == '1.4':
                        line_style_index = 4
                    label = r'$\varepsilon$'+' = '+method_name
                    #r'$\varepsilon$'+'='
                    


                y_min = min(min(res_mean[key]),y_min)
                y_max = max(max(res_mean[key]),y_max)

                # 绘制曲线
                ax.plot(self.x_axis, res_mean[key], label=label, zorder=line_style_index,c=self.colors[line_style_index], marker = self.markers[line_style_index],linestyle=self.line_style[line_style_index])

        # 设置 Y 轴为对数尺度
        # ax.set_yscale('log')
        ax.set_box_aspect(0.7)  # 设置图形的宽高比
       
        # ax.set_ylim(0.01, 1.2)
        ax.annotate(
            r'$\times 10^{-2}$',
            xy=(0, 1),                  # 定位在 y 轴顶部 (x=0, y=1)
            xycoords='axes fraction',    # 使用轴坐标系（0~1）
            xytext=(5, 0),           # 微调位置（像素偏移）
            textcoords='offset points',  # 偏移单位（像素）
            ha='right',                 # 水平右对齐
            va='bottom',                # 垂直底部对齐
            fontsize=plt.rcParams['ytick.labelsize']  # 使用刻度标签字号
        )

        # 设置 X 和 Y 轴的刻度标签字体大小和样式
        plt.xticks(self.x_axis, fontsize=30, fontstyle='normal') #24
        plt.yticks(fontsize=30, fontstyle='normal') #24
       

        # 根据指标名称设置 Y 轴标签
        if metric == 'RE':
            metric = 'RE'
        elif metric == 'MSE':
            metric = 'MSE'
        elif metric == 'NL1':
            metric = 'NL1'
        elif metric == 'L2Loss':
            metric = 'MSE'

        ylab = ''
        if self.x_name == 'epsilon':
            plt.xlabel(r'$\varepsilon$', fontsize=28, fontstyle='oblique') #28
            ylab = r'' + metric 
        elif self.x_name == 'k':
            plt.xlabel(r'$k$', fontsize=28, fontstyle='oblique')
            ylab = r'' + metric 

        plt.ylabel(ylab, fontsize=28) #28

        # 设置 X 轴的范围
        plt.xlim((self.x_axis[0] - 0.05, self.x_axis[-1] + 0.05))
        if self.task_name =="2-triangele_count" or self.task_name =="cycle_4_count":
            ax.margins(y=0.16)
        else:
            ax.margins(y=0.09)
        
        ax.xaxis.set_major_locator(FixedLocator([1, 5, 9, 13, 17,21, 25]))
        # ax.xaxis.set_major_locator(FixedLocator([1,4,7,10,13,16,19,22,25,28]))
        # 设置图例的位置和样式
        # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.22), ncol=3,frameon=False, fancybox=False, shadow=False, fontsize=14)

        # 添加网格线
        plt.grid(linestyle='--')

        # 保存图形为 PDF 文件   + str(datetime.datetime.now().strftime('%Y-%m-%d_%H_%M')) 
        plt.savefig(
            './figures/' + self.data_name + '_'+self.task_name +'_'+ metric + '_'+ self.x_name + '_exp_' + str(datetime.datetime.now().strftime('%Y-%m-%d_%H_%M')) +'.pdf')
        plt.show()  # 显示图形

    # 绘制图例的函数
    def exp_legend(self, metric, method_list):
        handle = []  # 存储图例的句柄
        labels = []  # 存储图例的标签

        # 遍历结果字典，生成图例
        for key in self.res_dict.keys():
            if metric == key.split()[1]:
                method_name = key.split()[0]
                if method_name not in method_list:
                    continue

                # # 根据方法名称设置线条样式和标签
                # if method_name == 'CDP':
                #     line_style_index = 1
                #     label = "CDP"
                # elif method_name == 'shuffle':
                #     line_style_index = 0
                #     label = "WShuffle"
                # elif method_name == 'DDP':
                #     line_style_index = 3
                #     label = "SCWS"
                # elif method_name == 'DDP_noAm':
                #     line_style_index = 4
                #     label = "SCWS-noAm"
                # elif method_name == 'DDP_1':
                #     line_style_index = 2
                #     label = "SCWS-1"
                # elif method_name == 'ARR':
                #     line_style_index = 5
                #     label = "ARR"
                # else:
                #     line_style_index = 6
                #     label = method_name.upper()

                if self.x_name == "epsilon":
                # # 根据方法名称设置线条样式和标签
                    if method_name == 'Intial':
                        line_style_index = 1
                        label = "Intial"
                        if self.task_name =="wedge_histogram":
                            label = r'$Intial_{\wedge}$'
                        else:
                            label = r'$Intial_{\Delta}$'
                    elif method_name == 'shuffle':
                        line_style_index = 0
                        label = "WShuffle"
                        if self.task_name =="wedge_histogram":
                            label = r'$WShuffle_{\wedge}$'
                        else:
                            label = r'$WShuffle_{\Delta}$'
                    elif method_name == 'DDP':
                        line_style_index = 3
                        if self.task_name =="wedge_histogram":
                            label = r'$PrivSPH_{\wedge}$'
                        else:
                            label = r'$PrivSPH_{\Delta}$'
                    elif method_name == 'DDP_noAm':
                        line_style_index = 4
                        if self.task_name =="wedge_histogram":
                            label = r'$PrivSPH_{\wedge}^*$'
                        else:
                            label = r'$PrivSPH_{\Delta}^*$'
                    elif method_name == 'DDP_1':
                        line_style_index = 2
                        if self.task_name =="wedge_histogram":
                            label = r'$PrivSPH1_{\wedge}$'
                        else:
                            label = r'$PrivSPH1_{\Delta}$'
                    elif method_name == 'ARR':
                        line_style_index = 5
                        label = "ARR"
                    else:
                        line_style_index = 6
                        label = method_name.upper()

                elif self.x_name == "k":
                    if method_name == '0.8':
                        line_style_index = 1
                    elif method_name == '1.0':
                        line_style_index = 2
                    elif method_name == '1.2':
                        line_style_index = 3
                    elif method_name == '1.4':
                        line_style_index = 4
                        
                    label = r'$\varepsilon$'+' = '+method_name
                    #r'$\varepsilon$'+'='

                # 绘制图例
                l1, = plt.plot(self.x_axis, [0]*len(self.x_axis), c=self.colors[line_style_index], marker = self.markers[line_style_index],linestyle=self.line_style[line_style_index], label=key)
                handle.append(l1)
                labels.append(label)

        # 创建子图并设置图例
        _, ax = plt.subplots(figsize=(20, 0.8))
        ax.legend(fontsize=46,handles=handle, labels=labels, mode='expand', ncol=len(labels), frameon=False,borderaxespad=0,prop={'weight': 'bold', 'size': 35})
        ax.axis('off')  # 去掉坐标轴刻度

        # 保存图例为 PDF 文件
        plt.savefig(
            './figures/' + self.data_name + '_'+self.task_name +'_'+ metric + '_'+ self.x_name + '_legend_' + str(datetime.datetime.now().strftime('%Y-%m-%d_%H_%M')) + '.pdf')
        plt.show()  # 显示图例


if __name__ == "__main__":
    plot_epsilon = Plot('./results/soc-twitter-higgs50-330_wedge_histogram_k.txt') #11
    # plot_epsilon = Plot('./results/soc-twitter-higgs50-330_triangle_histogram_k.txt') #11
    # plot_epsilon = Plot('./results/graph_Gplus50_300_triangle_histogram_k.txt') # 11 -2
    # plot_epsilon = Plot('./results/graph_Gplus50_300_wedge_histogram_k.txt') #10 -2
    # plot_epsilon = Plot('./results/graph_Gplus50_300_triangle_count_k.txt') #
    # plot_epsilon = Plot('./results/graph_Gplus50_300_2-triangele_count_k.txt') #
    # plot_epsilon = Plot('./results/graph_Gplus50_300_cycle_4_count_k.txt') #
    # plot_epsilon = Plot('./results/soc-twitter-higgs50-330_triangle_count_k.txt') #
    # plot_epsilon = Plot('./results/soc-twitter-higgs50-330_cycle_4_count_k.txt') #
    # plot_epsilon = Plot('./results/soc-twitter-higgs50-330_2-triangele_count_k.txt') #
    # plot_epsilon = Plot('./results/graph_nemeth_190_2-triangele_count_k.txt') #
    # plot_epsilon = Plot('./results/graph_nemeth_190_cycle_4_count_k.txt') #
    # plot_epsilon = Plot('./results/graph_nemeth_190_triangle_count_k.txt') #
    # plot_epsilon = Plot('./results/graph_crankseg50_270_2-triangele_count_k.txt') #
    # plot_epsilon = Plot('./results/graph_crankseg50_270_cycle_4_count_k.txt') #
    # plot_epsilon = Plot('./results/graph_crankseg50_270_triangle_count_k.txt') #
    plot_epsilon.plot_exp(['0.8','1.0','1.2','1.4'])
    # plot_epsilon = Plot('./results/graph_crankseg50_270_triangle_histogram_epsilon.txt')
    # plot_epsilon = Plot('./results/graph_crankseg50_270_wedge_histogram_epsilon.txt')
    # plot_epsilon = Plot('./results/graph_Gplus50_300_wedge_histogram_epsilon.txt')
    # plot_epsilon = Plot('./results/graph_Gplus50_300_triangle_histogram_epsilon.txt')
    # plot_epsilon = Plot('./results/soc-twitter-higgs50-330_wedge_histogram_epsilon.txt') #0.015
    # plot_epsilon = Plot('./results/soc-twitter-higgs50-330_triangle_histogram_epsilon.txt') #0.01
    # plot_epsilon.plot_exp(["DDP","DDP_1"])
    # plot_epsilon.plot_exp(["DDP","DDP_noAm"])
    # plot_epsilon.plot_exp(["DDP","Intial"])
    # plot_epsilon.plot_exp(["DDP","DDP_noAm","DDP_1","Intial"])
    #,"DDP_noAm","Intial"