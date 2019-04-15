import numpy as np

class Graph(object):
    def __init__(self, graph):
        self.graph = graph

    def sub_graphs_connected(self):
        sub_graphs = []
        # 遍历所有proposal索引
        for index in range(self.graph.shape[0]):
            if not self.graph[:, index].any() and self.graph[index, :].any():
                print("%")
                v = index
                sub_graphs.append([v])
                while self.graph[v,:].any():
                    v = np.where(self.graph[v,:])[0][0] # 得到一行连续的索引
                    sub_graphs[-1].append(v) # 加入子图
        return sub_graphs