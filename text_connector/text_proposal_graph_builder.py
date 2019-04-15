import numpy as np

from text_connector.graph import Graph
from text_connector.text_connect_cfg import Config as TextCfg

class TextProposalGraphBuilder:
    """
        Build Text proposals into a graph
    """

    # 给Index框找到x右侧邻近y相似的框索引
    def get_successions(self, index):
        box = self.text_proposals[index]
        results = []
        # x坐标在(box[0]+1)-(box[0]+50)遍历，即x坐标不同的框框，查看是否有连续的
        for left in range(int(box[0])+1, min(int(box[0]) + TextCfg.MAX_HORIZONTAL_GAP+1, self.im_size[1])):
            adj_box_indices = self.boxes_table[left] # 每一列的框框取出
            for adj_box_index in adj_box_indices:
                if self.meet_v_iou(adj_box_index, index): # 检查在y轴上的相似度
                    results.append(adj_box_index) # 可能会找到多个
            if len(results) != 0: # 一有results就返回
                return results

        return results

    # 给Index框找到x左侧邻近y相似的框索引
    def get_precursors(self, index):
        box = self.text_proposals[index]
        results = []
        for left in range(int(box[0])-1, max(int(box[0] - TextCfg.MAX_HORIZONTAL_GAP),0)-1, -1):
            adj_box_indices = self.boxes_table[left]
            for adj_box_index in adj_box_indices:
                if self.meet_v_iou(adj_box_index, index):
                    results.append(adj_box_index)
            if len(results) != 0:
                return results
        return results

    # 连续节点左侧确认
    def is_succession_node(self, index, succession_index):
        # 得到succession_index左侧连续的框框index
        precursors = self.get_precursors(succession_index)
        # 如果index是succession_index得到的左侧连续的框框中分数最高的，判定为连续节点！
        if self.scores[index] >= np.max(self.scores[precursors]):
            return True
        return False

    # 判断y轴上的相似度
    def meet_v_iou(self, index1, index2):
        def overlaps_v(index1, index2):
            # h1 = self.heights[index1]
            # h2 = self.heights[index2]
            # intersection
            y0 = max(self.text_proposals[index2][1], self.text_proposals[index1][1])
            y1 = min(self.text_proposals[index2][3], self.text_proposals[index1][3])
            # union
            a0 = min(self.text_proposals[index2][1], self.text_proposals[index1][1])
            a1 = max(self.text_proposals[index2][3], self.text_proposals[index1][3])
            return max(0, y1-y0+1) / max(0.5, a1-a0+1)

        def size_similarity(index1, index2):
            h1 = self.heights[index1]
            h2 = self.heights[index2]
            return min(h1,h2) / max(h1,h2)

        return overlaps_v(index1, index2) >= TextCfg.MIN_V_OVERLAPS and \
                size_similarity(index1, index2) >= TextCfg.MIN_SIZE_SIM

    def build_graph(self, text_proposals, scores, im_size):
        self.text_proposals = text_proposals
        self.scores = scores
        self.im_size = im_size
        self.heights = text_proposals[:, 3] - text_proposals[:, 1] + 1

        boxes_table = [[] for _ in range(self.im_size[1])]
        print(im_size)
        for index, box in enumerate(text_proposals):
            boxes_table[int(box[0])].append(index)  # 图，邻接链表，索引为每个左上x坐标(0,16,32,...)
        self.boxes_table = boxes_table

        graph = np.zeros((text_proposals.shape[0], text_proposals.shape[0]), np.bool)

        for index, box in enumerate(text_proposals):
            # 遍历每个proposal，找到在它右边连续的proposal索引
            successions = self.get_successions(index)
            if len(successions) == 0:
                continue
            # 选择score最大的几个proposal索引
            succession_index = successions[np.argmax(scores[successions])]
            #
            if self.is_succession_node(index, succession_index):
                # NOTE: a box can have multiple successions, if they have same score
                graph[index, succession_index] = True
        print("build_graph: ok ")
        return Graph(graph)
