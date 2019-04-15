import numpy as np

from text_connector.text_proposal_graph_builder import TextProposalGraphBuilder

class TextProposalConnector:
    def __init__(self):
        self.graph_builder = TextProposalGraphBuilder()

    def group_text_proposals(self, text_proposals, scores, im_size):
        graph = self.graph_builder.build_graph(text_proposals, scores, im_size)
        print("TextProposalConnector: ", graph.sub_graphs_connected())
        return graph.sub_graphs_connected()

    # 传入x1的序列,和y1的序列，做一个多项式拟合
    # 由拟合函数f输出f(x1),f(x2)
    def fit_y(self, X, Y, x1, x2):
        assert len(X) != 0, "proposals can't be None"

        if np.sum(X == X[0]) == len(X):
            return Y[0], Y[0]
        p = np.poly1d(np.polyfit(X, Y, 1))
        return p(x1), p(x2)

    def get_text_lines(self, text_proposals, scores, im_size):
        # tp = text proposal
        # 得到list: 全是[p1,p2,...]，含义是连续的（一行）多个proposal的索引
        tp_groups = self.group_text_proposals(text_proposals, scores, im_size)
        print("tp_groups: ", tp_groups)
        text_lines = np.zeros((len(tp_groups), 5), np.float32)

        for index, tp_indices in enumerate(tp_groups):
            text_line_boxes = text_proposals[list(tp_indices)]

            x0 = np.min(text_line_boxes[:,0]) # 选取一段连续的proposal中x最小值
            x1 = np.max(text_line_boxes[:,2]) # 选取一段连续的proposal中x最大值

            offset = (text_line_boxes[0,2] - text_line_boxes[0,0]) * 0.5
            # 整行的proposal的左上角的y确定
            lt_y, rt_y = self.fit_y(text_line_boxes[:,0], text_line_boxes[:,1], x0+offset, x1-offset)
            # 整行的proposal的右下角的y确定
            lb_y, rb_y = self.fit_y(text_line_boxes[:,0], text_line_boxes[:,3], x0+offset, x1-offset)

            score =scores[list(tp_indices)].sum() / float(len(tp_indices))

            text_lines[index, 0] = x0
            text_lines[index, 1] = min(lt_y, rt_y)
            text_lines[index, 2] = x1
            text_lines[index, 3] = max(lb_y, rb_y)
            text_lines[index, 4] = score

        text_lines = clip_boxes(text_lines, im_size)

        text_recs = np.zeros((len(text_lines), 9), np.float)
        index = 0
        for line in text_lines:
            xmin, ymin, xmax, ymax = line[0], line[1], line[2], line[3]
            text_recs[index, 0] = xmin
            text_recs[index, 1] = ymin
            text_recs[index, 2] = xmax
            text_recs[index, 3] = ymin
            text_recs[index, 4] = xmax
            text_recs[index, 5] = ymax
            text_recs[index, 6] = xmin
            text_recs[index, 7] = ymax
            text_recs[index, 8] = line[4]
            index = index + 1

        return text_recs


def threshold(coords, min_, max_):
    return np.maximum(np.minimum(coords, max_), min_)

def clip_boxes(boxes, im_shape):

    boxes[:,0::2] = threshold(boxes[:,0::2], 0, im_shape[1]-1)
    boxes[:,1::2] = threshold(boxes[:,1::2], 0, im_shape[0]-1)
    return boxes
