import numpy as np
from bbox.nms import nms

from text_connector.text_connect_cfg import Config as TextCfg
from text_connector.text_proposal_connector import TextProposalConnector

class TextDetector:
    def __init__(self, DETECT_MODE="H"):
        self.mode = DETECT_MODE
        if self.mode == "H":
            self.text_proposal_connector = TextProposalConnector()
        elif self.mode == "O":
            self.text_proposal_connector = None

    def detect(self, text_proposals, scores, size):
        # 删除得分较低的proposal
        keep_inds = np.where(scores > TextCfg.TEXT_PROPOSALS_MIN_SCORE)[0]
        text_proposals, scores = text_proposals[keep_inds], scores[keep_inds]

        # 按得分排序
        sorted_indices = np.argsort(scores.ravel())[::-1]
        text_proposals, scores = text_proposals[sorted_indices], scores[sorted_indices]

        # 对proposal做nms
        keep_inds = nms(np.hstack((text_proposals, scores)), TextCfg.TEXT_PROPOSALS_NMS_THRESHOLD)
        text_proposals, scores = text_proposals[keep_inds], scores[keep_inds]

        print(text_proposals)
        print(text_proposals.shape)
        print("******************************")
        # 获取检测结果
        text_results = self.text_proposal_connector.get_text_lines(text_proposals, scores, size)
        print("text_results.shape: ",text_results.shape)
        keep_inds = self.filter_boxes(text_results)
        return text_results[keep_inds]

    def filter_boxes(self, boxes):
        heights = np.zeros((len(boxes), 1), np.float)
        widths = np.zeros((len(boxes), 1), np.float)
        scores = np.zeros((len(boxes), 1), np.float)
        index = 0
        for box in boxes:
            heights[index] = (abs(box[5] - box[1]) + abs(box[7] - box[3])) / 2.0 + 1
            widths[index] = (abs(box[2] - box[0]) + abs(box[6] - box[4])) / 2.0 + 1
            scores[index] = box[8]
            index += 1

        return np.where((widths / heights > TextCfg.MIN_RATIO)& \
                        (scores > TextCfg.TEXT_PROPOSALS_MIN_SCORE)& \
                        (widths > (TextCfg.TEXT_PROPOSALS_WIDTH * TextCfg.MIN_NUM_PROPOSALS)))[0]