#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
# pytorch

# IOU计算
# 假设box1维度为[N,4]   box2维度为[M,4]
def iou(self, box1, box2):
    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(  # 左上角的点
        box1[:, :2].unsqueeze(1).expand(N, M, 2),   # [N,2]->[N,1,2]->[N,M,2]
        box2[:, :2].unsqueeze(0).expand(N, M, 2),   # [M,2]->[1,M,2]->[N,M,2]
    )

    rb = torch.min(
        box1[:, 2:].unsqueeze(1).expand(N, M, 2),
        box2[:, 2:].unsqueeze(0).expand(N, M, 2),
    )

    wh = rb - lt  # [N,M,2]
    wh[wh < 0] = 0   # 两个box没有重叠区域
    inter = wh[:,:,0] * wh[:,:,1]   # [N,M]

    area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # (N,)
    area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # (M,)
    area1 = area1.unsqueeze(1).expand(N,M)  # (N,M)
    area2 = area2.unsqueeze(0).expand(N,M)  # (N,M)

    iou = inter / (area1+area2-inter)
    return iou


# NMS算法
    # bboxes维度为[N,4]，scores维度为[N,], 均为tensor
def nms(self, bboxes, scores, threshold=0.5):
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]
    areas = (x2-x1)*(y2-y1)   # [N,] 每个bbox的面积
    _, order = scores.sort(0, descending=True)    # 降序排列

    keep = []
    while order.numel() > 0:       # torch.numel()返回张量元素个数
        if order.numel() == 1:     # 保留框只剩一个
            i = order.item()
            keep.append(i)
            break
        else:
            i = order[0].item()    # 保留scores最大的那个框box[i]
            keep.append(i)

        # 计算box[i]与其余各框的IOU(思路很好)
        xx1 = x1[order[1:]].clamp(min=x1[i])   # [N-1,]
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])
        inter = (xx2-xx1).clamp(min=0) * (yy2-yy1).clamp(min=0)   # [N-1,]

        iou = inter / (areas[i]+areas[order[1:]]-inter)  # [N-1,]
        idx = (iou <= threshold).nonzero().squeeze() # 注意此时idx为[N-1,] 而order为[N,]
        if idx.numel() == 0:
            break
        order = order[idx+1]  # 修补索引之间的差值
    return torch.LongTensor(keep)   # Pytorch的索引值为LongTensor




import numpy as np


boxes=np.array([[100,100,210,210,0.72],
        [250,250,420,420,0.8],
        [220,220,320,330,0.92],
        [100,100,210,210,0.72],
        [230,240,325,330,0.81],
        [220,230,315,340,0.9]]) 


def py_cpu_nms(dets, thresh):
    # dets:(m,5)  thresh:scaler
    
    x1 = dets[:,0]
    y1 = dets[:,1]
    x2 = dets[:,2]
    y2 = dets[:,3]
    
    areas = (y2-y1+1) * (x2-x1+1)
    scores = dets[:,4]
    keep = []
    
    index = scores.argsort()[::-1]
    
    while index.size >0:

        i = index[0]       # every time the first is the biggst, and add it directly
        keep.append(i)
        
        x11 = np.maximum(x1[i], x1[index[1:]])    # calculate the points of overlap 
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        
        w = np.maximum(0, x22-x11+1)    # the weights of overlap
        h = np.maximum(0, y22-y11+1)    # the height of overlap
       
        overlaps = w*h
        
        ious = overlaps / (areas[i]+areas[index[1:]] - overlaps)
        
        idx = np.where(ious<=thresh)[0]
        print(idx, ious)
        
        index = index[idx+1]   # because index start from 1
        
    return keep
        

import matplotlib.pyplot as plt
def plot_bbox(dets, c='k'):
    
    x1 = dets[:,0]
    y1 = dets[:,1]
    x2 = dets[:,2]
    y2 = dets[:,3]
    
    
    plt.plot([x1,x2], [y1,y1], c)
    plt.plot([x1,x1], [y1,y2], c)
    plt.plot([x1,x2], [y2,y2], c)
    plt.plot([x2,x2], [y1,y2], c)
    plt.title("after nms")
    plt.show()

plot_bbox(boxes,'k')   # before nms

keep = py_cpu_nms(boxes, thresh=0.7)
plot_bbox(boxes[keep], 'r')# after nms