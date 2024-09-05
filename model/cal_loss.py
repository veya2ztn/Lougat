from torch.nn import CrossEntropyLoss
import torch
import statistics
from scipy.ndimage import gaussian_filter
import torch
from torch import nn
import numpy as np

def box2cls(box_true, output_h, output_w, gaussian=False, sigma=1.0):
    bs_len = box_true.shape[0]
    yv = (box_true[..., 0, 1]+box_true[..., 1, 1])/2
    xv = (box_true[..., 0, 0]+box_true[..., 1, 0])/2   # 实际坐标(归一化)=[bs*length]
    yv = torch.clamp((yv*output_h).long(), 0, output_h-1)
    xv = torch.clamp((xv*output_w).long(), 0, output_w-1)

    cls_true = torch.zeros(bs_len,output_h*output_w, device=box_true.device)   # [bs*seq_len,28,21]
    indices  = yv * output_w + xv
    cls_true.scatter_(1, indices.unsqueeze(1), 1)
    cls_true = cls_true.view(bs_len, output_h, output_w)

    if gaussian:
        # 生成一个高斯分布的核，用于模糊
        # x, y = torch.meshgrid(torch.arange(0, output_w), torch.arange(0, output_h))
        xT, yT = np.meshgrid(np.arange(0, output_w), np.arange(0, output_h))
        x, y = torch.LongTensor(xT.T), torch.LongTensor(yT.T)
        kernel = torch.exp(-((x - xv)**2 + (y - yv)**2) / (2 * sigma**2))
        kernel /= kernel.sum()  # 归一化核
        cls_true = gaussian_filter(
            cls_true, sigma=sigma, mode='constant', cval=0)  # 将核应用于图像

    return cls_true

CE = CrossEntropyLoss()
def cal_loss_fast(logits,labels,prompt_pred,prompt_true,heat_map):
    BL1, vocab_size = logits.shape
    BL2, =  labels.shape
    
    BL3, D11,D12 = prompt_pred.shape
    BL4, D21,D22 = prompt_true.shape
    assert BL1 == BL2 and BL1 == BL3 and BL1 == BL4
    assert D11 == D12 and D11 == D21 and D11 == D22 and D11 == 2
    # if the attention task set properly, below code is not used
    # prompt_pred = prompt_pred[prompt_true.any(-1)] # lets omit the case that prompt_true is [0,0,0,0] # 20240704 still has many case need deal with it 
    # prompt_true = prompt_true[prompt_true.any(-1)] # lets omit the case that prompt_true is [0,0,0,0] # 20240704 still has many case need deal with it 
    diou,iou = diou_loss(pred=prompt_pred,target=prompt_true,pre5pos=None)  
    
    loss_pool =  {
        "token_loss": CE(logits, labels),
        "bbox_diou" : diou, 
        "bbox_iou"  : iou,
    }
    if heat_map is not None:
        loss_pool["hm_loss"]  = celoss(heat_map, prompt_true)
    return loss_pool
#from torchmetrics.detection import DistanceIntersectionOverUnion, IntersectionOverUnion
# DIOU = DistanceIntersectionOverUnion()
# def cal_diou_fast(prompt_pred,prompt_true):
#     labels = torch.arange(len(prompt_pred))
    
#     preds  = [{"boxes": prompt_pred,"labels": labels,}]
#     target = [{"boxes": prompt_true,"labels": labels,}]
#     #
#     return DIOU(preds, target)['diou']
# IOU = IntersectionOverUnion()
# def cal_iou_fast(prompt_pred,prompt_true):
#     labels = torch.arange(len(prompt_pred))
#     preds  = [{"boxes": prompt_pred,"labels": labels,}]
#     target = [{"boxes": prompt_true,"labels": labels,}]
#     return IOU(preds, target)['iou']


def token_loss(bs,logits,labels,):
    mask_txt = torch.ones_like(labels)
    mask_txt[torch.where(labels==1)] = 0    # 不计算pad部分，否则loss_txt学不会
    mask_math,mask_table,mask_start = torch.zeros_like(labels),torch.zeros_like(labels),torch.zeros_like(labels)
    # start tokens
    mask_txt,mask_start = mask_txt.reshape(bs,-1),mask_start.reshape(bs,-1)
    mask_txt[:,:5] = 0
    mask_start[:,:5] = 1
    mask_txt,mask_start = mask_txt.view(-1),mask_start.view(-1)
    # table and math
    '''begin_table = torch.tensor([82, 2722, 113, 10393, 115]).to(labels.device)
    end_table = torch.tensor([82, 493, 113, 10393, 115]).to(labels.device)
    begin_math1 = torch.tensor([82,30]).to(labels.device)
    end_math1 = torch.tensor([82,31]).to(labels.device)
    begin_math2 = torch.tensor([82,81]).to(labels.device)
    end_math2 = torch.tensor([82,83]).to(labels.device)
    
    idx0 = torch.nonzero(torch.eq(labels,begin_table[0])).squeeze(1)    # 匹配\
    start_math1 = (idx0+1)[torch.nonzero(torch.eq(labels[idx0+1],begin_math1[1])).squeeze(1)]   # \(
    tail_math1 = (idx0+1)[torch.nonzero(torch.eq(labels[idx0+1],end_math1[1])).squeeze(1)]      # \)
    if len(start_math1)==len(tail_math1):
        for i in range(len(start_math1)):
            if start_math1[i]<tail_math1[i]:
                mask_txt[start_math1[i]-1:tail_math1[i]+1] = 0
                mask_math[start_math1[i]-1:tail_math1[i]+1] = 1
    start_math2 = (idx0+1)[torch.nonzero(torch.eq(labels[idx0+1],begin_math2[1])).squeeze(1)]   # \[
    tail_math2 = (idx0+1)[torch.nonzero(torch.eq(labels[idx0+1],end_math2[1])).squeeze(1)]      # \]
    if len(start_math2)==len(tail_math2):
        for i in range(len(start_math2)):
            if start_math2[i]<tail_math2[i]:
                mask_txt[start_math2[i]-1:tail_math2[i]+1] = 0
                mask_math[start_math2[i]-1:tail_math2[i]+1] = 1
    start_table = (idx0+1)[torch.nonzero(torch.eq(labels[idx0+1],begin_table[1])).squeeze(1)]   # \begin
    tail_table = (idx0+1)[torch.nonzero(torch.eq(labels[idx0+1],end_table[1])).squeeze(1)]      # \end
    if len(start_table)>0 and len(tail_table)>0:
        start_table = (start_table+2)[torch.nonzero(torch.eq(labels[start_table+2],begin_table[3]))]    # \begin*table
        tail_table = (tail_table+2)[torch.nonzero(torch.eq(labels[tail_table+2],end_table[3]))]         # \end*table
        if len(start_table)>0 and len(start_table) == len(tail_table):
            for i in range(len(start_table)):
                if (labels[start_table[i]-3:start_table[i]+2] == begin_table).all() \
                    and (labels[tail_table[i]-3:tail_table[i]+2] == end_table).all() \
                    and start_table[i]<tail_table[i]:
                    mask_txt[start_table[i]-3:tail_table[i]+2] = 0
                    mask_table[start_table[i]-3:tail_table[i]+2] = 1'''

    loss_fct = CrossEntropyLoss()
    loss_txt = loss_fct(logits[mask_txt==1],labels[mask_txt==1])    # 除math\table\开头字符外的txt
    loss_math = None#loss_fct(logits[mask_math==1],labels[mask_math==1]) if labels[mask_math==1].shape[0] > 0 else None
    loss_table = None#loss_fct(logits[mask_table==1],labels[mask_table==1]) if labels[mask_table==1].shape[0] > 0 else None
    loss_start = loss_fct(logits[mask_start==1],labels[mask_start==1])
    
    return (loss_txt,loss_math ,loss_table, loss_start)

def cal_loss(bs,logits,labels,prompt_pred=None,prompt_true=None):
    import time
    begin_time = time.time()
    # loss_token
    hm_loss = diou = iou = None
    loss_txt,loss_math ,loss_table, loss_start = token_loss(bs,logits, labels)   
    
    if prompt_pred is not None:

        # loss_position
        # prompt_true:[bs,seq_len,2,2]->[bs*seq_len,4], diff: [0,0,0,0](pad)或[-1,-1,-1,-1](mask)返回0，其他返回非0，torch.where: 取非0的index
        valid_mask = torch.unique(torch.where(torch.diff(prompt_true.reshape(-1,4)))[0]) 
        assert  len(prompt_true.shape) == 4, prompt_true.shape
        seq_len = prompt_true.shape[1]
        pre5pos = torch.cat([torch.arange(0, 5) + seq_len * i for i in range(bs)]).to(valid_mask.device)
        pre5pos = torch.where(torch.isin(valid_mask, pre5pos))

        if len(valid_mask)>2*bs:   # 除</s></work>外的非mask非pad输入
            box_pred = prompt_pred[0].reshape(-1,2,2)[valid_mask]    # box
            box_true = prompt_true.reshape(-1,2,2)[valid_mask]
            diou,iou = diou_loss(pred=box_pred,target=box_true,pre5pos=pre5pos)  
            # prob = prompt_pred[1].reshape(-1)[valid_mask]            # box置信度
            hm = prompt_pred[1]                                        # [bs,seq_len,28,21]
            hm = hm.reshape(-1,hm.shape[-2],hm.shape[-1])[valid_mask]  # [bs*seq_len,28,21]
            # hm_loss = focal_loss(hm, box_true,gaussian=False)
            hm_loss = celoss(hm, box_true)
            
            
        else:   # 整个bs没有任何prompt输入
            diou,iou = None,None,None
    
    # print("time: ", time.time() - begin_time)
    
    return loss_txt,loss_math,loss_table,loss_start,diou,iou,hm_loss

    

def focal_loss(hm, box_true,gaussian=False):
    bs_len,output_h, output_w = hm.shape  # [bs,seq_len,28,21]
    target = box2cls(box_true,output_h,output_w,gaussian)
    
    pos_inds = target.eq(1).float().to(hm.device)
    neg_inds = target.lt(1).float().to(hm.device)
    hm = nn.Sigmoid()(hm)
    hm = torch.clamp(hm, 1e-6, 1-1e-6)  # clamp to 1e-6 ~ 1-1e-6
    pos_loss = torch.pow(1 - hm, 2) * torch.log(hm) * pos_inds
    neg_loss = torch.pow(hm, 2) * torch.log(1 - hm) * neg_inds 
    
    if gaussian:
        # The negative samples near the positive sample feature point have smaller weights
        neg_weights = torch.pow(1-target, 4)
        neg_loss *= neg_weights
    
    fl =  - 1/torch.numel(hm) * (pos_loss.sum() + neg_loss.sum()) 

    return fl

def celoss(hm, box_true):
    bs_len,output_h, output_w = hm.shape  # [bs,seq_len,28,21]
    pred                      = hm.view(bs_len,-1)    # [bs_len,588]
    target                    = box2cls(box_true,output_h,output_w)    # [bs*seq_len,28,21]
    target                    = target.view(bs_len,-1).max(dim=-1).indices.to(pred.device)  # [bs_len,588]-> [bs_len]
    loss_fct                  = CrossEntropyLoss()
    celoss                    = loss_fct(pred,target.view(-1))

    return celoss


def l1_loss(pred, target, mask):
    """
    Calculate l1 loss
    Args:
        pred: offset detection result
        target: offset ground truth
        mask: offset mask, only center point is 1, other place is 0

    Returns: l1 loss

    """
    expand_mask = torch.unsqueeze(mask, -1).repeat(1, 1, 1, 2)

    # Don't calculate loss in the position without ground truth.
    loss = F.l1_loss(pred * expand_mask, target * expand_mask, reduction='sum')

    loss = loss / (mask.sum() + 1e-7)

    return loss
  
def iou(pred,target,epsilon=1e-5):
    '''
    args: 
    pred/target: [bs,length,2,2]
    
    '''
    inter_x1 = torch.max(pred[:,0,0],target[:,0,0])
    inter_y1 = torch.max(pred[:,0,1],target[:,0,1])
    inter_x2 = torch.min(pred[:,1,0],target[:,1,0])
    inter_y2 = torch.min(pred[:,1,1],target[:,1,1])
    # 确保交集面积不小于0
    inter_area = torch.clamp(inter_x2-inter_x1,min=0)*torch.clamp(inter_y2-inter_y1,min=0)
    pred_area = (pred[:,1,0]-pred[:,0,0])*(pred[:,1,1]-pred[:,0,1])
    target_area = (target[:,1,0]-target[:,0,0])*(target[:,1,1]-target[:,0,1])
    union_area = pred_area + target_area - inter_area
    iou = (inter_area/(union_area+epsilon))
    

    return iou
    

def diou_loss(pred,target,pre5pos=None,prob=None,epsilon=1e-5,alpha=10,y_penalty=2):
    '''
    args: 
    pred/target: [bs,length,2,2]
    
    '''
    if len(pred.shape)  ==4:pred  =   pred.reshape(-1,2,2) # [bs*len,2,2]
    if len(target.shape)==4:target= target.reshape(-1,2,2) # [bs*len,2,2]

    iou_tensor      = iou(pred,target,epsilon) # [bs*len,1]
    
    pred,target     = pred,target
    pred_center_x   = (pred[:,1,0]+pred[:,0,0])/2
    pred_center_y   = (pred[:,1,1]+pred[:,0,1])/2
    target_center_x = (target[:,1,0]+target[:,0,0])/2
    target_center_y = (target[:,1,1]+target[:,0,1])/2
    d2 = (torch.square(pred_center_x-target_center_x)+torch.square(y_penalty*(pred_center_y-target_center_y)))
    out_x1 = torch.min(pred[:,0,0],target[:,0,0])
    out_y1 = torch.min(pred[:,0,1],target[:,0,1])
    out_x2 = torch.max(pred[:,1,0],target[:,1,0])
    out_y2 = torch.max(pred[:,1,1],target[:,1,1])
    c2 = (torch.square(out_x2-out_x1)+torch.square(out_y2-out_y1))
    diou_loss = 1-iou_tensor+alpha*d2
    # prob准确度
    if prob:
        prob_loss = 10*(prob-iou_tensor)**2
        diou_loss += prob_loss
    # 根据位置对每个token加权
    y1=target[torch.where(torch.diff(target[:,0,1],dim=0))[0]][:,0,1]   # 先定位到前后两个token的y值相同的token
    y2=target[torch.where(torch.diff(target[:,0,1],dim=0))[0]][:,1,1]
    mode = [round(h,2) for h in (y2-y1).tolist() if h>0]
    if len(mode) == 0:
        mode = 0
    else:
        mode = statistics.mode(mode)   # 本页行高：众数
    weights = torch.ones(target.shape[0],dtype=target.dtype).to(diou_loss.device)
    weights[torch.where(torch.diff(target[:,0,1],dim=0)>2*mode)[0]+1] = alpha

    # 前5个token
    if pre5pos:
        weights[pre5pos] = alpha

    diou_loss_mean = (diou_loss * weights).sum() / weights.sum()
    iou_mean = (iou_tensor * weights).sum() / weights.sum()
    
    return diou_loss_mean,iou_mean

         
    