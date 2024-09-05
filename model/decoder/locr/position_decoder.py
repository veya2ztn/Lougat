import torch
from torch import nn
import torch.nn.functional as F
import statistics
import numpy as np
from scipy.ndimage import gaussian_filter
from torch.nn import CrossEntropyLoss
from torchvision.transforms.functional import resize, rotate
from transformers.utils import logging
logger = logging.get_logger(__name__)
class PositionDecoder(nn.Module):
    def __init__(self,
                 decoder_attention_heads,
                 decoder_layers,
                 input_dim=588, 
                 hidden_dim=256, 
                 output_dim=5, 
                 num_layers=3,
                 bn_momentum=0.1,
                 image_size=[896,672],
                 scale_factor=2,
                 decay_rate=1,
                 use_image_bias=False):
        super().__init__()
        # import os
        # if 'decay' not in os.environ or os.environ['decay'] is None or os.environ['decay'] == '':
        #     self.decay_rate = 1.0
        # else:
        #     self.decay_rate = float(os.environ['decay'])
        self.decay_rate = decay_rate
        if decay_rate == 1: print('use position decay == 1, please note it is not the best setting for generation')
        #self.image_size = image_size
        self.in_channels = decoder_attention_heads*decoder_layers   # 16*4=64
        self.middle_channels = 16
        self.upscale_factor = scale_factor
        self.use_image_bias = use_image_bias
        # heatmap: 预测中心点所在网格，最大值为预测点，值为置信度
        self.cls_head = nn.Sequential(
            nn.Conv2d(self.in_channels, self.middle_channels, kernel_size=3, padding=1, bias=False),   # in_channels, out_channels, kernel_size, stride=1, padding=(1,0)
            # nn.BatchNorm2d(self.middle_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=self.upscale_factor),
            nn.Conv2d(self.middle_channels, self.middle_channels, kernel_size=3, padding=1, bias=False),   # in_channels, out_channels, kernel_size, stride=1, padding=(1,0)
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.middle_channels, momentum=bn_momentum),
            nn.Conv2d(self.middle_channels, self.middle_channels, kernel_size=3, padding=1, bias=False),   # in_channels, out_channels, kernel_size, stride=1, padding=(1,0)
            # nn.BatchNorm2d(self.middle_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.middle_channels, 1, kernel_size=1, stride=1, padding=0),
            # nn.Sigmoid()
        )
        # 每个点对应bbox的wh
        self.wh_head = nn.Sequential(
            nn.Conv2d(self.in_channels, self.middle_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(self.middle_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            
            nn.UpsamplingBilinear2d(scale_factor=self.upscale_factor),

            nn.Conv2d(self.middle_channels, self.middle_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(self.middle_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.middle_channels, momentum=bn_momentum),
            nn.Conv2d(self.middle_channels, self.middle_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(self.middle_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.middle_channels, 2, kernel_size=1, stride=1, padding=0),
            )    
        # center point offset
        self.offset_head = nn.Sequential(
            nn.Conv2d(self.in_channels, self.middle_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(self.middle_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            
            nn.UpsamplingBilinear2d(scale_factor=self.upscale_factor),
            
            nn.Conv2d(self.middle_channels, self.middle_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(self.middle_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.middle_channels, momentum=bn_momentum),
            nn.Conv2d(self.middle_channels, self.middle_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(self.middle_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.middle_channels, 2, kernel_size=1, stride=1, padding=0),
            )    
        
        self.do_we_warn = False
    
    def forward(self, heatmap, attention_valid_mask, full_prompt_in = None, image_tensors = None):
        

        if full_prompt_in is not None:
            logger.warning_once("full_prompt_in is not used in PositionDecoder.forward anymore")
        if image_tensors is not None:
            logger.warning_once("image_tensors is not used in PositionDecoder.forward anymore")
        decoder_layers,bs,num_heads,input_len,encoder_len = heatmap.shape  # [4,bs,16,len,588]
        if attention_valid_mask is None:
            attention_valid_mask = torch.ones(bs, input_len, device=heatmap.device)
        assert attention_valid_mask.ndim == 2
        attention_valid_mask = attention_valid_mask.bool()
        heatmap  = heatmap.permute(1,3,4,0,2).reshape(bs,input_len,encoder_len,-1)# [bs,len,588,4,16] -> [bs,len,588,64]
        oheatmap = heatmap.reshape(bs,input_len, decoder_layers*num_heads, 28,21)  # <<<<---- 这里做了一个mixing 
        
        heatmap = oheatmap[attention_valid_mask] ## (BL, 64, 28, 21) 
        hm      = self.cls_head(heatmap)        # (BL, 1, 28, 21) 
        wh      = self.wh_head(heatmap)         # (BL, 2, 28, 21) 
        offset  = self.offset_head(heatmap)     # (BL, 2, 28, 21) 

        bs_len, _, output_h, output_w = hm.shape
        _, indices = torch.max(hm.view(bs_len, -1),dim=-1)    # [BL,1,28,21]->[BL]
        indices_x  = indices   % (output_w)                     # [BL]
        indices_y  = indices   // (output_w)                    # [BL] 
        indices_bs_len = torch.arange(bs_len,device=heatmap.device)

        xv = indices_x.detach() / (output_w)           # [BL]
        yv = indices_y.detach() / (output_h)           # [BL]

        xv = xv.to(oheatmap)
        yv = yv.to(oheatmap)
        if self.training:
            xv += offset[indices_bs_len, 0, indices_y, indices_x]
            yv += offset[indices_bs_len, 1, indices_y, indices_x]
        else:
            xv += offset[indices_bs_len, 0, indices_y, indices_x].clamp(0,1.0/output_w)
            yv += offset[indices_bs_len, 1, indices_y, indices_x].clamp(0,1.0/output_h)

        half_w = wh[indices_bs_len, 0, indices_y, indices_x] / 2
        half_h = wh[indices_bs_len, 1, indices_y, indices_x] / 2

        x1        = torch.zeros(bs, input_len, device=hm.device, dtype=hm.dtype)
        y1        = torch.zeros(bs, input_len, device=hm.device, dtype=hm.dtype)
        x2        = torch.zeros(bs, input_len, device=hm.device, dtype=hm.dtype)
        y2        = torch.zeros(bs, input_len, device=hm.device, dtype=hm.dtype)
        output_hm = torch.zeros(bs, input_len, output_h, output_w, device=hm.device, dtype=hm.dtype)

        x1[attention_valid_mask]        = (xv - half_w)
        y1[attention_valid_mask]        = (yv - half_h)
        x2[attention_valid_mask]        = (xv + half_w)
        y2[attention_valid_mask]        = (yv + half_h)
        output_hm[attention_valid_mask] = hm.view(bs,input_len,output_h,output_w)

        return [x1,y1,x2,y2],output_hm

        
    def forward_old(self, heatmap, attention_valid_mask, full_prompt_in = None, image_tensors = None):
        '''
        args:
            heatmap:[bs,16,len(input_ids),588]
            full_prompt_in: 用于惩罚重复生成，在use cache时长度与input_len不同
        '''

        decoder_layers,bs,num_heads,input_len,encoder_len = heatmap.shape  # [4,bs,16,len,588]
        heatmap = heatmap.permute(1,3,4,0,2).reshape(bs,input_len,encoder_len,-1)  # [bs,len,588,4,16] -> [bs,len,588,64]
        heatmap = heatmap.reshape(bs*input_len, decoder_layers*num_heads, 28,21)   # <<<<---- 这里做了一个mixing 输入：(bs,C_in​,H_in,W_in)=[bs*seq_len,4*16,28,21]     
        hm      = self.cls_head(heatmap)         # 输出：(bs,C_out​,H_out,W_out)=[bs*seq_len,1,28,21] 56,42
        wh      = self.wh_head(heatmap)          # [bs*seq_len,2,28,21], 56,42
        offset  = self.offset_head(heatmap)     # [bs*seq_len,2,28,21] 56,42

        bs_len, _, output_h, output_w = hm.shape
        #print(hm.shape) # 8190, 1, 56, 42
        hm = hm.view([bs, input_len,-1]) # (2, 4095, 56*42)
        if self.decay_rate < 1:
            if full_prompt_in is None:  ### you should compute the 2D position embedding (a heat map) in CPU since it is not runtime necessary.
                # print("Not Decay")
                hmdecay = torch.zeros_like(hm)
            else:
                # print(full_prompt_in.shape)
                cls_in = box2cls(full_prompt_in.reshape(-1, 2,2), output_h, output_w).reshape(bs, -1, output_h,output_w)
                cnt_cls_in = cls_in.cumsum(dim = 1)[:, -input_len:]
                # print("Input Box Count")
                # print(cnt_cls_in)
                hmdecay = cnt_cls_in.reshape(hm.shape).square() * torch.log(torch.tensor(self.decay_rate))
            
            hm += hmdecay.to(hm.device)

        if not self.use_image_bias:
            if image_tensors is not None: 
                if not self.do_we_warn:
                    self.do_we_warn = True
                    print( "you are not supposed to use image tensor as image bias in position decoder")
        else:
            assert image_tensors is not None
            split_size = image_tensors.shape[-1] // output_w
            image_gray = image_tensors.mean(dim=-3)
            image_hsplit = torch.stack(image_gray.split(split_size,dim=-2),dim=1)
            image_split  = torch.stack(image_hsplit.split(split_size,dim=-1),dim=2)  # B, output_h, output_w, box_h, box_w
            image_split_std = image_split.std(dim=(-1,-2))
            image_split_std[...,:-1]  += image_split_std[...,1:] # shift left
            image_split_std[...,:-1]  += image_split_std[...,1:] # shift left
            image_split_std[...,-1,-1] = 1 #0.01/20   # end token
            bias = torch.log(20*image_split_std.clamp(1e-6,0.05)).view([bs,1,-1])
            #print(image_split_std.shape)
            # print(hm.shape)
            # print(image_tensors.shape)
            # print(image_split_std.shape) # (2, 56, 42)
            # print(bias.shape) # (2, 56, 42)
            # print("======================")
            # raise
            
            hm = hm + bias
        
        

        _, indices = torch.max(hm.view([bs_len,-1]),dim=-1)  # [bs*seq_len,1,28,21]->[bs*seq_len]
        indices_x = indices % (output_w)      # [bs*seq_len]
        indices_y = indices // (output_w)     # [bs*seq_len] 
        xv = indices_x.float() / (output_w)   # [bs*seq_len]
        yv = indices_y.float() / (output_h)

        indices_bs_len = torch.arange(bs_len,device=heatmap.device)
        
        if self.training:
            xv += offset[indices_bs_len, 0, indices_y, indices_x]
            yv += offset[indices_bs_len, 1, indices_y, indices_x]
        else:
            xv += offset[indices_bs_len, 0, indices_y, indices_x].clamp(0,1.0/output_w)
            yv += offset[indices_bs_len, 1, indices_y, indices_x].clamp(0,1.0/output_h)

        half_w = wh[indices_bs_len, 0, indices_y, indices_x] / 2
        half_h = wh[indices_bs_len, 1, indices_y, indices_x] / 2
        x1 = (xv - half_w).view(bs,input_len)
        y1 = (yv - half_h).view(bs,input_len)
        x2 = (xv + half_w).view(bs,input_len)
        y2 = (yv + half_h).view(bs, input_len)
     
        
        # attention_valid_mask=0 -> pred=[[0,0],[0,0]](pad坐标)
        if attention_valid_mask is None:
            hm = hm.reshape(bs,input_len,output_h,output_w)  # [bs*seq_len,1,28,21] -> [bs,seq_len,28,21]
            return [x1,y1,x2,y2],hm
        else:
            if attention_valid_mask.shape[1]==4095:   # train/validation with whole sentence ?????? why 4095????? 
                #attention_valid_mask = torch.cat((attention_valid_mask[:,1:],torch.zeros([bs,1]).to(heatmap.device)),dim=1) # attention_mask对应input_ids，对应prompt_true需要向后移动一位
                attention_valid_mask = torch.nn.functional.pad(attention_valid_mask, (0,1))[:,1:]
            x1 = torch.mul(x1,attention_valid_mask) # [bs,seq_len]
            x2 = torch.mul(x2,attention_valid_mask)
            y1 = torch.mul(y1,attention_valid_mask)
            y2 = torch.mul(y2,attention_valid_mask)
        
            hm = hm.reshape(bs,input_len,output_h,output_w)  # [bs*seq_len,1,28,21] -> [bs,seq_len,28,21]
            hm = torch.mul(hm,attention_valid_mask[..., None, None])
    
            return [x1,y1,x2,y2],hm


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


def box2cls_old(box_true,output_h,output_w,gaussian=False,sigma=1.0):
    bs_len = box_true.shape[0]
    cls_true = torch.zeros(bs_len,output_h,output_w, device=box_true.device)   # [bs*seq_len,28,21]
    yv,xv = (box_true[...,0,1]+box_true[...,1,1])/2,(box_true[...,0,0]+box_true[...,1,0])/2,   # 实际坐标(归一化)=[bs*length]
    yv,xv = torch.clamp(yv, 1e-6, 1-1e-6),torch.clamp(xv, 1e-6, 1-1e-6)
    yv,xv = torch.floor(yv*output_h).long(),torch.floor(xv*output_w).long()     # 格子indices
    yv[yv==output_h] = output_h-1
    xv[xv==output_w] = output_w-1
    indices_bs_len = torch.arange(bs_len, device=box_true.device)
    cls_true[indices_bs_len,yv,xv] = 1
    if gaussian:
        # 生成一个高斯分布的核，用于模糊
        #x, y = torch.meshgrid(torch.arange(0, output_w), torch.arange(0, output_h))
        xT, yT = np.meshgrid(np.arange(0, output_w), np.arange(0, output_h))
        x, y = torch.LongTensor(xT.T), torch.LongTensor(yT.T)
        kernel = torch.exp(-((x - xv)**2 + (y - yv)**2) / (2 * sigma**2))
        kernel /= kernel.sum()# 归一化核
        cls_true = gaussian_filter(cls_true, sigma=sigma, mode='constant', cval=0)# 将核应用于图像
    
    return cls_true
    




   
        
    


    
