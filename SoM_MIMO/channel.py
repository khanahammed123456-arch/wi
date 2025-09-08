import torch.nn as nn
import numpy as np
import os
import torch
import time
import math
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt

class Channel(nn.Module):
    def __init__(self, cfg):
        super(Channel, self).__init__()
        self.Nt=cfg.MODEL.MIMO.Nt
        self.Nr=cfg.MODEL.MIMO.Nr
        self.INFER_SNR=cfg.MODEL.MIMO.INFER_SNR
        self.bits=cfg.MODEL.MIMO.BITS
        self.cfg=cfg
        self.test_counter=0

    
    def channel_generator(self,device,batch_size): #生成信道和每路噪声功率
        B=batch_size
        Nt=self.Nt
        Nr=self.Nr
        SNR=self.INFER_SNR
        Ps=1
        Pn=1 / (10 ** (SNR/10))
        std=math.sqrt(Pn/Nt)
        h_real=torch.normal(mean=0.0, std=1/torch.sqrt(torch.tensor(2.0)), size=[B,Nr,Nt],device=device)
        h_imag=torch.normal(mean=0.0, std=1/torch.sqrt(torch.tensor(2.0)), size=[B,Nr,Nt],device=device)
        h = h_real + 1j* h_imag
        h = h / math.sqrt(Nt) 
        return [h,std]


    def QPSK_modulator(self,x_uint,device):
        symbols_per_int=self.bits//2
        [B, N, K] = x_uint.size()
        qpsk_values = torch.tensor([1 + 1j, -1 + 1j, 1 - 1j, -1-1j], device=device) #0b00，0b01，0b10，0b11
        x_uint_expanded = x_uint.unsqueeze(-1).expand(B, N, K, symbols_per_int).to(torch.uint8)
        shifts = torch.tensor([0, 2, 4, 6], device=device)[:symbols_per_int]
        two_low_bits = (x_uint_expanded >> shifts) & 0b11
        x_symbols = torch.take(qpsk_values, two_low_bits).view(B,N,symbols_per_int*K)
        return x_symbols
    def QPSK_demodulator(self,y_symbols,device):
        symbols_per_int=self.bits//2
        [B,N,K]=y_symbols.size()
        K//=symbols_per_int
        y_symbols_expanded=y_symbols.view(B,N,K,symbols_per_int)
        y_real_detect = torch.sign(y_symbols_expanded.real)
        y_imag_detect = torch.sign(y_symbols_expanded.imag) # Detection
        y_detect = (y_real_detect+1j*y_imag_detect).view(B,N,-1)
        # Mapping：1+j -> 00 -> 0, -1+j -> 01 -> 1, -1-j -> 11 -> 3, 1-j -> 10 -> 2 
        y_real_map= ((1-y_real_detect)/2).to(torch.uint8)
        y_imag_map= ((1-y_imag_detect)/2).to(torch.uint8)
        y_mapped= y_real_map+(y_imag_map << 1)
        tic = time.perf_counter()
        shifts=torch.tensor([0,2,4,6],device=device)[:symbols_per_int]
        y_unit=torch.sum(y_mapped << shifts,dim=-1)
        return y_unit, y_detect
    def source_to_channel(self,source_feature): # Reshape FPN to Nt streams
        shape = {}
        reshaped_tensor=[]
        for key, value in source_feature.items():
            shape[key]=value.shape
            reshape_temp=value.view(value.size(0),self.Nt,-1)
            reshaped_tensor.append(reshape_temp)
        reshaped_tensor= torch.cat(reshaped_tensor,dim=-1)
        return reshaped_tensor, shape
    def channel_to_source(self,noise_feature,shape): # Reshape Nt streams to FPN
        start_index=0
        reshape_feature={}
        for key, value in shape.items():
            temp=value[1]*value[2]*value[3]//self.Nt
            reshape_feature[key]=noise_feature[:,:,start_index:start_index+temp].reshape(value)
            start_index+=temp
        return reshape_feature

    def forward(self,x,h,std):

        # Precoding Matrices
        U,S,Vh = torch.linalg.svd(h)
        S=torch.diag_embed(S)
        S=torch.complex(S,torch.zeros_like(S))
        V = torch.conj(Vh.transpose(-1, -2))
        Uh = torch.conj(U.transpose(-1, -2))
        S_inv= torch.pinverse(S)
        device=x.device
        
        [B,N,K]=x.size()
        symbols_per_int=self.bits//2
        # Digital Tx
        x_mean=torch.mean(x) 
        x_std=torch.std(x)
        xn=(x-x_mean)/x_std #Normalize X
    
        xn_max=torch.max(xn)
        xn_min=torch.min(xn)
        quant_level=2**self.bits
        S=(xn_max-xn_min)/(quant_level-1.0)
        Z=(quant_level-1.0)-torch.round(xn_max/S)
        x_uint= torch.clamp(torch.round(xn/S)+Z,min=0.00,max=quant_level-1.0) #float32 to int
    

        x_symbols=self.QPSK_modulator(x_uint,device) #uint8 to symbols
        x_symbols_norm=(x_symbols/math.sqrt(2.0))/math.sqrt(self.Nt)

        x_tx= torch.bmm(V,x_symbols_norm) # SVD Rx-equalization precoder


        noise_real = torch.normal(mean=0.0, std=std/torch.sqrt(torch.tensor(2.0)), size=[B,N,symbols_per_int*K],device=x.get_device()).detach()
        noise_imag = torch.normal(mean=0.0, std=std/torch.sqrt(torch.tensor(2.0)), size=[B,N,symbols_per_int*K],device=x.get_device()).detach()
        noise = noise_real + 1j * noise_imag #AWGN

        x_fading= torch.bmm(h,x_tx)
        y_rx= x_fading+noise # channel fading with noise

        #Digital Rx
        y_symbols_norm= torch.bmm(S_inv,torch.bmm(Uh,y_rx)) # SVD Rx-equalization combiner


        y_symbols= y_symbols_norm*math.sqrt(2.0)*math.sqrt(self.Nt)
        y_uint, y_detect=self.QPSK_demodulator(y_symbols,device)
        yn=torch.clamp((y_uint-Z)*S, min=xn_min, max=xn_max) #uint8 to float32
        y=x_std*yn+x_mean #Denormalize Y
        # Equivalent Noise
        Eq_noise=(y-x).detach()
        Eq_noise.requires_grad=False
        y=x+Eq_noise

        return y

