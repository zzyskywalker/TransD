import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


entity2id = {}
relation2id = {}
relation_tph = {}#每个头实体对应的尾实体
relation_hpt = {}#bern

def traindata_loader(file1,file2,file3,file4):
    
    entity_df=pd.read_csv(file1,header=None,sep='\t')
    relation_df=pd.read_csv(file2,header=None,sep='\t')
    train_df=pd.read_csv(file3,header=None,sep='\t')
    vaild_df=pd.read_csv(file4,header=None,sep='\t')
    
    for x,y in entity_df.values:
        entity2id[x] = y
    for x,y in relation_df.values:
        relation2id[x]=y
        
    #print(relation2id)
    entity_set = set()
    relation_set = set()
    train_triple_list = []#承载三元组
    relation_head = {}#记录每个关系对应的头实体\尾实体的个数
    relation_tail = {}
    
    train_list=train_df.values.tolist()
    
    for i in train_list:
        h_=int(entity2id[i[0]])
        r_=int(relation2id[i[1]])
        t_=int(entity2id[i[2]])
        
        entity_set.add(h_)
        entity_set.add(t_)
        relation_set.add(r_)
        
        train_triple_list.append([h_,r_,t_])
        #计算每个关系对应的头实体\尾实体的个数
        if r_ in relation_head:
            if h_ in relation_head[r_]:
                relation_head[r_][h_] += 1
            else:
                relation_head[r_][h_] = 1
        else:
            relation_head[r_] = {}
            relation_head[r_][h_] = 1

        if r_ in relation_tail:
            if t_ in relation_tail[r_]:
                relation_tail[r_][t_] += 1
            else:
                relation_tail[r_][t_] = 1
        else:
            relation_tail[r_] = {}
            relation_tail[r_][t_] = 1
    
    #print(relation_head)
    #计算
    for r_ in relation_head:
        sum1, sum2 = 0, 0
        for head in relation_head[r_]:
            sum1 += 1
            sum2 += relation_head[r_][head]
        tph = sum2 / sum1
        relation_tph[r_] = tph

    for r_ in relation_tail:
        sum1, sum2 = 0, 0
        for tail in relation_tail[r_]:
            sum1 += 1
            sum2 += relation_tail[r_][tail]
        hpt = sum2 / sum1
        relation_hpt[r_] = hpt
    
    vaild_list=vaild_df.values.tolist()
    
    vaild_triple_list=[]
    for i in vaild_list:
        h_=int(entity2id[i[0]])
        r_=int(relation2id[i[1]])
        t_=int(entity2id[i[2]])
        
        
        vaild_triple_list.append([h_,r_,t_])
        
    return entity_set,relation_set,train_triple_list,vaild_triple_list


class TransD(nn.Module):
    def __init__(self,en_num,re_num,en_dim=50,r_dim=50,norm=2,margin=1,learn_rate=0.01)
        super(TransR, self).__init__()
        self.entity_num = entity_num
        self.relation_num = relation_num
        self.ent_dim = ent_dim
        self.rel_dim = rel_dim
        self.margin = margin
        self.norm=norm
        
        self.ent_embedding = torch.nn.Embedding(num_embeddings=self.entity_num,
                                                          embedding_dim=self.ent_dim).cuda()
        self.ent_transfer=torch.nn.Embedding(num_embeddings=self.entity_num, 
                                                          embedding_dim=self.ent_dim).cuda()
        
        self.rel_embedding = torch.nn.Embedding(num_embeddings=self.relation_num,
                                                           embedding_dim=self.rel_dim).cuda()
        self.rel_transfer = torch.nn.Embedding(num_embeddings= self.relation_num,
                                                           embedding_dim=self.rel_dim).cuda()
        
        self.identity_matrix=torch.eye(ent_dim,rel_dim).cuda()#单位矩阵
        
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)#随机初始化参数
		nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
		nn.init.xavier_uniform_(self.ent_transfer.weight.data)
		nn.init.xavier_uniform_(self.rel_transfer.weight.data)
        self.loss_F = nn.MarginRankingLoss(self.margin, reduction="mean").cuda()
        """
        if margin != None:
			self.margin = nn.Parameter(torch.Tensor([margin]))
			self.margin.requires_grad = False
			self.margin_flag = True
		else:
			self.margin_flag = False
        """
   
    def _resize(self, tensor, axis, size):
		shape = tensor.size()
		osize = shape[axis]
		if osize == size:
			return tensor
		if (osize > size):
			return torch.narrow(tensor, axis, 0, size)
		paddings = []
		for i in range(len(shape)):
			if i == axis:
				paddings = [0, size - osize] + paddings
			else:
				paddings = [0, 0] + paddings
		print (paddings)
		return F.pad(tensor, paddings = paddings, mode = "constant", value = 0)

	def _calc(self, h, t, r):#f(h,r,t)
		
  
  
        head = self.ent_embedding(h)
        head_transfer=self.ent_transfer(h)
        rel = self.rel_embedding(r)
        rel_transfer = self.rel_transfer(r)
        tail = self.ent_embedding(t)
        tail_transfer=self.ent_transfer

        h_=self._transfer(head,head_transfer,rel_transfer)
        t_=self._transfer(tail,tail_transfer,rel_transfer)

        h = F.normalize(h_, 2, -1)#归一化
		r = F.normalize(r, 2, -1)
		t = F.normalize(t_, 2, -1)

		score = h + r - t
		score = torch.norm(score, self.norm, -1).flatten()#l2范数，然后压扁
		return score
    def _transfer(self,ent,ent_transfer,rel_transfer):#从hp，rp转换的h_,r_
        ent=F.normalize(ent,2,-1)
        ent_transfer=torch.unsqueeze(ent_transfer,dim=0)
        rel_transfer=torch.unsqueeze(rel_transfer,dim=1)
        
        matrix=torch.mm(rel_transfer,ent_transfer)+self.identity_matrix
        
        ent=torch.unsqueeze(ent,dim=1)
        h_=torch.mm(matrix,ent)
        
        return torch.squeeze(h_)
    """      
	def _transfer(self, e, e_transfer, r_transfer):
		if e.shape[0] != r_transfer.shape[0]:
			e = e.view(-1, r_transfer.shape[0], e.shape[-1])
			e_transfer = e_transfer.view(-1, r_transfer.shape[0], e_transfer.shape[-1])
			r_transfer = r_transfer.view(-1, r_transfer.shape[0], r_transfer.shape[-1])
			e = F.normalize(
				self._resize(e, -1, r_transfer.size()[-1]) + torch.sum(e * e_transfer, -1, True) * r_transfer,
				p = 2, 
				dim = -1
			)			
			return e.view(-1, e.shape[-1])
		else:
			return F.normalize(
				self._resize(e, -1, r_transfer.size()[-1]) + torch.sum(e * e_transfer, -1, True) * r_transfer,
				p = 2, 
				dim = -1
			)
    """
	def forward(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		mode = data['mode']
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)
		h_transfer = self.ent_transfer(batch_h)
		t_transfer = self.ent_transfer(batch_t)
		r_transfer = self.rel_transfer(batch_r)
		h = self._transfer(h, h_transfer, r_transfer)
		t = self._transfer(t, t_transfer, r_transfer)
		score = self._calc(h ,t, r, mode)
		if self.margin_flag:
			return self.margin - score
		else:
			return score

	def regularization(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)
		h_transfer = self.ent_transfer(batch_h)
		t_transfer = self.ent_transfer(batch_t)
		r_transfer = self.rel_transfer(batch_r)
		regul = (torch.mean(h ** 2) + 
				 torch.mean(t ** 2) + 
				 torch.mean(r ** 2) + 
				 torch.mean(h_transfer ** 2) + 
				 torch.mean(t_transfer ** 2) + 
				 torch.mean(r_transfer ** 2)) / 6
		return regul

	def predict(self, data):
		score = self.forward(data)
		if self.margin_flag:
			score = self.margin - score
			return score.cpu().data.numpy()
		else:
			return score.cpu().data.numpy()

        #self.loss_F = nn.MarginRankingLoss(self.margin, reduction="mean").cuda()

class TransD_Train():
    def __init__(self,entity_set,relation_set,train,vaild=None,e_dim=50,r_dim=50,margin=1,learn_rate=0.01,batch_size=200,norm=L2):
        self.entity=entity_set
        self.relation=relation_set
        self.train=train
        self.vaild=vaild
        self.e_dim=e_dim
        self.r_dim=r_dim
        self.margin=margin
        self.batch_size=batch_size
        self.norm=norm
    
if __name__ == "__main__":
    entity,relation,train_triple,vaild_triple=traindata_loader("FB15k\\entity2id.txt","FB15k\\relation2id.txt",
                                                 "FB15k\\freebase_mtr100_mte100-train.txt","FB15k\\freebase_mtr100_mte100-valid.txt")
    #print(entity2id)