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
    def __init__(self,en_num,re_num,en_dim=50,r_dim=50,margin=1,learn_rate=0.01)
        super(TransR, self).__init__()
        self.entity_num = entity_num
        self.relation_num = relation_num
        self.ent_dim = ent_dim
        self.rel_dim = rel_dim
        self.margin = margin
        
        
        self.ent_embedding = torch.nn.Embedding(num_embeddings=self.entity_num,
                                                          embedding_dim=self.ent_dim).cuda()
        self.ent_projection=torch.nn.Embedding(num_embeddings=self.entity_num, 
                                                          embedding_dim=self.ent_dim).cuda()
        
        self.rel_embedding = torch.nn.Embedding(num_embeddings=self.relation_num,
                                                           embedding_dim=self.rel_dim).cuda()
        self.rel_projection = torch.nn.Embedding(num_embeddings= self.relation_num,
                                                           embedding_dim=self.ent_dim*self.rel_dim).cuda()
        
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
		nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
		nn.init.xavier_uniform_(self.ent_transfer.weight.data)
		nn.init.xavier_uniform_(self.rel_transfer.weight.data)
  
        if margin != None:
			self.margin = nn.Parameter(torch.Tensor([margin]))
			self.margin.requires_grad = False
			self.margin_flag = True
		else:
			self.margin_flag = False
        
        self.loss_F = nn.MarginRankingLoss(self.margin, reduction="mean").cuda()

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