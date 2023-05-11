import dgl.nn.pytorch
import torch as th
from torch import nn,einsum
from dgl import function as fn
from dgl.nn import pytorch as pt
device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

class MAMFGAT(nn.Module):
    def __init__(self, args):
        super(MAMFGAT, self).__init__()
        self.args = args
        self.lin_m=nn.Linear(args.miRNA_number,args.in_feats,bias=False)
        self.lin_d=nn.Linear(args.disease_number,args.in_feats,bias=False)

        self.gcn_mm_1 = pt.GATConv(args.miRNA_number,128,num_heads=10,feat_drop=0.2)
        self.gcn_mm_2 = pt.GATConv(1280, 64, num_heads=10, feat_drop=0.2)
        self.gcn_mm_3 = pt.GATConv(640,args.out_feats,num_heads=1,feat_drop=0.2)
        self.res_l_1 = pt.nn.Linear(args.miRNA_number,64)

        self.gcn_dd_1 = pt.GATConv(args.disease_number, 128, num_heads=10,feat_drop=0.2)
        self.gcn_dd_2 = pt.GATConv(1280, 64, num_heads=10, feat_drop=0.2)
        self.gcn_dd_3 = pt.GATConv(640,args.out_feats,num_heads=1,feat_drop=0.2)
        self.res_l_2 = pt.nn.Linear(args.disease_number, 64)

        self.gcn_md_1 = pt.GATConv(args.in_feats,128,num_heads=10,feat_drop=0.2, allow_zero_in_degree=True)
        self.gcn_md_2 = pt.GATConv(1280, 64, num_heads=10, feat_drop=0.2, allow_zero_in_degree=True)
        self.gcn_md_3 = pt.GATConv(640,args.out_feats,num_heads=1,feat_drop=0.2, allow_zero_in_degree=True)
        self.res_l_3 = pt.nn.Linear(args.in_feats,args.out_feats)



        self.elu = nn.ELU()
        self.mlp = nn.Sequential()
        self.dropout = nn.Dropout(args.dropout)
        in_feat = 2 * args.out_feats
        for idx, out_feat in enumerate(args.mlp):
            if idx==0:
                self.mlp.add_module(str(idx), nn.Linear(in_feat, out_feat))
                self.mlp.add_module('elu', nn.ELU())
                self.mlp.add_module('dropout', nn.Dropout(p=0.2))
                in_feat = out_feat
            else:
                self.mlp.add_module(str(idx), nn.Linear(in_feat, out_feat))
                self.mlp.add_module('sigmoid', nn.Sigmoid())
                self.mlp.add_module('dropout',nn.Dropout(p=0.2))
                in_feat = out_feat

        self.fuse_weight_1 = nn.Parameter(th.FloatTensor(1), requires_grad=True)
        self.fuse_weight_2 = nn.Parameter(th.FloatTensor(1), requires_grad=True)

        self.fuse_weight_1.data.fill_(0.5)
        self.fuse_weight_2.data.fill_(0.5)


        self.fuse_weight_m = nn.Parameter(th.FloatTensor(1), requires_grad=True)
        self.fuse_weight_d = nn.Parameter(th.FloatTensor(1), requires_grad=True)
        self.fuse_weight_md = nn.Parameter(th.FloatTensor(1), requires_grad=True)

        self.fuse_weight_m.data.fill_(0.5)
        self.fuse_weight_d.data.fill_(0.5)
        self.fuse_weight_md.data.fill_(0.5)



    def forward(self, mm_graph, dd_graph, md_graph, miRNA, disease, samples):
        mm_graph=mm_graph.to(device)
        dd_graph=dd_graph.to(device)
        md_graph=md_graph.to(device)
        miRNA=miRNA.to(device)
        disease=disease.to(device)

        res_mi=self.elu(self.res_l_1(miRNA))
        res_di=self.elu(self.res_l_2(disease))

        md=th.cat((self.lin_m(miRNA), self.lin_d(disease)), dim=0)
        #md=self.dropout(self.relu(md))
        res_md=self.elu(self.res_l_3(md))

        res_mm = res_md[:self.args.miRNA_number, :]
        res_dd = res_md[self.args.miRNA_number:, :]
        res_mmdd = th.cat((res_mm,res_dd),dim=0)

        emb_mm_sim_1 = self.elu(self.gcn_mm_1(mm_graph, miRNA))
        emb_mm_sim_1 = emb_mm_sim_1.view(emb_mm_sim_1.size(0), -1)
        emb_mm_sim_2 = self.elu(self.gcn_mm_2(mm_graph,emb_mm_sim_1))
        emb_mm_sim_2 = emb_mm_sim_2.view(emb_mm_sim_2.size(0), -1)
        emb_mm_sim_3 = self.elu(self.gcn_mm_3(mm_graph, emb_mm_sim_2))
        emb_mm_sim_3 = emb_mm_sim_3.view(emb_mm_sim_3.size(0), -1)
        emb_mm_sim_3 = self.fuse_weight_m*emb_mm_sim_3 + (1-self.fuse_weight_m)*res_mi

        emb_dd_sim_1 = self.elu(self.gcn_dd_1(dd_graph, disease))
        emb_dd_sim_1 = emb_dd_sim_1.view(emb_dd_sim_1.size(0), -1)
        emb_dd_sim_2 = self.elu(self.gcn_dd_2(dd_graph,emb_dd_sim_1))
        emb_dd_sim_2 = emb_dd_sim_2.view(emb_dd_sim_2.size(0), -1)
        emb_dd_sim_3 = self.elu(self.gcn_dd_3(dd_graph, emb_dd_sim_2))
        emb_dd_sim_3 = emb_dd_sim_3.view(emb_dd_sim_3.size(0), -1)
        emb_dd_sim_3 = self.fuse_weight_d*emb_dd_sim_3 + (1-self.fuse_weight_d)*res_di

        emb_ass_1 = self.elu(self.gcn_md_1(md_graph, th.cat((self.lin_m(miRNA),self.lin_d(disease)), dim=0)))
        emb_ass_1 = emb_ass_1.view(emb_ass_1.size(0), -1)
        emb_ass_2 = self.elu(self.gcn_md_2(md_graph,emb_ass_1 ))
        emb_ass_2 = emb_ass_2.view(emb_ass_2.size(0), -1)
        emb_ass_3 = self.elu(self.gcn_md_3(md_graph, emb_ass_2))
        emb_ass_3 = emb_ass_3.view(emb_ass_3.size(0), -1)
        emb_ass_3 = self.fuse_weight_md*emb_ass_3 + (1-self.fuse_weight_md)*res_mmdd

        emb_mm_ass = emb_ass_3[:self.args.miRNA_number, :]
        emb_dd_ass = emb_ass_3[self.args.miRNA_number:, :]


        emb_mm=self.fuse_weight_1*emb_mm_sim_3+(1-self.fuse_weight_1)*emb_mm_ass
        emb_dd=self.fuse_weight_2*emb_dd_sim_3+(1-self.fuse_weight_2)*emb_dd_ass

        emb = th.cat((emb_mm[samples[:, 0]], emb_dd[samples[:, 1]]), dim=1)
        result=self.mlp(emb)
        return result,emb_mm_sim_3,emb_mm_ass,emb_dd_sim_3,emb_dd_ass

