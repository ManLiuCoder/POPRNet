import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modeling.backbone import resnet101_features
from models.modeling import utils

from os.path import join
import pickle
#  add import for parts localization by lm begin
from models.modeling.anchors import generate_default_anchor_maps, hard_nms
import numpy as np
from torch.autograd import Variable
import copy
Norm =nn.LayerNorm



base_architecture_to_features = {
    'resnet101': resnet101_features,
}
def truncated_normal_(tensor, mean=0, std=0.09):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor
class ProposalNet(nn.Module):
    def __init__(self):
        super(ProposalNet, self).__init__()
        self.down1 = nn.Conv2d(2048, 128, 3, 1, 1)
        self.down2 = nn.Conv2d(128, 128, 3, 2, 1)
        self.down3 = nn.Conv2d(128, 128, 3, 2, 1)
        self.ReLU = nn.ReLU()
        self.tidy1 = nn.Conv2d(128, 6, 1, 1, 0)
        self.tidy2 = nn.Conv2d(128, 6, 1, 1, 0)
        self.tidy3 = nn.Conv2d(128, 9, 1, 1, 0)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.downsample1 = nn.Conv2d(128, 128, 3, 2, 1)
        self.downsample2 = nn.Conv2d(128, 128, 3, 2, 1)
        self.sigmoid = nn.Sigmoid()

 

    def unsample1(self, x, y):
        _, _, H, W = x.size()
        t = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=True)
        t = torch.mean(t, dim=1, keepdim=True)
        t = self.sigmoid(t)
        return t

    def forward(self, x):
        batch_size = x.size(0)
        d1 = self.ReLU(self.down1(x))
        d2 = self.ReLU(self.down2(d1))
        d3 = self.ReLU(self.down3(d2))

        d2_1 = self.unsample1(d1, d2)
        e2_1 = d2_1 * d1
        d1_final = d1 - e2_1
        d2_2 = d2 + self.downsample1(e2_1)

        d3_1 = self.unsample1(d2_2, d3)
        e3_1 = d3_1 * d2_2
        d2_final = d2_2 - e3_1
        d3_final = d3 + self.downsample1(e3_1)


        t1 = self.tidy1(d1_final).view(batch_size, -1)
        t2 = self.tidy2(d2_final).view(batch_size, -1)
        t3 = self.tidy3(d3_final).view(batch_size, -1)
        return torch.cat((t1, t2, t3), dim=1)

class Classifier(nn.Module):
    def __init__(self, in_panel, out_panel, bias=False):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(in_panel, out_panel, bias=bias)

    def forward(self, input):
        logit = self.fc(input)
        if logit.dim()==1:
            logit =logit.unsqueeze(0)
        return logit

class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, reduction // 8)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(reduction // 8, reduction)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x



class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()
        self.logsoft = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        b = self.softmax(x) * self.logsoft(x)
        return -1.0 * b.mean(1)

class SFTLayer(nn.Module):
    # 参考SFTGAN
    def __init__(self):
        super(SFTLayer, self).__init__()
        self.SFT_scale_fc0 = nn.Linear(312, 156)
        self.SFT_scale_fc1 = nn.Linear(156, 312)
        self.SFT_shift_fc0 = nn.Linear(312, 156)
        self.SFT_shift_fc1 = nn.Linear(156, 312)
        self.ReLu = nn.ReLU()
    def forward(self, x):
        # x[0]: fea; x[1]: cond
        scale = self.SFT_scale_fc1(self.ReLu(self.SFT_scale_fc0(x[1])))
        shift = self.SFT_shift_fc1(self.ReLu(self.SFT_shift_fc0(x[1])))
        return x[0] * (scale + 1) + shift

class ResBlock_SFT(nn.Module):
    def __init__(self):
        super(ResBlock_SFT, self).__init__()
        self.sft0 = SFTLayer()
        self.fc0 = nn.Linear(312, 312)
        self.sft1 = SFTLayer()
        self.fc1 = nn.Linear(312, 312)

    def forward(self, x):
        fea = self.sft0(x)
        fea = F.relu(self.fc0(fea), inplace=True)
        fea = self.sft1((fea, x[1]))
        fea = self.fc1(fea)
        return x[0] + fea



class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):

        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, reduction, 1, padding=0, bias=True),  # channel // reduction
                nn.ReLU(inplace=True),
                nn.Conv2d(reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
    
class Normqkv(nn.Module):
    def __init__(self,dim):
        super(Normqkv, self).__init__()
        self.norm_q, self.norm_k, self.norm_v,self.norm = Norm(dim), Norm(dim), Norm(dim),Norm(dim)
        self.to_q = nn.Linear(dim, dim, True)
        self.to_k = nn.Linear(dim, dim, True)
        self.to_v = nn.Linear(dim, dim, True)
        self.proj = nn.Linear(dim, dim, True)

    def forward(self, q,kv):
        q, k, v = self.norm_q(q), self.norm_k(kv), self.norm_v(kv)
        q, k, v = self.to_q(q), self.to_k(k), self.to_v(v)
        return q,k,v
class Normqkv_self(nn.Module):
    def __init__(self,dim):
        super(Normqkv_self, self).__init__()
        self.norm_q, self.norm_k, self.norm_v,self.norm = Norm(dim), Norm(dim), Norm(dim),Norm(dim)
        self.to_q = nn.Linear(dim, dim, True)
        self.to_k = nn.Linear(dim, dim, True)
        self.to_v = nn.Linear(dim, dim, True)
        self.proj = nn.Linear(dim, dim, True)

    def forward(self, q):
        q, k, v = self.norm_q(q), self.norm_k(q), self.norm_v(q)
        q, k, v = self.to_q(q), self.to_k(k), self.to_v(v)
        return q,k,v

class attention_p(nn.Module):
    def __init__(self,dim):
        super(attention_p, self).__init__()
        self.proj = nn.Linear(dim, dim, True)
        self.normqkv = Normqkv(dim)

    def forward(self, q,kv):
        q, k, v = self.normqkv(q,kv)
        q=q.unsqueeze(-1) 
        attn = torch.einsum('bijk,bkv->bijv', q, k) 
        attn = F.softmax(attn, dim=-1) 
        out = torch.einsum('bijk,bmk->bijm', attn, v).squeeze(-1)  
        out = self.proj(out)
        return out


class encoder(nn.Module):
    def __init__(self,dim,num_parts):
        super(encoder, self).__init__()
        self.norm =Norm(dim)
        self.attention = attention_p(dim)
        self.linear=nn.Conv1d(num_parts,num_parts,kernel_size=1, bias=False)
    def reason(self,x):
        out = self.norm(x)
        out = self.linear(out)
        return x+out
    def forward(self, q,kv):
        q_out = self.attention(q,kv) 
        parts = q + q_out
        out = self.reason(parts)
        return out


class attention_w(nn.Module):
    def __init__(self, dim):
        super(attention_w, self).__init__()
        self.proj = nn.Linear(dim, dim, True)
        self.normqkv = Normqkv(dim)

    def forward(self, q, kv):
        q, k, v = self.normqkv(q, kv)
        attn = torch.einsum('bik,bjk->bji', q, k)  
        attn = attn.transpose(2,1) 
        attn = F.softmax(attn, dim=-1) 
        out = torch.einsum('bik,bkj->bij', attn, v).squeeze(-1)  
        out = self.proj(out)
        return out
class attention_s(nn.Module):
    def __init__(self, dim):
        super(attention_s, self).__init__()
        self.proj = nn.Linear(dim, dim, True)
        self.normqkv = Normqkv_self(dim)

    def forward(self, q):
        q, k, v = self.normqkv(q)
        q = q.unsqueeze(-1)  
        attn = torch.einsum('bijk,bkv->bijv', q, k)  
        attn = F.softmax(attn, dim=-1) 
        out = torch.einsum('bijk,bmk->bijm', attn, v).squeeze(-1)  
        out = self.proj(out)
        return out
class decoder(nn.Module):
    def __init__(self,dim):
        super(decoder, self).__init__()
        self.attention = attention_w(dim)
        self.sf_atten= attention_s(dim)
    def forward(self, q,kv):
        q_out = self.attention(q,kv) 
        whole = q + q_out
        out = self.sf_atten(whole) + whole
        return out
class POPRNet(nn.Module):
    def __init__(self, res101, img_size, c, w, h,
                 attritube_num, cls_num, ucls_num, attr_group, w2v,
                 scale=20.0, device=None):

        super(POPRNet, self).__init__()
        self.device = device

        self.img_size = img_size
        self.attritube_num = attritube_num

        self.feat_channel = c
        self.feat_channel_shallow = 1024
        self.feat_w = w
        self.feat_h = h

        self.ucls_num = ucls_num
        self.scls_num = cls_num - ucls_num
        self.attr_group = attr_group
        self.dim = 312
        self.w2v_att = torch.from_numpy(w2v).float().to(self.device)  # 312 * 300
        assert self.w2v_att.shape[0] == self.attritube_num

        if scale<=0:
            self.scale = nn.Parameter(torch.ones(1) * 20.0)
        else:
            self.scale = nn.Parameter(torch.tensor(scale), requires_grad=False)


        self.backbone = res101
        self.conv4 = nn.Sequential(*list(res101.children())[:-1])
        self.conv5 = nn.Sequential(*list(res101.children())[-1])

        self.conv4_parts = copy.deepcopy(self.conv4)
        self.conv5_parts = copy.deepcopy(self.conv5)
        self.convD = nn.Conv2d(self.feat_channel_shallow+312, self.feat_channel_shallow, kernel_size=1)
        self.W = nn.Parameter(truncated_normal_(torch.empty(self.w2v_att.shape[1], self.feat_channel_shallow)),
                              requires_grad=True)  # 300 * 2048
        self.FC = nn.Linear(self.w2v_att.shape[1], self.feat_channel_shallow)  # 300 * 2048
        self.topN_train = 4
        self.topN = 4
        self.batch = 10
       
        self.V1 = nn.Parameter(truncated_normal_(torch.empty(self.feat_channel, self.attritube_num)), requires_grad=True)

        self.V2 = nn.Parameter(truncated_normal_(torch.empty(self.feat_channel, self.attritube_num)),
                               requires_grad=True)  # 1024,312


        # loss
        self.Reg_loss = nn.MSELoss()
        self.CLS_loss = nn.CrossEntropyLoss()

        _, edge_anchors, _ = generate_default_anchor_maps()
        self.proposal_net = ProposalNet()
        self.pad_side = 224
        self.edge_anchors = (edge_anchors + 224).astype(np.int)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.cls_gate = nn.Sequential(Classifier(2048, 1024, bias=True), Classifier(1024, 2, bias=True))
        self.gate = copy.deepcopy(self.backbone)
        self.attr_cls1 = Classifier(self.topN * 312, 312, bias=True)
        self.attr_cls2 =  Classifier(312, 312, bias=True)
        self.ca = CALayer(312,28)

        self.encoder = encoder(self.dim,self.topN)
        self.decoder = decoder(self.dim)
        self.attrSFT = ResBlock_SFT() #SFTLayer()
        self.attr_cond = nn.Parameter(truncated_normal_(torch.empty(1, 312)),
                               requires_grad=True)


    def conv_features(self, x):
        '''
        the feature input to prototype layer
        '''
        x = self.backbone(x)
        return x

    def compute_score(self, gs_feat,seen_att,att_all,flag):
        N, C = gs_feat.shape
        if flag == True:
            # no-rank
            gs_feat = gs_feat.view(self.batch, -1)
            if N != self.batch:
                # for parts else for raw features
                gs_feat = self.attr_cls1(gs_feat)

        gs_feat = self.attr_cls2(gs_feat)
        gs_feat_norm = torch.norm(gs_feat, p=2, dim = 1).unsqueeze(1).expand_as(gs_feat)
        gs_feat_normalized = gs_feat.div(gs_feat_norm + 1e-5)

        temp_norm = torch.norm(att_all, p=2, dim=1).unsqueeze(1).expand_as(att_all)  # [150,312]
        seen_att_normalized = att_all.div(temp_norm + 1e-5)

        cos_dist = torch.einsum('bd,nd->bn', gs_feat_normalized, seen_att_normalized)  # [8,150]
        score_o = cos_dist

        d, _ = seen_att.shape
        if d == 200:
            score = score_o*self.scale
        if d == 150:
            score = score_o[:, :150]*self.scale
            if self.training:
                ### compute mean,std
                mean1 = score_o[:, :150].mean(1)
                std1 = score_o[:, :150].std(1)
                mean2 = score_o[:, -50:].mean(1)
                std2 = score_o[:, -50:].std(1)
                mean_score = F.relu(mean1 - mean2 + 0.005)
                std_score = F.relu(std1 - std2)
                if flag == False:
                    mean_score = mean_score.view(self.batch,self.topN).mean(0)
                    std_score = std_score.view(self.batch,self.topN).mean(0)
                    mean_loss = torch.sum(mean_score)+torch.sum(std_score)
                else:
                    mean_loss = mean_score.mean(0) + std_score.mean(0)
                return score, mean_loss*self.scale
        if d == 50:
            score = score_o[:, -50:]*self.scale
        return score, _

    def base_module(self, atten_attr,global_feat, seen_att,att_all,flag):
        N, C = global_feat.shape
        global_feat = global_feat
        gs_feat = torch.einsum('bc,cd->bd', global_feat, self.V1) 
        gs_feat = F.softmax(atten_attr,dim=-1) * gs_feat + gs_feat
        score,a = self.compute_score(gs_feat,seen_att,att_all,flag)
        return gs_feat,score, a

    def base_module_p(self,global_feat, seen_att,att_all,flag):
        N, C = global_feat.shape
        global_feat = global_feat
        gs_feat = torch.einsum('bc,cd->bd', global_feat, self.V2)  
        score,a = self.compute_score(gs_feat,seen_att,att_all,flag)
        return gs_feat,score, a


    def VSIMModule(self, x):

        N, C, W, H = x.shape  # [8,2048,14,14]
        x = x.reshape(N, C, W * H)  
        query = torch.einsum('lw,wv->lv', self.w2v_att, self.W) 
        atten_map = torch.einsum('lv,bvr->blr', query, x) 
        intensity = self.ca(atten_map.view(N, -1, W, H))
        intensity = F.max_pool2d(intensity, kernel_size=(W,H)).view(N, -1)
        return intensity


    def cross_learn(self, x_o,x_p):
        x_o = x_o.unsqueeze(1) #[b,1,312] 
        x_p = x_p.view(self.batch,-1,self.dim) 
        x_p_out = self.encoder(x_p,x_o) #q[b,p,312] kv[b,1,312] out=[b,p,312]
        x_o_out = self.decoder(x_o,x_p_out)
        x_o_out = x_o_out.squeeze(1) 
        return x_o_out, x_p_out
   


    def ranking_loss(self,score, targets, proposal_num=6):
        loss = Variable(torch.zeros(1).to(self.device))
        batch_size = score.size(0)
        for i in range(proposal_num):
            targets_p = (targets > targets[:, i].unsqueeze(1)).type(torch.cuda.FloatTensor)
            pivot = score[:, i].unsqueeze(1)
            loss_p = (1 - pivot + score) * targets_p
            loss_p = torch.sum(F.relu(loss_p))
            loss += loss_p
        return loss / batch_size

    def list_loss(self,logits, targets):
        temp = F.log_softmax(logits, -1)
        loss = [-temp[i][targets[i].item()] for i in range(logits.size(0))]
        return torch.stack(loss)



    def forward(self, x, att=None, label=None, seen_att=None, att_all=None):

        features_shallow = self.conv4(x)
        features = self.conv5(features_shallow)
        ## object-semantic attention
        intensity = self.VSIMModule(features_shallow) #intensity
        ### parts localization
        self.batch = features.size(0)
        x_pad = F.pad(x, (self.pad_side, self.pad_side, self.pad_side, self.pad_side), mode='constant', value=0)
        rpn_score = self.proposal_net(features.detach())
        all_cdds = [
            np.concatenate((x.reshape(-1, 1), self.edge_anchors.copy(), np.arange(0, len(x)).reshape(-1, 1)), axis=1)
            for x in rpn_score.data.cpu().numpy()]
        top_n_cdds = [hard_nms(x, topn=self.topN_train, iou_thresh=0.3) for x in all_cdds]
        top_n_cdds = np.array(top_n_cdds)
        top_n_index = top_n_cdds[:, :, -1].astype(np.int)
        top_n_index = torch.from_numpy(top_n_index).cuda()

        top_n_index = torch.as_tensor(top_n_index, dtype=torch.long).cuda()
        top_n_prob = torch.gather(rpn_score, dim=1, index=top_n_index)
        part_imgs = torch.zeros([self.batch, self.topN_train, 3, 224, 224]).cuda()

       
        for i in range(self.batch):
            for j in range(self.topN_train):
                [y0, x0, y1, x1] = top_n_cdds[i][j, 1:5].astype(np.int)
                part_imgs[i:i + 1, j] = F.interpolate(x_pad[i:i + 1, :, y0:y1, x0:x1], size=(224, 224), mode='bilinear',
                                                      align_corners=True)
        part_imgs = part_imgs.view(self.batch * self.topN_train, 3, 224, 224)

      
        features_part_all_shallow = self.conv4_parts(part_imgs.detach())
        features_part_all = self.conv5_parts(features_part_all_shallow)
        
        features_part_all = self.avgpool(features_part_all).view(self.batch, self.topN_train,-1)
        features_part_rank = features_part_all.view(self.batch * self.topN_train, -1)
        feature_raw = self.avgpool(features).view(self.batch, -1)
        object_feat,score_raw,b1 = self.base_module(intensity,feature_raw, seen_att,att_all, True)
        parts_feat, score_rank, b2 = self.base_module_p(features_part_rank, seen_att, att_all, False)
        object_out, parts_out = self.cross_learn(object_feat, parts_feat) #lm7
       
        att_all = self.attrSFT((att_all,self.attr_cond))
        score_w, b = self.compute_score(object_out, seen_att, att_all, True)
        
        score = score_w
        

        if not self.training:
            return score
        Lreg = self.Reg_loss(intensity, att)
        partcls_loss = self.CLS_loss(score_rank.view(self.batch * self.topN_train, -1),
                                            label.unsqueeze(1).repeat(1, self.topN_train).view(-1))
        Lcls = self.CLS_loss(score, label)+partcls_loss
        
      
        scale = self.scale.item()

        loss_dict = {
            'Reg_loss': Lreg,
            'Cls_loss': Lcls,
            'scale': scale,
            'bias_loss':b
        }

        return loss_dict


def build_POPRNet(cfg):
    dataset_name = cfg.DATASETS.NAME
    info = utils.get_attributes_info(dataset_name)
    attritube_num = info["input_dim"]
    cls_num = info["n"]
    ucls_num = info["m"]

    attr_group = utils.get_attr_group(dataset_name)

    img_size = cfg.DATASETS.IMAGE_SIZE


    # res101 feature size
    c,w,h = 2048, img_size//32, img_size//32

    scale = cfg.MODEL.SCALE

    pretrained = cfg.MODEL.BACKBONE.PRETRAINED
    model_dir = cfg.PRETRAINED_MODELS

    res101 = resnet101_features(pretrained=pretrained, model_dir=model_dir)

    w2v_file = dataset_name+"_attribute.pkl"
    w2v_path = join(cfg.MODEL.ATTENTION.W2V_PATH, w2v_file)


    with open(w2v_path, 'rb') as f:
        w2v = pickle.load(f)

    device = torch.device(cfg.MODEL.DEVICE)

    return POPRNet(res101=res101, img_size=img_size,
                  c=c, w=w, h=h, scale=scale,
                  attritube_num=attritube_num,
                  attr_group=attr_group, w2v=w2v,
                  cls_num=cls_num, ucls_num=ucls_num,
                  device=device)