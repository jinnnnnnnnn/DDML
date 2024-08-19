import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from pytorch_metric_learning import miners, losses
from torch.distributions import Normal 

class MultivariateNormalDiag():
    
    def __init__(self, locs, scales):
        super(MultivariateNormalDiag, self).__init__()
        self.locs = locs
        self.scales = scales

    def log_prob(self, x):

        normal = Normal(loc=self.locs, scale=self.scales)
        return normal.log_prob(x).sum(-1)
    
    def sample(self, shape=()):
        eps = torch.randn(shape + (self.locs.shape[1],)).cuda() 
        return self.locs + self.scales * eps
  


def disentML(input, input_specific, proxy_l2, target, scale, num_classes):
    '''
    input: [batch_size, dims*2]  embedding bottleneck features before reparameterization trick
    input_specific: [batch_size, dims*2] specific bottleneck features before reparameterization trick
    proxy_l2: [n_classes, dims] l2-normalized proxy parameters
    target: [batch_size] labels
    scale : hyper-parameter for decoder

    '''
    proxy_l2 = F.normalize(proxy_l2, p=2, dim=1)
    dim=int(input.shape[1]/2)

    prior = MultivariateNormalDiag(torch.zeros(dim).cuda(), torch.ones(dim).cuda())

    mu = input[:,:dim]
    logvar = input[:,dim:]
    std = F.softplus(logvar-5)
    z_dist = MultivariateNormalDiag(mu, std)
    input_rdn = z_dist.sample ((mu.shape[0],))
    input_rdn_l2 = F.normalize(input_rdn, p=2, dim=1)

    sim_mat = input_rdn_l2.matmul(proxy_l2.t())
        
    logits = scale * sim_mat

    one_hot = F.one_hot(target, num_classes)
    neg_target = torch.full(size = (input_rdn.shape[0], one_hot.shape[1]), fill_value = 1/(one_hot.shape[1]), dtype = torch.float).cuda()
    
    agnostic_loss = F.cross_entropy(logits, neg_target)

    rel_mu = input_specific[:,:dim]
    rel_var = input_specific[:,dim:]
    rel_std = F.softplus(rel_var-5,beta=1)
    z_s_dist = MultivariateNormalDiag(rel_mu, rel_std)
    input_spcf_rdn = z_s_dist.sample ((rel_mu.shape[0],))

    split_loss = (z_s_dist.log_prob(input_spcf_rdn) - prior.log_prob(input_spcf_rdn)).mean()

    input_spcf_l2_rdn = F.normalize(input_spcf_rdn, p=2, dim=1)
    sim_mat_spcf = input_spcf_l2_rdn.matmul(proxy_l2.t())     
    logits_spcf = scale * sim_mat_spcf
    
    specific_loss = F.cross_entropy(logits_spcf, target)

    return agnostic_loss, split_loss, specific_loss, logits, logits_spcf


def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    T = torch.FloatTensor(T).cuda()
    return T

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output

class Proxy_Anchor(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg = 0.1, alpha = 32, al=0.0, be=0.0, gam=0.0):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        # ProxyAnchor hyper-parameter
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha
        self.scale = 23.0

        # DDML hyper-parameter
        self.al = al
        self.be = be
        self.gam = gam
        
    def forward(self, X, x, T, disent):
        if disent:
            P = self.proxies
            agnostic_loss, split_loss, specific_loss, logits, logits_spcf = disentML(X, x, l2_norm(P), T, self.scale, num_classes=self.nb_classes)
            AGReg =  self.al *agnostic_loss + self.be *specific_loss + self.gam * split_loss

            emb_mu = X[:,:self.sz_embed]
            input_rdn = emb_mu
            input_rdn_l2 = F.normalize(input_rdn, p=2, dim=1)

            cos = F.linear(input_rdn_l2, l2_norm(P))  # Calcluate cosine similarity
            P_one_hot = binarize(T = T, nb_classes = self.nb_classes)
            N_one_hot = 1 - P_one_hot
        
            pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
            neg_exp = torch.exp(self.alpha * (cos + self.mrg))

            with_pos_proxies = torch.nonzero(P_one_hot.sum(dim = 0) != 0).squeeze(dim = 1)   # The set of positive proxies of data in the batch
            num_valid_proxies = len(with_pos_proxies)   # The number of positive proxies
            
            P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0) 
            N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
            
            pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
            neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
            dml_loss = pos_term + neg_term   

     

            loss = AGReg + dml_loss

            return loss, logits, logits_spcf, [ agnostic_loss, specific_loss,  split_loss ]

        P = self.proxies

        cos = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity
        P_one_hot = binarize(T = T, nb_classes = self.nb_classes)
        N_one_hot = 1 - P_one_hot
    
        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim = 0) != 0).squeeze(dim = 1)   # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)   # The number of positive proxies
        
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0) 
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
        
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term     
        
        return loss

# We use PyTorch Metric Learning library for the following codes.
# Please refer to "https://github.com/KevinMusgrave/pytorch-metric-learning" for details.
class Proxy_NCA(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, scale=32):
        super(Proxy_NCA, self).__init__()
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.scale = scale
        self.loss_func = losses.ProxyNCALoss(num_classes = self.nb_classes, embedding_size = self.sz_embed, softmax_scale = self.scale).cuda()

    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss
    
class MultiSimilarityLoss(torch.nn.Module):
    def __init__(self, ):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.epsilon = 0.1
        self.scale_pos = 2
        self.scale_neg = 50
        
        self.miner = miners.MultiSimilarityMiner(epsilon=self.epsilon)
        self.loss_func = losses.MultiSimilarityLoss(self.scale_pos, self.scale_neg, self.thresh)
        
    def forward(self, embeddings, labels):
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, hard_pairs)
        return loss
    
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5, **kwargs):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.loss_func = losses.ContrastiveLoss(neg_margin=self.margin) 
        
    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss
    
class TripletLoss(nn.Module):
    def __init__(self, margin=0.1, **kwargs):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.miner = miners.TripletMarginMiner(margin, type_of_triplets = 'semihard')
        self.loss_func = losses.TripletMarginLoss(margin = self.margin)
        
    def forward(self, embeddings, labels):
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, hard_pairs)
        return loss
    
class NPairLoss(nn.Module):
    def __init__(self, l2_reg=0):
        super(NPairLoss, self).__init__()
        self.l2_reg = l2_reg
        self.loss_func = losses.NPairsLoss(l2_reg_weight=self.l2_reg, normalize_embeddings = False)
        
    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss