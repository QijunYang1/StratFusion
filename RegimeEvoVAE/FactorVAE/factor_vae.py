import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats


class FeatureExtractor(nn.Module):
    """ 
    Extracts stocks latent features e from the historical sequential characteristics x.
    """
    
    def __init__(self, C: int, H: int, time_length, num_layers=1):
        """
        Initialize parameters.
            
        Args:
            C: Dimension of characteristics.
            H: Dimension of hidden states.
        """
        super().__init__()

        self.proj = nn.Sequential(
            nn.LayerNorm(C),
            nn.Linear(C,H),
            nn.LeakyReLU(),
            nn.GRU(H,H,num_layers=num_layers, batch_first=True))
    
    def forward(self,x):
        '''
        Extracts latent features.

        Args:
            x: Stock characteristics tensor with shape of (stock_size, time_length[seq_len], characteristics_size).

        Return:
            e: Latent features of stocks with shape of (stock_size, H).
        '''      
        # Feed x to the FC and GRU
        _,h_n = self.proj(x) # shape: (stock_size, H)
        return h_n.squeeze(0)

class FactorEncoder(nn.Module):
    def __init__(self, H, portfolio_num, factor_num):
        """
        Initialize parameters.
            
        Args:
            H: Dimension of hidden states.
            portfolio_num: Number of portfolios.
            factor_num: Number of contructed factors.
        """
        super().__init__()

        # Portfolio Layer (fully connected layer)
        self.portfolio_fc = nn.Linear(H, portfolio_num)
        self.softmax = nn.Softmax(dim=1) 

        # Mapping Layer to map portfolio returns to mu and sigma of factors
        self.factor_mu_fc = nn.Linear(portfolio_num,factor_num)
        self.factor_sigma_fc = nn.Sequential(
            nn.Linear(portfolio_num,factor_num),
            nn.Softplus()
        )

    def forward(self, e, ft):
        """
        Posterior factors.
        Construct a set of portfolios inspired by (Gu, Kelly, and Xiu 2021), 
        these portfolios are dynamically re-weighted on the basis of stock latent features. Then calculate
        the mu and sigma of factors.
        
        Args:
            e: Stocks latent features with shape of (stock_size, H).
            ft: Future returns with shape of (stock_size,).
            
        Returns:
            mu:  Posterior mean of factors (factor_num,)
            sigma: Posterior logsigma of factors (factor_num,)
        """
        port_wts = self.softmax(self.portfolio_fc(e)) # Portfolio weights: (stock_size, portfolio_num)
        port_rts = (ft.unsqueeze(0) @ port_wts).squeeze(0) # (portfolio_num,)

        mu = self.factor_mu_fc(port_rts) # mu: (factor_num,)
        sigma = self.factor_sigma_fc(port_rts) # sigma: (factor_num,)
        
        return mu, sigma

class FactorDecoder(nn.Module):
    def __init__(self, H, factor_num):
        """
        Initialize parameters.
            
        Args:
            H: Dimension of hidden states.
            factor_num: Number of contructed factors.
        """
        super().__init__()

        # Alpha layer outputs idiosyncratic returns α from the latent features e.
        self.h_fc = nn.Sequential(
            nn.Linear(H,H),
            nn.LeakyReLU()
        )
        self.alpha_mu_fc = nn.Linear(H,1)
        self.alpha_sigma_fc = nn.Sequential(
            nn.Linear(H,1),
            nn.Softplus()
        )

        # Beta layer calculates factor exposure from the latent features e by linear mapping.
        self.beta_fc = nn.Linear(H,factor_num)

    def reparameterize(self, mu, sigma):
        eps = torch.randn_like(sigma)
        return mu + eps * sigma

    def forward(self, factor_mu, factor_sigma, e):
        """
        Decoder forward pass. Uses factors z and the latent feature e to calculate stock returns y.
        
        Args:
            factor_mu: Mean of factors with shape of (factor_num,).
            factor_sigma: Sigma of factors with shape of (factor_num,).
            e: Stocks latent features with shape of (stock_size, H).

        Returns: 
            rt_mu:  Mean of future returns with shape of (stock_size, )
            rt_sigma: Sigma of future returns with shape of (stock_size, )
        """
        # Calculate alpha
        h = self.h_fc(e)    # (stock_size, H)
        alpha_mu = self.alpha_mu_fc(h).squeeze(-1)    # (stock_size,)
        alpha_sigma = self.alpha_sigma_fc(h).squeeze(-1)  # (stock_size,)
        

        # Calculate Beta
        beta = self.beta_fc(e)  # (stock_size, factor_num)
        
        # Replace any zero values in factor_sigma with a small value
        factor_sigma[factor_sigma == 0] = 1e-6
        # Calculate mu and sigma
        rt_mu = alpha_mu + (beta @ factor_mu) # (stock_size, )
        rt_sigma = ( alpha_sigma.square() + (beta.square() @ factor_sigma.square()) +1e-6).sqrt() # (stock_size, )
        rt = self.reparameterize(rt_mu, rt_sigma)
        return rt_mu, rt_sigma, rt

class Attention(nn.Module):
    def __init__(self, H):
        """
        Initialize parameters.
            
        Args:
            H: Dimension of hidden states.
            num_heads: Number of parallel attention heads.
        """
        super().__init__()
        self.q = nn.Parameter(torch.randn(H)) # (H,)
        self.k = nn.Linear(H,H) # (stock_size, H).
        self.v = nn.Linear(H,H) # (stock_size, H).
        self.dropout = nn.Dropout(0.1)

    def forward(self,x):
        # Calculate Key and Value
        k = self.k(x) # (stock_size, H)
        v = self.v(x) # (stock_size, H)
        # Calcualte scale -> L2-norm
        # scale = torch.norm(k,p=2,dim=1) # (stock_size,)
        scale = (k.square().sum(dim=1)).sqrt() # (stock_size,)
        eps = torch.finfo(scale.dtype).eps # the code below prevents from "exp overflow"
        eps = torch.tensor([eps]).expand_as(scale)
        scale = torch.maximum(scale,eps) # (stock_size,)
        # Use ReLu and normalize the row instead of directly using SoftMax.  
        attention = (self.q @ k.transpose(-1,-2))/scale 
        attention = self.dropout(attention)   
        attention = F.relu(attention) # (stock_size,)
        attention =  F.softmax(attention, dim=0) # (N) # (stock_size,)
        if torch.isnan(attention).any() or torch.isinf(attention).any():
            return torch.zeros_like(self.v[0])
        else:
            h_atten = (attention @ v).unsqueeze(0) # (1,H)
            return h_atten 

class FactorPredictor(nn.Module):
    def __init__(self, H, num_heads):
        """
        Initialize parameters.
            
        Args:
            H: Dimension of hidden states.
            num_heads: Number of parallel attention heads (factor_num).
        """
        super().__init__()

        # Multi-head attention
        self.mha = nn.ModuleList([Attention(H) for _ in range(num_heads)])

        # Predicted factor layer outputs 
        self.h_fc = nn.Sequential(
            nn.Linear(H,H),
            nn.LeakyReLU()
        )
        self.mu_fc = nn.Linear(H,1)
        self.sigma_fc = nn.Sequential(
            nn.Linear(H,1),
            nn.Softplus()
        )

    def forward(self,x):
        '''
        Calculate Attention then get predicted mu and std.
        Args:
            x: Stock characteristics tensor with shape of (stock_size, time_length[seq_len], characteristics_size).
        Returns: 
            mu:  Prior mean of factors (factor_num,)
            sigma: Prior sigma of factors (factor_num,)
        '''
        # Calculate attention
        h_multi = []
        for attention in self.mha:
            h_multi.append(attention(x))
        h_multi = torch.cat(h_multi,dim=0) # (factor_num,H)

        # Calculate predicted alpha
        h = self.h_fc(h_multi)    # (factor_num, H)
        mu = self.mu_fc(h).squeeze(-1)    # (factor_num,)
        sigma = self.sigma_fc(h).squeeze(-1)  # (factor_num,)

        return mu, sigma

class FactorVAE(nn.Module):
    def __init__(self, C, H, portfolio_num, factor_num, time_length, gru_layers=1):
        '''
        Args: 
            C: Dimension of characteristics.
            H: Dimension of hidden states.
            portfolio_num: Number of portfolios.
            factor_num: Number of contructed factors. (num_heads: Number of parallel attention heads.)
            gru_layers: GRU stacking layers.
        '''
        super().__init__()

        self.FeatureExtractor = FeatureExtractor(C,H,time_length,gru_layers)
        self.FactorEncoder = FactorEncoder(H, portfolio_num, factor_num)
        self.FactorDecoder = FactorDecoder(H, factor_num)
        self.FactorPredictor = FactorPredictor(H, factor_num)

    def forward(self, x, ft):
        """Implements forward pass of FactorVAE.
        
        Args:
            x: Stock characteristics tensor with shape of (stock_size, time_length[seq_len], characteristics_size).
            e: Stocks latent features with shape of (stock_size, H).
            ft: Future returns with shape of (stock_size,).
        
        Returns:
            prior_mu:  Prior mean of factors (factor_num,)
            prior_sigma: Prior log sigma of factors (factor_num,)
            post_mu:  Posterior mean of factors (factor_num,)
            post_sigma: Posterior log sigma of factors (factor_num,)
            rt_mu:  Mean of future returns with shape of (stock_size, )
            rt_sigma: Sigma of future returns with shape of (stock_size, )
        """
        e = self.FeatureExtractor(x)
        post_mu, post_sigma = self.FactorEncoder(e, ft)
        prior_mu, prior_sigma = self.FactorPredictor(e)
        rt_mu, rt_sigma, rt = self.FactorDecoder(post_mu, post_sigma, e)
        return prior_mu, prior_sigma, post_mu, post_sigma, rt_mu, rt_sigma

    def predict(self,x):
        e = self.FeatureExtractor(x)
        prior_mu, prior_sigma = self.FactorPredictor(e)
        rt_mu, rt_sigma, _ = self.FactorDecoder(prior_mu, prior_sigma, e)
        return rt_mu, rt_sigma

def loss_VAE_v1(ft, prior_mu, prior_sigma, post_mu, post_sigma, rt_mu, rt_sigma):
    """
    Computes the loss = -ELBO = Negative Log-Likelihood + KL Divergence.
    
    Args: 
        ft: Future returns with shape of (stock_size,).
        prior_mu:  Prior mean of factors (factor_num,)
        prior_sigma: Prior log sigma of factors (factor_num,)
        post_mu:  Posterior mean of factors (factor_num,)
        post_sigma: Posterior log sigma of factors (factor_num,)
        rt_mu:  Mean of future returns with shape of (stock_size, )
        rt_sigma: Sigma of future returns with shape of (stock_size, )
    """
    
    NLL = F.gaussian_nll_loss(rt_mu,ft,rt_sigma.square())
    if torch.any(prior_sigma == 0):
            prior_sigma[prior_sigma == 0] = 1e-6
    post_var = post_sigma**2
    prior_var = prior_sigma**2
    KLD = (post_var/prior_var - 1 + (prior_mu-post_mu)**2/prior_var + torch.log(prior_var/post_var)).sum()
    return NLL + KLD

def loss_VAE_v2(ft, prior_mu, prior_sigma, post_mu, post_sigma, pred_rt):
    """
    Computes the loss = -ELBO = Negative Log-Likelihood + KL Divergence.
    
    Args: 
        ft: Future returns with shape of (stock_size,).
        prior_mu:  Prior mean of factors (factor_num,)
        prior_sigma: Prior log sigma of factors (factor_num,)
        post_mu:  Posterior mean of factors (factor_num,)
        post_sigma: Posterior log sigma of factors (factor_num,)
        pred_rt:  predicted returns (stock_size, )
    """
    reconstruction_loss = F.mse_loss(pred_rt, ft)
    if torch.any(prior_sigma == 0):
            prior_sigma[prior_sigma == 0] = 1e-6
    post_var = post_sigma**2
    prior_var = prior_sigma**2
    KLD = 0.5 * (post_var/prior_var - 1 + (prior_mu-post_mu)**2/prior_var + torch.log(prior_var/post_var)).sum()
    return reconstruction_loss + KLD
