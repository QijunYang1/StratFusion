import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.distributions import Normal,kl_divergence
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import joblib
from scipy import stats
from itertools import combinations
import numpy as np
import pandas as pd
from numba import njit

class MarketRegimeAttention(nn.Module):
    def __init__(self, C):
        """
        Initialize parameters.
            
        Args:
            C: Dimension of characteristics.
        """
        super().__init__()
        self.q = nn.Parameter(torch.rand(C)) # (C,)
        self.k = nn.Linear(C,C) # (L, C).
        self.v = nn.Linear(C,C) # (L, C).

    def forward(self,x):
        # Calculate Key and Value
        k = self.k(x) # (L, C).
        v = self.v(x) # (L, C).

        # Calcualte scale -> L2-norm
        scale = torch.norm(self.q, p=2) * torch.norm(k,p=2,dim=1) # (L,)
        eps = torch.finfo(scale.dtype).eps # the code below prevents from "exp overflow"
        eps = torch.tensor([eps]).expand_as(scale)
        scale = torch.maximum(scale,eps) # (L,)

        # Use ReLu and normalize using SoftMax.
        attention = (self.q @ k.transpose(-1,-2))/scale   
        attention = F.leaky_relu(attention) # (L,)
        attention = F.softmax(attention, dim=0) # (L)
        h_atten = (attention @ v).unsqueeze(0) # (1,C)
        return h_atten

class MarketFeatureExtractor(nn.Module):
    """ 
    Extracts market latent features e from the historical sequential characteristics x.
    """
    
    def __init__(self, C_market: int, H_market: int,  num_market_feature: int):
        """
        Initialize parameters.
            
        Args:
            C_market: Dimension of market characteristics.
            H_market: Dimension of hidden market features.
            num_market_feature: Number of hidden market features.
        """
        super().__init__()

        # Multi-head attention to different market regime
        self.bn = nn.LayerNorm(C_market)
        self.mha = nn.ModuleList([MarketRegimeAttention(C_market) for _ in range(num_market_feature)]) # (num_features,C)
        self.fc = nn.Linear(C_market,H_market)
    
    def forward(self,x):
        '''
        Extracts market latent features.

        Args:
            x: Market characteristics tensor with shape of (time_length[seq_len], characteristics_size).

        Return:
            e: Latent features of stocks with shape of (num_market_feature, H_market).
        '''      
        # Normalization
        x = self.bn(x)
        
        # Calculate attention
        h_multi = []
        for attention in self.mha:
            h_multi.append(attention(x))
        h_multi = torch.cat(h_multi,dim=0) # (num_market_feature, C_market)
        h_multi = self.fc(h_multi) # (num_market_feature, H_market)
        return h_multi

class MarketFactorEncoder(nn.Module):
    def __init__(self, H_market, num_market_portfolio, num_market_factor, mfi_dim, num_market_feature):
        """
        Initialize parameters.
            
        Args:
            H_market: Dimension of hidden states.
            num_market_portfolio: Number of portfolios.
            num_market_factor: Number of contructed factors.
            mfi_dim: Dimension of market future infomation.
            num_market_feature: Number of market hidden features.
        """
        super().__init__()

        # Portfolio Layer (fully connected layer)
        self.portfolio_weights_fc = nn.Sequential(
            nn.Linear(H_market, num_market_portfolio),
            nn.Softmax(dim=1) 
        ) # Portfolio possible weights: (num_market_feature, num_market_portfolio)

        # Market return Layer
        self.market_return_layer = nn.Linear(mfi_dim, num_market_feature)

        # Mapping Layer to map portfolio returns to mu and sigma of factors
        self.factor_mu_fc = nn.Linear(num_market_portfolio,num_market_factor)
        self.factor_sigma_fc = nn.Sequential(
            nn.Linear(num_market_portfolio,num_market_factor),
            nn.Softplus()
        )

    def reparameterize(self, mu, sigma):
        eps = torch.randn_like(sigma)
        return mu + eps * sigma
    
    def forward(self, e, mfi):
        """
        Posterior market factors.
        These portfolios are dynamically re-weighted on the basis of market latent features. Then calculate
        the mu and sigma of market factors.
        
        Args:
            e: Market latent features with shape of (num_market_feature, H_market).
            mfi: Market future returns with shape of (mfi_dim,).
            
        Returns:
            mu:  Posterior mean of market factors (num_market_factor,)
            sigma: Posterior logsigma of market factors (num_market_factor,)
        """
        port_wts = self.portfolio_weights_fc(e) # Portfolio weights: (num_market_feature, num_market_portfolio)
        mft =  self.market_return_layer(mfi) # Market Portfolio Returns: (num_market_feature,)
        port_rts = (mft.unsqueeze(0) @ port_wts).squeeze(0) # (num_market_portfolio,)

        mu = self.factor_mu_fc(port_rts) # mu: (num_market_factor,)
        sigma = self.factor_sigma_fc(port_rts) # sigma: (num_market_factor,)
        
        return mu, sigma, self.reparameterize(mu, sigma)

class MarketFactorPredictor(nn.Module):
    def __init__(self, H_market, num_market_factor):
        """
        Initialize parameters.
            
        Args:
            H_market: Dimension of hidden states.
            num_market_factor: Number of parallel attention heads (num_market_factor).
        """
        super().__init__()

        # Multi-head attention to different market factors
        self.mha = nn.ModuleList([MarketRegimeAttention(H_market) for _ in range(num_market_factor)]) # (num_market_factor,H_market)

        # Predicted factor layer outputs 
        self.h_fc = nn.Sequential(
            nn.Linear(H_market,H_market),
            nn.LeakyReLU()
        )
        self.mu_fc = nn.Linear(H_market,1)
        self.sigma_fc = nn.Sequential(
            nn.Linear(H_market,1),
            nn.Softplus()
        )

    def forward(self,x):
        '''
        Calculate market attention then get predicted mu and std.
        Args:
            x: Market characteristics tensor with shape of (market_state,H).
        Returns: 
            mu:  Prior mean of factors (factor_num,)
            sigma: Prior sigma of factors (factor_num,)
        '''
        # Calculate attention
        h_multi = []
        for attention in self.mha:
            h_multi.append(attention(x))
        h_multi = torch.cat(h_multi,dim=0) # (market_factor_num,H)

        # Calculate predicted alpha
        h = self.h_fc(h_multi)    # (market_factor_num, H)
        mu = self.mu_fc(h).squeeze(-1)    # (market_factor_num,)
        sigma = self.sigma_fc(h).squeeze(-1) # (market_factor_num,)

        return mu, sigma

class MarketRegimeExtractor(nn.Module):
    def __init__(self, num_market_factor, num_market_regime, beta):
        """
        Initialize parameters.
            
        Args:
            num_market_feature: Dimension of market hidden feature.
            H_market: Dimension of hidden market states.
            num_market_factor: Number of contructed market factors.
            num_market_regime: Number of market regime.
        """
        super().__init__()

        # Intialize 
        self.mr_mu = torch.randn(num_market_regime)
        self.mr_sigma = torch.randn(num_market_regime)
        self.beta = beta

        # Market bias layer (using market feature e)
        self.h_fc = nn.Sequential(
            nn.Linear(num_market_factor, num_market_factor),
            nn.LeakyReLU()
        )
        self.mb_mu_fc = nn.Linear(num_market_factor,num_market_regime)
        self.mb_sigma_fc = nn.Sequential(
            nn.Linear(num_market_factor,num_market_regime),
            nn.Softplus()
        )

    def forward(self, market_factor):
        """
        Decoder forward pass. Uses factors z and the latent feature e to calculate market clusters c.
        
        Args:
            market_factor_mu: Mean of factors with shape of (factor_num,).
            market_factor_sigma: Sigma of factors with shape of (factor_num,).
            e: Market latent features with shape of (market_states, H).

        Returns: 
            ms_mu:  Mean of market state with shape of (market_states, )
            ms_sigma: Sigma of market state with shape of (market_states, )
        """
        # Calculate market bias
        h = self.h_fc(market_factor)    # (num_market_factor, )
        mb_mu = self.mb_mu_fc(h)  # (market_regime_num,)
        mb_sigma = self.mb_sigma_fc(h)  # (market_regime_num,)
        
        # sort
        idx = mb_mu.argsort()
        mr_mu = self.beta * self.mr_mu + (1-self.beta) * mb_mu[idx]
        mr_sigma = self.beta * self.mr_sigma + (1-self.beta) * mb_sigma[idx]
        self.mr_mu = mr_mu.detach()
        self.mr_sigma = mr_sigma.detach()
        return mr_mu, mr_sigma

class MarketRegimePredictor(nn.Module):
    def __init__(self,num_market_factor):
        super().__init__()

        # Market regime prediction layer (using market factors factors)
        self.proj = nn.Linear(num_market_factor,1) # (C,)

    def forward(self, market_factor, ms_mu, ms_sigma):
        # Calculate market regime prediction
        score = self.proj(market_factor)

        # Calculate distance between generated market states
        ws2_dis = -ms_sigma.log() - 0.5*((score-ms_mu)**2/ms_sigma) # (num_market_regime, )
        ws2_dis = ws2_dis - ws2_dis.max()
        prob = F.softmax(ws2_dis,dim=0) # (num_market_regime, ) 
        return prob.argmax()
    

class StockFeatureExtractor(nn.Module):
    """ 
    Extracts stocks latent features e from the historical sequential characteristics x.
    """
    
    def __init__(self, C_stock: int, H_stock: int, time_length, gru_num_layers=1):
        """
        Initialize parameters.
            
        Args:
            C_stock: Dimension of characteristics.
            H_stock: Dimension of hidden states.
        """
        super().__init__()

        self.proj = nn.Sequential(
            nn.LayerNorm(C_stock),
            nn.Linear(C_stock,H_stock),
            nn.LeakyReLU(),
            nn.GRU(H_stock,H_stock,num_layers=gru_num_layers, batch_first=True))
        
    def forward(self,x):
        '''
        Extracts latent features.

        Args:
            x: Stock characteristics tensor with shape of (stock_size, time_length[seq_len], characteristics_size).

        Return:
            e: Latent features of stocks with shape of (stock_size, H).
        '''      
        # Feed x to the FC and GRU
        out,_ = self.proj(x) # shape: (stock_size,time_length, H)
        return out[:,-1,:]  # shape: (stock_size, H)

class StockFactorEncoder(nn.Module):
    def __init__(self, H_stock, num_stock_portfolio, num_stock_factor, num_market_factor):
        """  
        Initialize parameters.
            
        Args:
            H_stock: Dimension of hidden stock features.
            num_stock_portfolio: Number of portfolios.
            num_stock_factor: Number of contructed factors.
            num_market_factor: Number of market factors.
        """
        super().__init__()


        # Portfolio Layer (fully connected layer)
        self.portfolio_weights_fc = nn.Sequential(
            nn.Linear(H_stock+num_market_factor, num_stock_portfolio),
            nn.Softmax(dim=1) 
        ) 

        # Mapping Layer to map portfolio returns to mu and sigma of factors
        self.factor_mu_fc = nn.Linear(num_stock_portfolio,num_stock_factor)
        self.factor_sigma_fc = nn.Sequential(
            nn.Linear(num_stock_portfolio,num_stock_factor),
            nn.Softplus()
        )

    def reparameterize(self, mu, sigma):
        eps = torch.randn_like(sigma)
        return mu + eps * sigma
    
    def forward(self, es, ft, market_factor):
        """
        Posterior factors.
        Construct a set of portfolios inspired by (Gu, Kelly, and Xiu 2021), 
        these portfolios are dynamically re-weighted on the basis of stock latent features. Then calculate
        the mu and sigma of factors.
        
        Args:
            es: Stock latent features with shape of (stock_size, H_stock).
            ft: Stock future returns with shape of (stock_size,).
            market_factor_mu: Market factors mean with shape of (num_market_factor, ).
            market_factor_sigma: Market factors sigma with shape of (num_market_factor, ).

        Returns:
            mu:  Posterior mean of factors (num_stock_factor,)
            sigma: Posterior logsigma of factors (num_stock_factor,)
        """
        es = torch.concat([es, market_factor.expand(es.shape[0],-1) ], dim=1) # (stock_size, H_stock + num_market_factor)

        # Combine market info with stock info together
        port_wts = self.portfolio_weights_fc(es) # Portfolio weights: (stock_size, num_stock_portfolio)
        port_rts = (ft.unsqueeze(0) @ port_wts).squeeze(0) # (num_stock_portfolio,)

        mu = self.factor_mu_fc(port_rts) # mu: (num_stock_factor,)
        sigma = self.factor_sigma_fc(port_rts) # sigma: (num_stock_factor,)
        
        return mu, sigma, self.reparameterize(mu,sigma)

class StockAttention(nn.Module):
    def __init__(self, H):
        """
        Initialize parameters.
            
        Args:
            H: Dimension of hidden states.
        """
        super().__init__()
        self.q = nn.Parameter(torch.rand(H)) # (H,)
        self.k = nn.Linear(H,H) # (stock_size, H).
        self.v = nn.Linear(H,H) # (stock_size, H).

    def forward(self,x):
        # Calculate Key and Value
        k = self.k(x) # (stock_size, H)
        v = self.v(x) # (stock_size, H)
        # Calcualte scale -> L2-norm
        scale = torch.norm(self.q, p=2) * torch.norm(k,p=2,dim=1) # (stock_size,)
        eps = torch.finfo(scale.dtype).eps # the code below prevents from "exp overflow"
        eps = torch.tensor([eps]).expand_as(scale)
        scale = torch.maximum(scale,eps) # (stock_size,)
        # Use ReLu and normalize the row instead of directly using SoftMax.     
        attention = F.relu( (self.q @ k.transpose(-1,-2))/scale ) # (stock_size,)
        attention = attention/(attention.sum()+eps)
        h_atten = (attention @ v).unsqueeze(0) # (1,H)
        return h_atten

class StockFactorPredictor(nn.Module):
    def __init__(self, H_stock, num_stock_factor, num_market_factor):
        """
        Initialize parameters.
            
        Args:
            H_stock: Dimension of hidden states.
            num_stock_factor: Number of parallel attention heads (num_stock_factor).
            num_market_factor: Number of market factors.
        """
        super().__init__()

        # Multi-head attention
        self.mha = nn.ModuleList([StockAttention(H_stock+num_market_factor) for _ in range(num_stock_factor)])

        # Predicted factor layer outputs 
        self.h_fc = nn.Sequential(
            nn.Linear(H_stock+num_market_factor,H_stock+num_market_factor),
            nn.LeakyReLU()
        )
        self.mu_fc = nn.Linear(H_stock+num_market_factor,1)
        self.sigma_fc = nn.Sequential(
            nn.Linear(H_stock+num_market_factor,1),
            nn.Softplus()
        )

    def forward(self, es, market_factor):
        '''
        Calculate StockAttention then get predicted mu and std.
        Args:
            es: Stock latent features with shape of (stock_size, H_stock).
            market_factor_mu: Market factors mean with shape of (num_market_factor, ).
            market_factor_sigma: Market factors sigma with shape of (num_market_factor, ).
        Returns: 
            mu:  Prior mean of factors (num_stock_factor,)
            sigma: Prior sigma of factors (num_stock_factor,)
        '''

        # Add market factor info to stock features
        es = torch.concat([es, market_factor.expand(es.shape[0],-1) ], dim=1) # (stock_size, H_stock + num_market_factor)

        # Calculate attention
        h_multi = []
        for attention in self.mha:
            h_multi.append(attention(es))
        h_multi = torch.cat(h_multi,dim=0) # (num_stock_factor, H_stock + num_market_factor)

        # Calculate predicted alpha
        h = self.h_fc(h_multi)    # (num_stock_factor, H_stock + num_market_factor)
        mu = self.mu_fc(h).squeeze(-1)    # (num_stock_factor,)
        sigma = self.sigma_fc(h).squeeze(-1)  # (num_stock_factor,)

        return mu, sigma
    
class SingleFactorDecoder(nn.Module):
    def __init__(self, H_stock, num_stock_factor, num_market_factor):
        """
        Initialize parameters.
            
        Args:
            H_stock: Dimension of hidden states.
            num_stock_factor: Number of stock factors.
            num_market_factor: Number of market factors.
        """
        super().__init__()
        
        # Alpha layer outputs idiosyncratic returns α from the latent features e.
        self.h_fc = nn.Sequential(
            nn.Linear(H_stock + num_market_factor,H_stock + num_market_factor),
            nn.LeakyReLU()
        )
        self.alpha_mu_fc = nn.Linear(H_stock + num_market_factor,1)

        # Beta layer calculates factor exposure from the latent features e by linear mapping.
        self.beta_fc = nn.Linear(H_stock + num_market_factor, num_stock_factor)

    def forward(self, stock_factor, es, market_factor):
        """
        Decoder forward pass. Uses factors z and the latent feature e to calculate stock returns y.
        
        Args:
            stock_factor_mu: Mean of stock factors with shape of (num_stock_factor,).
            stock_factor_sigma: Sigma of stock factors with shape of (num_stock_factor,).
            es: Stock latent features with shape of (stock_size, H).
            market_factor_mu: Mean of market factors with shape of (market_factor_num,).
            market_factor_sigma: Sigma of market factors with shape of (market_factor_num,).

        Returns: 
            rt_mu:  Mean of future returns with shape of (stock_size, )
            rt_sigma: Sigma of future returns with shape of (stock_size, )
        """
        # Combine market info with stock info together
        es = torch.concat([es, market_factor.expand(es.shape[0],-1) ], dim=1) # (stock_size, H_stock + num_market_factor)

        # Calculate alpha
        h = self.h_fc(es)    # (stock_size, H_stock + num_market_factor)
        alpha = self.alpha_mu_fc(h).squeeze(-1)    # (stock_size, )

        # Calculate Beta
        beta = self.beta_fc(es)  # (stock_size,num_stock_factor)
        
        # Calculate mu and sigma
        rt = alpha + beta @ stock_factor # (stock_size)
        return rt

class FactorDecoder(nn.Module):
    def __init__(self, H_stock, num_stock_factor, num_market_factor, num_market_regime):
        """
        Initialize parameters.
            
        Args:
            H_stock: Dimension of hidden states.
            num_stock_factor: Number of stock factors.
            num_market_factor: Number of market factors.
            num_market_regime: Number of different market states.
        """
        super().__init__()

        # Multiple market states
        self.mms = nn.ModuleList(
            [SingleFactorDecoder(H_stock, num_stock_factor, num_market_factor) for _ in range(num_market_regime)]
            )

    def forward(self, stock_factor, es, market_factor, prob):
        """
        Decoder forward pass. Uses factors z and the latent feature e to calculate stock returns y.
        
        Args:
            stock_factor_mu: Mean of stock factors with shape of (num_stock_factor,).
            stock_factor_sigma: Sigma of stock factors with shape of (num_stock_factor,).
            es: Stock latent features with shape of (stock_size, H_stock).
            market_factor_mu: Mean of market factors with shape of (num_market_factor,).
            market_factor_sigma: Sigma of market factors with shape of (num_market_factor,).
            prob: Probability of different market states. (market_states, )

        Returns: 
            rt_mu:  Mean of future returns with shape of (stock_size, )
            rt_sigma: Sigma of future returns with shape of (stock_size, )
        """
        # Calculate attention
        rt = self.mms[prob](stock_factor, es, market_factor)
        return rt
    

class RegimeFactorVAE(nn.Module):
    def __init__(self, C_stock, C_market, H_stock, H_market, num_market_feature, num_stock_factor, num_market_factor, num_stock_portfolio, num_market_portfolio, time_length, num_market_regime, mfi_dim, beta, gru_num_layers):
        '''
        Args: 
            C_stock: Dimension of stock characteristics.
            C_market: Dimension of market characteristics.
            H_stock: Dimension of hidden stock features.
            H_market: Dimension of hidden market features.
            num_market_feature: Number of hidden market features.
            num_stock_factor: Number of contructed stock factors. (num_heads: Number of parallel attention heads.)
            num_market_factor: Number of contructed market factors.
            num_stock_portfolio: Number of constructed stock portfolio.
            num_market_portfolio: Number of possible market portfolio.
            time_length: Length of sequence.
            num_market_regime: Number of market regimes.
            mfi_dim: Dimension of market future infomation.
            beta: Decay rate of cluster mean and sigma.
            gru_layers: GRU stacking layers.
        '''
        super().__init__()

        # Market VAE
        self.MarketFeatureExtractor = MarketFeatureExtractor(C_market, H_market, num_market_feature)
        self.MarketFactorEncoder = MarketFactorEncoder(H_market, num_market_portfolio, num_market_factor, mfi_dim, num_market_feature)
        self.MarketFactorPredictor = MarketFactorPredictor(H_market, num_market_factor)
        self.MarketRegimeExtractor = MarketRegimeExtractor(num_market_factor, num_market_regime, beta)
        self.MarketRegimePredictor = MarketRegimePredictor(num_market_factor)
        # Stock VAE
        self.StockFeatureExtractor = StockFeatureExtractor(C_stock, H_stock, time_length, gru_num_layers)
        self.StockFactorEncoder = StockFactorEncoder(H_stock, num_stock_portfolio, num_stock_factor, num_market_factor)
        self.StockFactorPredictor = StockFactorPredictor(H_stock, num_stock_factor, num_market_factor)
        # Decoder
        self.FactorDecoder = FactorDecoder(H_stock, num_stock_factor, num_market_factor, num_market_regime)

    def forward(self, x_stock, y_stock, x_market, y_market):
        """
        Implements forward pass of RegimeFactorVAE.
        
        Args:
            x_stock: Stock characteristics tensor with shape of (stock_size, time_length[seq_len], characteristics_size).
            y_stock: Future returns with shape of (stock_size,).
            x_market: Market characteristics tensor with shape of (time_length[seq_len], characteristics_size).
            y_market: Market future returns with shape of (mfi_dim,).
        
        Returns:           
            market_post_mu: Posterior mean of market factors. (market_factor_num, )
            market_post_sigma: Posterior sigma of market factors. (market_factor_num, )
            market_prior_mu: Prior mean of market factors. (market_factor_num, )
            market_prior_sigma: Prior sigma of market factors. (market_factor_num, )
            market_regime_mu: Mean of market regime clusters. (states_num, )
            market_regime_sigma: Sigma of market regime clusters. (states_num, )
            stock_post_mu: Posterior mean of stock factors. (stock_factor_num, )
            stock_post_sigma: Posterior sigma of stock factors. (stock_factor_num, )
            stock_prior_mu: Prior mean of stock factors. (stock_factor_num, )
            stock_prior_sigma: Prior sigma of stock factors. (stock_factor_num, )
            rt_mu: Mean of future returns with shape of (stock_size, )
            rt_sigma: Sigma of future returns with shape of (stock_size, )
        """

        em = self.MarketFeatureExtractor(x_market) # enbedding_market: Market hidden features
        market_post_mu, market_post_sigma, market_posterior_factor = self.MarketFactorEncoder(em, y_market) # posterior market factor distribution
        market_prior_mu, market_prior_sigma = self.MarketFactorPredictor(em) # prior market factor distribution
        market_regime_mu, market_regime_sigma = self.MarketRegimeExtractor(market_posterior_factor)
        market_regime_prob = self.MarketRegimePredictor(market_posterior_factor, market_regime_mu, market_regime_sigma)

        es = self.StockFeatureExtractor(x_stock) # enbedding_stock: Stock hidden features
        stock_post_mu, stock_post_sigma, stock_posterior_factor = self.StockFactorEncoder(es, y_stock, market_posterior_factor) # posterior stock factor distribution
        stock_prior_mu, stock_prior_sigma = self.StockFactorPredictor(es, market_posterior_factor) # prior stock factor distribution

        rt = self.FactorDecoder(stock_posterior_factor, es, market_posterior_factor, market_regime_prob)

        return market_post_mu, market_post_sigma, market_prior_mu, market_prior_sigma, market_regime_mu, market_regime_sigma, stock_post_mu, stock_post_sigma, stock_prior_mu, stock_prior_sigma, rt


    def predict(self,x_stock, x_market):
        em = self.MarketFeatureExtractor(x_market) # enbedding_market: Market hidden features
        market_prior_mu, market_prior_sigma = self.MarketFactorPredictor(em) # prior market factor distribution
        market_regime_mu, market_regime_sigma = self.MarketRegimeExtractor(market_prior_mu)
        market_regime_prob = self.MarketRegimePredictor(market_prior_mu, market_regime_mu, market_regime_sigma)
        es = self.StockFeatureExtractor(x_stock) # enbedding_stock: Stock hidden features
        stock_prior_mu, stock_prior_sigma = self.StockFactorPredictor(es, market_prior_mu) # prior stock factor distribution
        rt_mu = self.FactorDecoder(stock_prior_mu, es, market_prior_mu, market_regime_prob)
        return rt_mu

def gaussian_kl_div(mu1, sigma1, mu2, sigma2):
    '''
    Args:
        mu1 & sigma1 -> gaussian a
        mu2 & sigma2 -> gaussian b
    Return:
        KLD(a,b) = KLD(a||b)
    '''
    # Calculate KL divergence between two Gaussian distributions
    if torch.any(sigma1 == 0):
        sigma1[sigma1 == 0] = 1e-6
    if torch.any(sigma2 == 0):
        sigma2[sigma2 == 0] = 1e-6
    res = (torch.log(sigma2/ sigma1) + (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2) - 0.5).sum()
    return res

def gaussian_ws_2_dis(mu1, sigma1, mu2, sigma2):
    '''
    Args:
        mu1 & sigma1 -> gaussian a
        mu2 & sigma2 -> gaussian b
    Return:
        WS2(a,b)
    '''
    res = (mu1-mu2).abs() + sigma1**2 + sigma2**2 - 2*(sigma1*sigma1)
    return res

def loss_RegimeFactorVAE_MSE(ft, market_post_mu, market_post_sigma, market_prior_mu, market_prior_sigma, market_regime_mu, market_regime_sigma, stock_post_mu, stock_post_sigma, stock_prior_mu, stock_prior_sigma, rt):
    """
    Computes the loss = -ELBO = Negative Log-Likelihood + KL Divergence(stock & market) + KL Divergence(between each cluster).
    
    Args: 
        ft: Future returns with shape of (stock_size,).
        market_post_mu: Posterior mean of market factors. (market_factor_num, )
        market_post_sigma: Posterior sigma of market factors. (market_factor_num, )
        market_prior_mu: Prior mean of market factors. (market_factor_num, )
        market_prior_sigma: Prior sigma of market factors. (market_factor_num, )
        market_regime_mu: Mean of market regime clusters. (states_num, )
        market_regime_sigma: Sigma of market regime clusters. (states_num, )
        stock_post_mu: Posterior mean of stock factors. (stock_factor_num, )
        stock_post_sigma: Posterior sigma of stock factors. (stock_factor_num, )
        stock_prior_mu: Prior mean of stock factors. (stock_factor_num, )
        stock_prior_sigma: Prior sigma of stock factors. (stock_factor_num, )
        rt_mu: Mean of future returns with shape of (stock_size, )
        rt_sigma: Sigma of future returns with shape of (stock_size, )
    """
    
    # reconstruction loss
    NLL = F.mse_loss(rt, ft)
    
    # KLD between market posterior and prior
    KLD_market = gaussian_kl_div(market_post_mu, market_post_sigma, market_prior_mu, market_prior_sigma)
    
    # KLD between stock posterior and prior
    KLD_stock = gaussian_kl_div(stock_post_mu, stock_post_sigma, stock_prior_mu, stock_prior_sigma)

    # Ensure that market regime clusters are as distinctly separated as possible.
    ws2d_market_regime = 0
    num_states = market_regime_mu.shape[0]
    for x,y in combinations(range(num_states),2):
        ws2d_market_regime += gaussian_ws_2_dis(market_regime_mu[x], market_regime_sigma[x], market_regime_mu[y], market_regime_sigma[y])
        
    return NLL + KLD_market + KLD_stock + 1/ws2d_market_regime


@njit(fastmath=True)
def cal_rdv(series):
    return  series[-1] / np.mean(series)

@njit(fastmath=True)
def calc_realized_volatility(series):
    return np.sqrt(np.sum(series**2))

@njit(fastmath=True)
def calc_cumu_rts(series):
    return series[-1]/series[0] - 1

def calc_features(df,window):
    # the levels of relative daily value
    rdv = df.rolling(window).apply(cal_rdv, engine='numba', raw=True)
    rdv = rdv.add_suffix(f'_rdv_{window}')
    
    # Market Volatility
    rv = df.rolling(window).apply(calc_realized_volatility, engine='numba', raw=True)
    rv = rdv.add_suffix(f'_rv_{window}')

    # Market Momentums
    cumu_rts = df.rolling(window).apply(calc_cumu_rts, engine='numba', raw=True)
    cumu_rts = cumu_rts.add_suffix(f'_cumu_rts_{window}')
    return pd.concat([rdv,rv,cumu_rts],axis=1).sort_index(level=[0,1])

def calc_market_feeatures(df):
    m_feature_20 = calc_features(df,20)
    m_feature_40 = calc_features(df,40)
    m_feature_60 = calc_features(df,60)
    m_feature_80 = calc_features(df,80)
    m_feature = pd.concat([df,m_feature_20,m_feature_40,m_feature_60,m_feature_80],axis=1).dropna()
    return m_feature