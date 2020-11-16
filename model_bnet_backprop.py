# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 10:43:41 2020

@author: Tianyang
"""

import os
os.chdir('C:/Users/Tianyang/Documents/Option Project/source')
exec(open('initialization.py').read())

# -------------------------------------------------------------------------------------
### define class for NN based bayes network with back prop

PI = 0.5
SIGMA_1 = torch.FloatTensor([math.exp(-0)])
SIGMA_2 = torch.FloatTensor([math.exp(-6)])

## define Gaussian: the Gaussian coefficients generators
class Gaussian(object):
    def __init__(self, mu, rho):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(0,1)
    
    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))
    
    def sample(self):
        epsilon = self.normal.sample(self.rho.size())
        return self.mu + self.sigma * epsilon
    
    def log_prob(self, input):
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.sigma)
                - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()
        
## define ScaleMixtureGaussian: the Gaussian mixture prior
class ScaleMixtureGaussian(object):
    def __init__(self, pi, sigma1, sigma2):
        super().__init__()
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.gaussian1 = torch.distributions.Normal(0,sigma1)
        self.gaussian2 = torch.distributions.Normal(0,sigma2)
    
    def log_prob(self, input):
        prob1 = torch.exp(self.gaussian1.log_prob(input))
        prob2 = torch.exp(self.gaussian2.log_prob(input))
        return (torch.log(self.pi * prob1 + (1-self.pi) * prob2)).sum()
    
## define BayesianLinear: the single layer Bayes network
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5,-4))
        self.weight = Gaussian(self.weight_mu, self.weight_rho)
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5,-4))
        self.bias = Gaussian(self.bias_mu, self.bias_rho)
        # Prior distributions
        self.weight_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
        self.bias_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)

    def forward(self, input, sample="sample"):
        if sample=="sample":
            try:
                del self.weight_value
                del self.bias_value
            except:
                pass
            weight = self.weight.sample()
            bias = self.bias.sample()
            self.weight_value = weight
            self.bias_value = bias
        if sample=="repeat":
            weight = self.weight_value
            bias = self.bias_value
        if sample=="mean":
            weight = self.weight_mu
            bias = self.bias_mu
            self.weight_value = weight
            self.bias_value = bias
            
        self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
        self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)

        return F.linear(input, weight, bias)

## define BayesianNetwork: the multi layers Bayes network
class BayesianNetwork(nn.Module):
    def __init__(self, data, test,
                 neuron = 40, penalty = 1, 
                 bayes_lr = 1e-4, bayes_step = 100):
        super().__init__()
        self.l1 = BayesianLinear(2, neuron)
        self.l2 = BayesianLinear(neuron, neuron)
        self.l3 = BayesianLinear(neuron, neuron)
        self.l4 = BayesianLinear(neuron, 1)
        
        self.sample_mode = "sample"
        self.penalty = penalty
        self.bayes_lr = bayes_lr
        self.bayes_step = bayes_step
        
        data = data.groupby(['moneyness','busT2M']).mean().reset_index()
        tensor = torch.from_numpy(data.values).float()
        self.x = tensor[:,0:2]
        self.y = tensor[:,2].reshape((-1,1))
        self.test = torch.from_numpy(test.values).float()
        self.grid_gen()
    
    # define grid_gen: generate constraining grids
    def grid_gen(self):
        x = self.x
        t_max = x.max(0).values[1]
        k_max = x.max(0).values[0]
        k_min = x.min(0).values[0]
        
        t_grid = np.exp(np.linspace(np.log(1/365),np.log(t_max+1),100)).reshape((-1))
        k_grid = (np.linspace(np.cbrt(2*k_min),np.cbrt(2*k_max),100)**3).reshape((-1))
        t_mgrid,k_mgrid = np.meshgrid(t_grid,k_grid)
        self.constr_grid = torch.from_numpy(np.concatenate((k_mgrid.reshape((-1,1)),t_mgrid.reshape((-1,1))),1)).float()
    
    # define forward: pass on network prediction
    def forward(self, x):
        x = F.relu(self.l1(x, self.sample_mode))
        x = F.relu(self.l2(x, self.sample_mode))
        x = F.relu(self.l3(x, self.sample_mode))
        x = F.relu(self.l4(x, self.sample_mode))
        return x
    
    # define w_first_derivative_ReLU: first order derivative of w on x, on ReLU activation
    def w_first_derivative_ReLU(self):
        x = self.constr_grid
        ReLU_output = []
        x = F.relu(self.l1(x, sample="repeat"))
        ReLU_output.append(x)
        x = F.relu(self.l2(x, sample="repeat"))
        ReLU_output.append(x)
        x = F.relu(self.l3(x, sample="repeat"))
        ReLU_output.append(x)
        x = F.relu(self.l4(x, sample="repeat"))
        ReLU_output.append(x)
        
        linear_paras = [self.l1.weight_value.T, self.l2.weight_value.T, self.l3.weight_value.T, self.l4.weight_value.T]

        for i in range(4):
            output = ReLU_output[i]
            para = linear_paras[i]
            
            zeros = torch.zeros(output.shape)
            ones = torch.ones(output.shape)
            output_binary = torch.where(output>0,ones,zeros)
            output_tensor = torch.diag_embed(output_binary)
            
            para_tensor = para.unsqueeze(0).expand(output_tensor.shape[0],para.shape[0],para.shape[1])
            B_new = torch.bmm(para_tensor,output_tensor)
            
            if i==0:
                B = B_new
            else:
                B = torch.bmm(B,B_new)
        
        B = B.squeeze(2)
        self.w_first_derivative = B
        
    # define loss_c4: implement Monotonicity loss function
    def loss_c4(self):
        p = self.w_first_derivative
        p_t = torch.mm(p,torch.tensor([[0],[1]],dtype=torch.float32))
        
        c4_loss = torch.clamp(-p_t,0,np.inf)
        return(self.penalty*c4_loss.sum())
            
    # define loss_c5: implement Durrleman's condition loss function
    def loss_c5(self):
        x = self.constr_grid
        k = x[:,0].unsqueeze(1)
        w = self.forward(x) + 1e-5
        p = self.w_first_derivative
        p_k = torch.mm(p,torch.tensor([[1],[0]],dtype=torch.float32))
        
        c5 = (1 - k*p_k/(2*w))**2 - p_k/4*(1/w + 1/4)
        c5_loss = torch.clamp(-c5,0,np.inf)
        return(self.penalty*c5_loss.sum())
    
    # define log_prior: the log of prior distribution
    def log_prior(self):
        return self.l1.log_prior \
               + self.l2.log_prior \
               + self.l3.log_prior \
               + self.l4.log_prior
    
    # define log_variational_posterior: the log of posterior approximation
    def log_variational_posterior(self):
        return self.l1.log_variational_posterior \
               + self.l2.log_variational_posterior \
               + self.l3.log_variational_posterior \
               + self.l4.log_variational_posterior
    
    # define elbo: the ultimate loss function
    def elbo(self):
        yhat = self.forward(self.x)
        self.w_first_derivative_ReLU()
        
        log_prior = self.log_prior()
        log_variational_posterior = self.log_variational_posterior()
        negative_log_likelihood = (yhat - self.y).pow(2).sum()
        loss_c4 = self.loss_c4()
        loss_c5 = self.loss_c5()

        loss = log_variational_posterior - log_prior + negative_log_likelihood + loss_c4 + loss_c5
        return loss, log_prior, log_variational_posterior, negative_log_likelihood, loss_c4, loss_c5

    # define bayes_fit: the fitting function
    def bayes_fit(self):
        if self.sample_mode=="mean":
            optimizer = torch.optim.Adam([self.l1.weight_mu,self.l2.weight_mu,self.l3.weight_mu,self.l4.weight_mu], lr=self.bayes_lr)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.bayes_lr)
        tqdm_list = tqdm(range(self.bayes_step))
        for t in tqdm_list:
            loss, log_prior, log_variational_posterior, negative_log_likelihood, loss_c4, loss_c5 = self.elbo()
            self.zero_grad()
            loss.backward()
            optimizer.step()
            tqdm_list.set_description("Prior " + str(log_prior.item()) + 
                                      " Post " + str(log_variational_posterior.item()) + 
                                      " Likeli " + str(negative_log_likelihood.item()) + 
                                      " c4 " + str(loss_c4.item()) +
                                      " c5 " + str(loss_c5.item()))
            
    # define evaluation: evaluate the model based on test data
    def evaluation(self, sample_size = 1000):
        y_pred_list = []
        mse_list = []
        c4_list = []
        c5_list = []
        
        for t in tqdm(range(sample_size)):        
            y_pred = self.forward(self.test[:,0:2])
            mse = (y_pred - self.test[:,2]).pow(2).mean().detach().item()
            
            temp = self.constr_grid
            self.constr_grid = self.test[:,0:2]
            self.w_first_derivative_ReLU()
            c4 = (self.loss_c4().detach()/self.penalty).item()
            c5 = (self.loss_c5().detach()/self.penalty).item()
            self.constr_grid = temp
            
            if c4==0 and c5==0:
                y_pred_list.append(y_pred)
                mse_list.append(mse)
                c4_list.append(c4)
                c5_list.append(c5)
            
        self.y_pred_list = y_pred_list
        self.mse_list = mse_list
        self.c4_list = c4_list
        self.c5_list = c5_list
            
        return(np.average(self.mse_list), np.average(self.c4_list), np.average(self.c5_list), len(self.c5_list))
        
            

if __name__ == '__main__':
    torch.manual_seed(4)
    PI = 0.5
    SIGMA_1 = torch.FloatTensor([math.exp(-0)])
    SIGMA_2 = torch.FloatTensor([math.exp(-6)])
    model = BayesianNetwork(train, test, neuron=50, bayes_step = 500, bayes_lr = 1e-4)
    model.sample_mode = "mean"
    model.bayes_fit()
    model.sample_mode = "sample"
    model.bayes_step = 1000
    model.bayes_fit()
    print(model.evaluation())
    
    y_pred_mat = torch.cat(model.y_pred_list,axis=1).detach()
    y_pred_mat = np.array(y_pred_mat).T
    q_pred_mat = np.quantile(y_pred_mat, [0.05, 0.95], axis=0)
    
    x = np.array(test[['moneyness','busT2M']])
    y = np.array(test['totalvar']).reshape((-1,1))
    
    ax = plt.axes(projection='3d')
    ax.scatter3D(x[:,0], x[:,1], y.reshape((-1)), c='blue',marker='.')
    ax.scatter3D(x[:,0], x[:,1], q_pred_mat[0,:], c='red',marker='.')
    ax.scatter3D(x[:,0], x[:,1], q_pred_mat[1,:], c='red',marker='.')
    ax.set_xlabel('Moneyness')
    ax.set_ylabel('T2M')
    ax.set_zlabel('Total Variance')
    ax.locator_params(axis='y',nbins=5)
    ax.locator_params(axis='x',nbins=5)
    ax.view_init(elev = None, azim=20)
    plt.show()
    
    
    
    
    
    



