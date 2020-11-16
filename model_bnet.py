# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 11:57:42 2020

@author: Tianyang
"""

import os
os.chdir('C:/Users/Tianyang/Documents/Option Project/source')
exec(open('initialization.py').read())

# -------------------------------------------------------------------------------------
# define class for NN based bayes network
class bnet(nn.Module):
    
    # initiate network settings
    def __init__(self,data,test,
                 layer=4,neuron=40,
                 original_lr = 1e-4, constr_lr = 1e-4, bayes_lr = 1e-3,
                 original_step = 300, constr_step = 300, bayes_step = 100,
                 bayes_batchp = 0.2,
                 penalty = 10,
                 plot=True):
        super(bnet, self).__init__()
        self.layer = layer
        self.neuron = neuron
        self.original_lr = original_lr
        self.constr_lr = constr_lr
        self.bayes_lr = bayes_lr
        self.original_step = original_step
        self.constr_step= constr_step
        self.bayes_step = bayes_step
        self.bayes_batch_size = round(data.shape[0]*bayes_batchp)
        self.penalty = penalty
        self.plot = plot
        
        data = data.groupby(['moneyness','busT2M']).mean().reset_index()
        tensor = torch.from_numpy(data.values).float()
        self.x = tensor[:,0:2]
        self.y = tensor[:,2].reshape((-1,1))
        self.test = torch.from_numpy(test.values).float()
        self.grid_gen()

        self.hidden = nn.ModuleList()
        self.hidden.append(nn.Linear(2,neuron))
        self.hidden.append(nn.ReLU())
        for k in range(layer-1):
            self.hidden.append(nn.Linear(neuron,neuron))
            self.hidden.append(nn.ReLU())
        self.hidden.append(nn.Linear(neuron,1))
        self.hidden.append(nn.ReLU())
            
    # define forward: the forward function for model
    def forward(self,x):
        for i, f in enumerate(self.hidden):
            x = f(x)
        return x
    
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
    
    # define w_first_derivative_ReLU: first order derivative of w on x, on ReLU activation
    def w_first_derivative_ReLU(self):
        ReLU_output = []
        x = self.constr_grid
        for i, f in enumerate(self.hidden):
            x = f(x)
            if (i+1)%2==0:
                ReLU_output.append(x)
        linear_paras = [list(self.hidden.parameters())[2*j].T for j in range(self.layer)]
        
        for i in range(self.layer):
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
            
        last_output = ReLU_output[-1]
        last_para = list(self.hidden.parameters())[2*self.layer].T
        zeros = torch.zeros(last_output.shape)
        ones = torch.ones(last_output.shape)
        last_output_binary = torch.where(last_output>0,ones,zeros)
        last_output_tensor = torch.diag_embed(last_output_binary)
        last_para_tensor = last_para.unsqueeze(0).expand(output_tensor.shape[0],last_para.shape[0],last_para.shape[1])
        B_last = torch.bmm(last_para_tensor,last_output_tensor)
        
        B = torch.bmm(B,B_last)
        
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

    # define loss_ori: original loss function of fitting
    def loss_ori(self):
        y_pred = self.forward(self.x)
        ori_loss = (y_pred - self.y).pow(2).sum()
        return(ori_loss)
            
    # define fit_0: initialize fitting process without constraints
    def fit_0(self):
        optimizer = torch.optim.Adam(self.hidden.parameters(), lr=self.original_lr)
        def closure_original():
            loss = self.loss_ori()
            optimizer.zero_grad()
            loss.backward()
            return loss
        for t in range(self.original_step):
            optimizer.step(closure_original)
        self.hidden_backup = self.hidden
            
    # define fit: initialize fitting process with constraints
    def fit(self):
        optimizer = torch.optim.Adam(self.hidden.parameters(), lr=self.constr_lr)
        def closure_constr():
            self.w_first_derivative_ReLU()
            loss = self.loss_ori() + self.loss_c4() +  self.loss_c5()
            optimizer.zero_grad()
            print(loss)
            loss.backward()
            return loss
        for t in range(self.constr_step):
            optimizer.step(closure_constr)
        self.hidden_backup = self.hidden
            
    # define gen_theta: generate bayes posterior samples (assuming sigma_mat = np.eye)
    def gen_theta(self):
        theta_list = []
        criterion = torch.nn.MSELoss()
        for t in tqdm(range(self.bayes_step)):
            ind = random.sample(list(np.arange(0,self.x.shape[0])), self.bayes_batch_size)
            sub_x = self.x[ind,:]
            sub_y = self.y[ind,:]
            
            self.zero_grad()
            sub_yhat = self.forward(sub_x)
            loss = criterion(sub_yhat, sub_y)
            loss.backward()
            
            theta = list(self.hidden.parameters())
            for i in range(len(theta)):
                log_prior = - theta[i]
                log_likeli = self.x.shape[0] / self.bayes_batch_size * theta[i].grad
                noise = Variable(theta[i].data.new(theta[i].size()).normal_(0, self.bayes_lr)).type_as(theta[i])
                delta_theta = self.bayes_lr*(log_prior + log_likeli) + noise
                theta[i] = delta_theta + theta[i]
                
            theta_list.append(theta)            
        
        self.theta_list = theta_list
            
    # define train_prune: prune the thetas to accomodate c4 c5
    def train_prune(self):
        theta_pruned = []
        for t in tqdm(range(len(self.theta_list))):
            theta = self.theta_list[t]
            for i in range(int(len(theta)/2)):
                self.hidden[2*i].weight = nn.parameter.Parameter(theta[2*i])
                self.hidden[2*i].bias = nn.parameter.Parameter(theta[2*i+1])
            self.w_first_derivative_ReLU()
            loss_c4 = self.loss_c4()
            loss_c5 = self.loss_c5()
            if loss_c4.item()==0 and loss_c5.item()==0:
                theta_pruned.append(self.theta_list[t])
        
        self.theta_list = theta_pruned
        
    # define gen_theta_train_prune: generate bayes pruned posterior samples (assuming sigma_mat = np.eye)
    def gen_theta_train_prune(self):
        theta_list = []
        criterion = torch.nn.MSELoss()
        for t in tqdm(range(self.bayes_step)):
            ind = random.sample(list(np.arange(0,self.x.shape[0])), self.bayes_batch_size)
            sub_x = self.x[ind,:]
            sub_y = self.y[ind,:]
            
            self.zero_grad()
            sub_yhat = self.forward(sub_x)
            loss = criterion(sub_yhat, sub_y)
            loss.backward()
            
            temp_bayes_lr = self.bayes_lr/(t+1)
            hidden_0 = self.hidden
            
            while True:
                theta = list(self.hidden.parameters()).copy()
                theta_data = []
                for i in range(len(theta)):
                    log_prior = - theta[i]
                    log_likeli = self.x.shape[0] / self.bayes_batch_size * theta[i].grad
                    noise = Variable(theta[i].data.new(theta[i].size()).normal_(0, temp_bayes_lr)).type_as(theta[i])
                    delta_theta = temp_bayes_lr*(log_prior + log_likeli) + noise
                    theta_data.append(delta_theta.data + theta[i].data)
                    
                for i in range(int(len(theta)/2)):
                    self.hidden[2*i].weight.data = nn.parameter.Parameter(theta_data[2*i])
                    self.hidden[2*i].bias.data = nn.parameter.Parameter(theta_data[2*i+1])
                self.w_first_derivative_ReLU()
                loss_c4 = self.loss_c4()
                loss_c5 = self.loss_c5()
                
                if loss_c4.item()==0 and loss_c5.item()==0:
                    theta_list.append(theta_data)
                    hidden_0 = self.hidden
                    break
                else:
                    self.hidden = hidden_0
                    
#                print(str(criterion(sub_yhat, sub_y).item()) + ' ' + str(loss_c4.item()) + ' ' + str(loss_c5.item()))
                    
        self.theta_list = theta_list
            
    # define evaluation: evaluate the model based on test data
    def evaluation(self):
        y_pred_list = []
        mse_list = []
        c4_list = []
        c5_list = []
        
        for t in tqdm(range(len(self.theta_list))):
            theta_data = self.theta_list[t]
            for i in range(int(len(theta_data)/2)):
                self.hidden[2*i].weight.data = nn.parameter.Parameter(theta_data[2*i])
                self.hidden[2*i].bias.data = nn.parameter.Parameter(theta_data[2*i+1])
        
            y_pred = self.forward(self.test[:,0:2])
            mse = (y_pred - self.test[:,2]).pow(2).mean().detach().item()
            
            temp = self.constr_grid
            self.constr_grid = self.test[:,0:2]
            self.w_first_derivative_ReLU()
            c4 = (self.loss_c4().detach()/self.penalty).item()
            c5 = (self.loss_c5().detach()/self.penalty).item()
            self.constr_grid = temp
            
            y_pred_list.append(y_pred)
            mse_list.append(mse)
            c4_list.append(c4)
            c5_list.append(c5)
            
        self.y_pred_list = y_pred_list
        self.mse_list = mse_list
        self.c4_list = c4_list
        self.c5_list = c5_list
            
        return(np.average(self.mse_list), np.average(self.c4_list), np.average(self.c5_list))
            
        
    
if __name__ == '__main__':
    model = bnet(train, test, bayes_step = 100, bayes_lr = 1e-3)
    model.load_state_dict(torch.load('pretrain'))
    model.fit()
#    model.gen_theta()
#    model.train_prune()
    model.fit_0()
    model.gen_theta_train_prune()
    model.evaluation()
        
    y_pred_mat = torch.cat(model.y_pred_list,axis=1).detach()
    y_pred_mat = np.array(y_pred_mat).T
    q_pred_mat = np.quantile(y_pred_mat, [0.3, 0.7], axis=0)
    
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
    