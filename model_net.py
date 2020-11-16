# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 13:01:43 2020

@author: Tianyang
"""

import os
os.chdir('C:/Users/Tianyang/Documents/Option Project/source')
exec(open('initialization.py').read())

# -------------------------------------------------------------------------------------
# define class for NN based models
class net(nn.Module):
    
    # initiate network settings
    def __init__(self,data,test,
                 layer=4,neuron=40,
                 original_lr = 1e-4, constr_lr = 1e-5, 
                 original_step = 300, constr_step = 300,
                 penalty = 10,
                 plot=True):
        super(net, self).__init__()
        self.layer = layer
        self.neuron = neuron
        self.original_lr = original_lr
        self.constr_lr = constr_lr
        self.original_step = original_step
        self.constr_step= constr_step
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
            
        if (self.plot):
            ax = plt.axes(projection='3d')
            ax.scatter3D(self.x[:,0], self.x[:,1], self.y.reshape((-1)), c='blue',marker='.')
            ax.set_xlabel('Moneyness')
            ax.set_ylabel('T2M')
            ax.set_zlabel('Total Variance')
            ax.locator_params(axis='y',nbins=5)
            ax.locator_params(axis='x',nbins=5)
            ax.view_init(elev = None, azim=20)
            plt.show()
            
            bx = plt.axes(projection='3d')
            bx.scatter3D(self.x[:,0], self.x[:,1], self.forward(self.x).detach(), c='red',marker='.')
            bx.set_xlabel('Moneyness')
            bx.set_ylabel('T2M')
            bx.set_zlabel('Total Variance')
            bx.locator_params(axis='y',nbins=5)
            bx.locator_params(axis='x',nbins=5)
            bx.view_init(elev = None, azim=20)
            bx.set_zlim(ax.get_zlim())
            bx.set_ylim(ax.get_ylim())
            bx.set_xlim(ax.get_xlim())
            plt.show()
            
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
            
        if (self.plot):
            ax = plt.axes(projection='3d')
            ax.scatter3D(self.x[:,0], self.x[:,1], self.y.reshape((-1)), c='blue',marker='.')
            ax.set_xlabel('Moneyness')
            ax.set_ylabel('T2M')
            ax.set_zlabel('Total Variance')
            ax.locator_params(axis='y',nbins=5)
            ax.locator_params(axis='x',nbins=5)
            ax.view_init(elev = None, azim=20)
            plt.show()
            
            bx = plt.axes(projection='3d')
            bx.scatter3D(self.x[:,0], self.x[:,1], self.forward(self.x).detach(), c='red',marker='.')
            bx.set_xlabel('Moneyness')
            bx.set_ylabel('T2M')
            bx.set_zlabel('Total Variance')
            bx.locator_params(axis='y',nbins=5)
            bx.locator_params(axis='x',nbins=5)
            bx.view_init(elev = None, azim=20)
            bx.set_zlim(ax.get_zlim())
            bx.set_ylim(ax.get_ylim())
            bx.set_xlim(ax.get_xlim())
            plt.show()
            
    # define evaluation: evaluate the model based on test data
    def evaluation(self):
        y_pred = self.forward(self.test[:,0:2])
        mse = (y_pred - self.test[:,2]).pow(2).mean().detach()
        
        temp = self.constr_grid
        self.constr_grid = self.test[:,0:2]
        self.w_first_derivative_ReLU()
        c4 = self.loss_c4().detach()/self.penalty
        c5 = self.loss_c5().detach()/self.penalty
        self.constr_grid = temp
        return((mse.item(),c4.item(),c5.item()))
        
        
        
