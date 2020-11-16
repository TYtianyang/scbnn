# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 13:00:57 2020

@author: Tianyang
"""

import os
os.chdir('C:/Users/Tianyang/Documents/Option Project/source')
exec(open('initialization.py').read())

# -------------------------------------------------------------------------------------
# define class for B splines on k
class B_splines():
    
    # function __init__
    def __init__(self,data,newx,cons_newx=True,penalty=1,auto_tic=True,tic=1,order=3,s=500,
                 plot=True):
        
        self.auto_tic = auto_tic
        self.tic = tic
        self.s = s
        self.order = order
        self.plot = plot
        self.prediction = np.zeros((newx.shape[0]))
        self.penalty = penalty
        self.cons_newx = cons_newx
        self.newx = newx
        
        self.unique_t = np.unique(data['busT2M'])
        
        self.sub_data_dic = {}
        self.newx_dic = {}
        prediction_x = []
        for i in range(len(self.unique_t)):
            sub_data = data[data['busT2M']==self.unique_t[i]].groupby(['moneyness']).mean().reset_index()
            self.sub_data_dic[str(i)] = sub_data
            self.newx_dic[str(i)] = newx[newx['busT2M']==self.unique_t[i]]
            prediction_x.append(self.newx_dic[str(i)])
        
        self.prediction_x = np.concatenate(tuple(prediction_x))
        
        # redundant. just for testing purpose
        sub_data = self.sub_data_dic[str(0)]
        sub_newx = self.newx_dic[str(0)]
        self.trainx = np.array(sub_data['moneyness']).reshape((-1,1))
        self.trainy = np.array(sub_data['totalvar']).reshape((-1,1))
        self.predx = np.array(sub_newx['moneyness']).reshape((-1,1))
    
    # function constr_grid: initiate grid points for constraints
    def constr_grid(self):
        k_max = np.max(self.trainx)
        k_min = np.min(self.trainx)
        self.grid = np.linspace(k_min,k_max,100).reshape((-1,1))

    # function init_tic: initiate tics (knots)
    def init_tic(self):
        if (self.auto_tic):
            self.tic = interpolate.splrep(self.trainx,self.trainy,s=self.s)[0]
    
    # function init_N: initiate B spline basis for training data
    def init_N(self):
        basis = bspline.Bspline(self.tic,self.order)
        self.N = basis.collmat(np.squeeze(self.trainx))
        self.Np = basis.collmat(np.squeeze(self.trainx),deriv_order=1)
        self.Npp = basis.collmat(np.squeeze(self.trainx),deriv_order=2)
    
    # function init_newN: initiate B splines basis for prediction data
    def init_newN(self):
        basis = bspline.Bspline(self.tic,self.order)
        self.newN = basis.collmat(np.squeeze(self.predx))
    
    # function init_n: initiate B spline basis for grid
    def init_n(self):
        basis = bspline.Bspline(self.tic,self.order)
        if (self.cons_newx):
            self.n = basis.collmat(np.squeeze(self.predx))
            self.np = basis.collmat(np.squeeze(self.predx),deriv_order=1)
            self.npp = basis.collmat(np.squeeze(self.predx),deriv_order=2)
        else:
            self.n = basis.collmat(np.squeeze(self.grid))
            self.np = basis.collmat(np.squeeze(self.grid),deriv_order=1)
            self.npp = basis.collmat(np.squeeze(self.grid),deriv_order=2)
        
    # function loss_f: loss function for LSQ
    def loss_f(self,theta):
        return (self.trainy - self.N.dot(theta.reshape((-1,1)))).T.dot(
                self.trainy - self.N.dot(theta.reshape((-1,1))))[0]
        
    # function loss_grad: loss gradient for LSQ
    def loss_grad(self,theta):
        return (-2*self.N.T.dot(self.trainy - self.N.dot(theta.reshape((-1,1))))).reshape((-1))
        
    # function nl_f: nonlinear constraint penalty function
    def nl_f(self,theta):
        N = self.n
        Np = self.np
        Npp = self.npp
        if (self.cons_newx):
            k_vec = self.predx
        else:
            k_vec = self.grid
        w = N.dot(theta.reshape((-1,1)))
        wp = Np.dot(theta.reshape((-1,1)))
        wpp = Npp.dot(theta.reshape((-1,1)))
        
        return ((1 - k_vec*wp/(2*(w+0.0001)))**2 - wp/4*(1/(w+0.0001) + 1/4) + wpp/2).reshape((-1))
    
    # function nl_grad: nonlinear cnonstraint penalty gradient function
    def nl_grad(self,theta):
        N = self.n
        Np = self.np
        Npp = self.npp
        if (self.cons_newx):
            k_vec = self.predx
        else:
            k_vec = self.grid
        w = N.dot(theta.reshape((-1,1)))
        wp = Np.dot(theta.reshape((-1,1)))
        
        grad1 = 2*(1 - k_vec*wp/(2*(w+0.0001))).T*(-k_vec/2).T*(Np.T*w.T - N.T*wp.T)/((w+0.0001).T**2)
        grad2 =  - (N.T/4*(1/(w+0.0001).T + 1/4) + wp.T/4*(- N.T/(w+0.0001).T**2))
        grad3 = Npp.T/2
        return grad1 + grad2 + grad3
    
    # function soft_loss_f: loss + penalty function
    def soft_loss_f(self,theta):
        loss_1 = self.loss_f(theta)
        loss_2 = self.penalty*np.sum(np.maximum(-self.nl_f(theta),0))
        return loss_1 + loss_2
    
    # function soft_loss_grad: loss + penalty gradient 
    def soft_loss_grad(self,theta):
        grad_1 = self.loss_grad(theta)
        grad_2 = self.penalty*np.sum(self.nl_grad(theta)*np.where(np.maximum(-self.nl_f(theta),0)!=0,1,0)
                ,axis=1)
        return grad_1 + grad_2
    
    # function init_every: init everything before fitting
    def init_every(self,i):
        sub_data = self.sub_data_dic[str(i)]
        self.trainx = np.array(sub_data['moneyness']).reshape((-1,1))
        self.predx = np.array(self.newx_dic[str(i)]['moneyness']).reshape((-1,1))
        self.trainy = np.array(sub_data['totalvar']).reshape((-1,1))
        
        self.constr_grid()
        self.init_tic()
        self.init_newN()
        self.init_N()
        self.init_n()
    
    # function lsq_fit_all: fit all the models without constraints
    def lsq_fit_all(self):
        self.theta_dic = {}
        prediction_list = []
        for i in range(len(self.unique_t)):
            self.init_every(i)
            theta = np.squeeze(np.linalg.inv(self.N.T.dot(self.N)).dot(self.N.T).dot(self.trainy))
            
            if (self.plot):
                ind = np.argsort(self.trainx[:,0])
                plt.plot(self.trainx[ind,0],self.N.dot(theta.reshape((-1,1)))[ind,0],'red')
                plt.plot(self.trainx[ind,0],self.trainy[ind,0],'blue')
                plt.show()
            
            self.theta_dic[str(i)] = theta
            prediction_list.append(self.newN.dot(theta.reshape((-1,1))).reshape((-1)))
        
        self.prediction = np.concatenate(tuple(prediction_list))
            
    # function cons_fit_all: refit all the models with constraints after lsq_fit_all
    def cons_fit_all(self):
        self.cons_theta_dic = {}
        prediction_list = []
        for i in range(len(self.unique_t)):
            self.init_every(i)
            theta = self.theta_dic[str(i)]
            
            if np.all(self.nl_f(theta)>=0):
                cons_theta = theta
            else:
                res = minimize(self.soft_loss_f, theta, method='BFGS', jac=self.soft_loss_grad,options={'maxiter':1000})
                cons_theta = res.x
            
            if (self.plot):
                ind = np.argsort(self.trainx[:,0])
                plt.plot(self.trainx[ind,0],self.N.dot(cons_theta.reshape((-1,1)))[ind,0],'red')
                plt.plot(self.trainx[ind,0],self.trainy[ind],'blue')
                plt.show()
            
            self.cons_theta_dic[str(i)] = cons_theta
            prediction_list.append(self.newN.dot(cons_theta.reshape((-1,1))).reshape((-1)))
        
        self.prediction = np.concatenate(tuple(prediction_list))
        
    # function evaluation: evaluate on predx
    def evaluation(self):
        self.lsq_fit_all()
        self.cons_fit_all()
        mse = np.average((np.array(self.newx['totalvar']) - self.prediction)**2)
        c5 = 0
        temp = self.cons_newx
        self.cons_newx = True
        for i in range(len(self.unique_t)):
            self.init_every(i)
            c5 = c5 + np.sum(np.maximum(-self.nl_f(self.cons_theta_dic[str(i)]),0))
        self.cons_newx = temp
        return((mse,c5))
        
    
if __name__ == '__main__':
    m = B_splines(data,newx=data.sample(n=200,random_state=1),cons_newx=False,penalty=1,auto_tic=True,tic=1,order=3,
                     plot=False)
    m.evaluation()