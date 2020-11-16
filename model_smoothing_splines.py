# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 12:59:40 2020

@author: Tianyang
"""

import os
os.chdir('C:/Users/Tianyang/Documents/Option Project/source')
exec(open('initialization.py').read())

# -------------------------------------------------------------------------------------
# define class for smoothing splines on k
class smoothing_splines():
    
    # class object for smoothing spline. For each unique t, fit smoothing spline on k
    # Input: "data": [n x 3], DataFrame; "newx": [n x 2], DataFrame
    # Parameters: "smooth": smooth parameter
    
    # function __init__: initiate core data. knots: 0,1,2,...,K+1
    def __init__(self,data,newx,cons_newx = True,penalty=1,even_knots= True,
                 smooth_vec=1,cv=True,cv_low=0.01,cv_high=5,reweight=False,plot=True):
            
        self.plot = plot
        self.prediction = np.zeros((newx.shape[0]))
        self.penalty = penalty
        self.reweight = reweight
        self.cons_newx = cons_newx
        self.newx = newx
        self.cv = cv
        self.cv_low = cv_low
        self.cv_high = cv_high
        self.unique_t = np.unique(data['busT2M'])
        self.even_knots = even_knots
        self.sub_data_dic = {}
        self.newx_dic = {}
        prediction_x = []
        for i in range(len(self.unique_t)):
            if (reweight):
                sub_data = data[data['busT2M']==self.unique_t[i]].groupby(['moneyness']).mean().reset_index()
            else:
                sub_data = data[data['busT2M']==self.unique_t[i]]
            self.sub_data_dic[str(i)] = sub_data
            self.newx_dic[str(i)] = newx[newx['busT2M']==self.unique_t[i]]
            prediction_x.append(self.newx_dic[str(i)])
#            print('Spline detect unique knots: ' + str(np.unique(sub_data['moneyness']).shape[0]) +
#                  ' for T2M ' + str(self.unique_t[i]))
        
        self.prediction_x = np.concatenate(tuple(prediction_x))
        
        if (cv):
            self.smooth_vec = np.zeros((len(self.unique_t)))
        else:
            self.smooth_vec = smooth_vec
        sub_data = self.sub_data_dic[str(0)]
        sub_newx = self.newx_dic[str(0)]
        self.smooth = smooth_vec[0]
        self.trainx = np.array(sub_data[['moneyness','busT2M']])
        self.predx = np.array(sub_newx)
        self.trainy = np.array(sub_data['totalvar']).reshape((-1,1))
        self.knots = np.unique(sub_data['moneyness'])
        self.K = self.knots.shape[0] - 2
        
    # function constr_grid: initiate grid points for constraints
    def constr_grid(self):
        k_max = np.max(self.knots[1:-1])
        k_min = np.min(self.knots[1:-1])
        self.grid = (np.linspace(np.cbrt(2*k_min),np.cbrt(2*k_max),100)**3).reshape((-1))
        
    # function init_N: initiate basis matrix for smoothing spline and linear component. n x (K + 4)
    def init_N(self):
        k_vec = self.trainx[:,0]
        N = np.zeros((self.trainx.shape[0],self.K+4))
        N[:,0] = 1
        N[:,1] = k_vec
        N[:,2] = k_vec**2
        N[:,3] = k_vec**3
        N[:,4:] = np.maximum(k_vec.reshape((-1,1)) - self.knots[1:-1],0)**3
        self.N = N
    
    # function init_newN: initiate basis matrix for newx.
    def init_newN(self):
        k_vec = self.predx[:,0]
        N = np.zeros((self.predx.shape[0],self.K+4))
        N[:,0] = 1
        N[:,1] = k_vec
        N[:,2] = k_vec**2
        N[:,3] = k_vec**3
        N[:,4:] = np.maximum(k_vec.reshape((-1,1)) - self.knots[1:-1],0)**3
        self.newN = N
        
    # function init_rough: initiate roughness matrix for smoothing spline. (K+4) x (K+4)
    def init_rough(self):
        e_up = max(self.knots)
        e_low = min(self.knots)
        omega_up = np.zeros((self.K+4,self.K+4))
        omega_low = omega_up.copy()
        
        omega_up[2:4,2:4] = np.array([[4*e_up,6*e_up**2],[6*e_up**2,12*e_up**3]])
        omega_low[2:4,2:4] = np.array([[4*e_low,6*e_low**2],[6*e_low**2,12*e_low**3]])
        
        omega_up[4:,2] = 6*(e_up - self.knots[1:-1])**2
        omega_up[2,4:] = 6*(e_up - self.knots[1:-1])**2
        
        omega_up[4:,3] = 12*e_up**3 - 18*self.knots[1:-1]*e_up**2
        omega_up[3,4:] = 12*e_up**3 - 18*self.knots[1:-1]*e_up**2
        omega_low[4:,3] = -6*self.knots[1:-1]**3
        omega_low[3,4:] = -6*self.knots[1:-1]**3
        
        e_i = np.ones((self.K,1)).dot(self.knots[1:-1].reshape((1,-1)))
        e_j = e_i.copy().T
        epe = e_i + e_j
        eme = e_i * e_j
        e_max = np.maximum(e_i,e_j)
        omega_up[4:,4:] = 12*e_up**3 - 18*epe*e_up**2 + 36*eme*e_up
        omega_low[4:,4:] = 12*e_max**3 - 18*epe*e_max**2 + 36*eme*e_max
        
        self.omega = omega_up - omega_low

    # function loss_f: define loss function.
    def loss_f(self,theta):
        resid = self.trainy - self.N.dot(theta.reshape((-1,1)))
        loss_1 = resid.T.dot(resid)
        loss_2 = self.smooth*(theta.reshape((-1,1)).T.dot(self.omega).dot(theta.reshape((-1,1))))
        return(loss_1 + loss_2)[0,0]
        
    # function loss_grad: define gradient for loss function
    def loss_grad(self,theta):
        resid = self.trainy - self.N.dot(theta.reshape((-1,1)))
        grad_1 = 2*resid.T.dot(-self.N).T
        grad_2 = np.zeros((self.K+4,1))
        grad_2[:,0] = 2*self.smooth*theta.reshape((-1,1)).T.dot(self.omega)
        return(np.squeeze(grad_1 + grad_2))
        
    # function loss_hess: define hessian matrix for loss function
    def loss_hess(self,theta):
        hess_1 = 2*self.N.T.dot(self.N)
        hess_2 = 2*self.smooth*self.omega
        return(hess_1 + hess_2)
    
    # function init_n: define grid basis on knots [K x (K+4)]
    def init_n(self):
        if (self.cons_newx):
            self.n = self.newN
        else:
            k_vec = self.grid
            n = np.zeros((k_vec.shape[0],self.K+4))
            n[:,0] = 1
            n[:,1] = k_vec
            n[:,2] = k_vec**2
            n[:,3] = k_vec**3
            n[:,4:] = np.maximum(k_vec.reshape((-1,1)) - self.knots[1:-1],0)**3
            self.n = n
        
    # function init_n_k: define first partial derivative of basis on knots [K x (K+5)]
    def init_n_k(self):
        if (self.cons_newx):
            k_vec = self.predx[:,0]
        else:
            k_vec = self.grid
        
        n_k = np.zeros((k_vec.shape[0],self.K+4))
        n_k[:,1] = 1
        n_k[:,2] = 2*k_vec
        n_k[:,3] = 3*k_vec**2
        n_k[:,4:] = 3*np.maximum(k_vec.reshape((-1,1)) - self.knots[1:-1],0)**2
        self.n_k = n_k
    
    # function init_n_kk: define second partial derivative of basis on knots [K x (K+5)]
    def init_n_kk(self):
        if (self.cons_newx):
            k_vec = self.predx[:,0]
        else:
            k_vec = self.grid
        n_kk = np.zeros((k_vec.shape[0],self.K+4))
        n_kk[:,2] = 2
        n_kk[:,3] = 6*k_vec
        n_kk[:,4:] = 6*np.maximum(k_vec.reshape((-1,1)) - self.knots[1:-1],0)
        self.n_kk = n_kk
    
    # function w: define w on k. [K] or newx
    def w(self,theta):
        return np.squeeze(self.n.dot(theta.reshape((-1,1))))
    
    # function w_k: define partial derivative of w on k. [K] or newx
    def w_k(self,theta):
        return np.squeeze(self.n_k.dot(theta.reshape((-1,1))))
    
    # function w_kk: define second partial derivative of w on k. [K] or newx
    def w_kk(self,theta):
        return np.squeeze(self.n_kk.dot(theta.reshape((-1,1))))        
    
    # function nl_f: define constraint value for nonlinear constriant
    def nl_f(self,theta):
        if (self.cons_newx):
            k_vec = self.predx[:,0]
        else:
            k_vec = self.grid
        w = self.w(theta)
        w_k = self.w_k(theta)
        w_kk = self.w_kk(theta)
        f = (1 - k_vec*w_k/(2*(w+0.0001)))**2 - w_k/4*(1/(w+0.0001)+1/4) + w_kk/2
        return f
    
    # function nlf_s: penalty function for soft constraint
    def nl_f_s(self,theta):
        f = np.maximum(-self.nl_f(theta),0)
        return np.sum(f)
    
    # function nl_grad: define jacobian matrix of constraint
    def nl_grad(self,theta):
        if (self.cons_newx):
            k_vec = self.predx[:,0]
        else:
            k_vec = self.grid
        w = self.w(theta)
        w_k = self.w_k(theta)
        
        jac_1 = 2*(1 - k_vec*w_k/(2*w)).reshape((-1,1))*((k_vec/2/(w)**2).reshape(-1,1))*((
                self.n.T.dot(w_k.reshape((-1,1))) - self.n_k.T.dot(w.reshape((-1,1)))).T)
        jac_2 = (-1)*((self.n.T*(1/w+1/4)).T + (- self.n.T/(w**2)*w_k/4).T)
        jac_3 = self.n_kk/2
        return jac_1 + jac_2 + jac_3
    
    # function nl_grad_s: penalty gradient for soft constraint
    def nl_grad_s(self,theta):
        f = np.maximum(-self.nl_f(theta),0)
        f = np.where(f.copy!=0,1,0)
        jac = -self.nl_grad(theta) * f.reshape((-1,1))
        return np.sum(jac,axis=0)
    
    # function soft_loss_f: define soft constraint loss function
    def soft_loss_f(self,theta):
        return self.loss_f(theta) + self.penalty*self.nl_f_s(theta)
    
    # function soft_loss_grad: define soft constraint loss gradient
    def soft_loss_grad(self,theta):
        return self.loss_grad(theta) + self.penalty*self.nl_grad_s(theta)
    
    # function init_every: init everything before fitting
    def init_every(self,i):
        sub_data = self.sub_data_dic[str(i)]
        self.trainx = np.array(sub_data[['moneyness','busT2M']])
        self.predx = np.array(self.newx_dic[str(i)])
        self.trainy = np.array(sub_data['totalvar']).reshape((-1,1))
        unique_k = np.unique(sub_data['moneyness'])
        if (self.even_knots):
            k_max = np.max(unique_k)
            k_min = np.min(unique_k)
            self.knots = np.linspace(k_min,k_max,round(unique_k.shape[0]/10))
        else:
            self.knots = unique_k
            
        self.K = self.knots.shape[0] - 2
        self.smooth = self.smooth_vec[i]
        
        self.constr_grid()
        self.init_N()
        self.init_newN()
        self.init_rough()
        self.init_n()
        self.init_n_k()
        self.init_n_kk()
    
    # function lsq_fit: fit the model without constraints for a single T2M
    def lsq_fit(self,lamb):
        theta = np.squeeze(np.linalg.inv(self.N.T.dot(self.N) + lamb*self.omega).dot(self.N.T).dot(self.trainy))
        return(theta)
        
    # function cv_lsq_fit: use CV grid search to find optimal smooth factor
    def cv_lsq_fit(self):
        cv_1 = np.linspace(self.cv_low,self.cv_high,100)
        dis_1 = []
        for lamb in cv_1:
            theta = np.squeeze(np.linalg.inv(self.N.T.dot(self.N) + lamb*self.omega).dot(self.N.T).dot(self.trainy))
            dis_1.append(np.average(self.N.dot(theta.reshape((-1,1)))**2))
        best_lamb = cv_1[np.argmin(dis_1)]
        return(best_lamb)
    
    # function lsq_fit_all: fit all the models without constraints
    def lsq_fit_all(self):
        self.theta_dic = {}
        prediction_list = []
        for i in range(len(self.unique_t)):
            self.init_every(i)
            if (self.cv):
                lamb = self.cv_lsq_fit()
                self.smooth_vec[i] = lamb
            else:
                lamb = self.smooth
            theta = self.lsq_fit(lamb)
            
            if (self.plot):
                ind = np.argsort(self.trainx[:,0])
                plt.plot(self.trainx[ind,0],self.N.dot(theta.reshape((-1,1)))[ind,0],'red')
                plt.plot(self.trainx[ind,0],self.trainy[ind],'blue')
                plt.show()
            
            self.theta_dic[str(i)] = theta
            prediction_list.append(self.newN.dot(theta.reshape((-1,1))))
        
        self.prediction = np.concatenate(tuple(prediction_list))
            
    # function cons_fit_all: refit all the models with constraints after lsq_fit_all
    def cons_fit_all(self):
        self.cons_theta_dic = {}
        prediction_list = []
        for i in range(len(self.unique_t)):
            self.init_every(i)
            theta = self.theta_dic[str(i)]
            
            if (np.all(self.w(theta)>=0) & (np.all(self.nl_f(theta)>=0))):
                cons_theta = theta
            else:
                res = minimize(self.soft_loss_f, theta, method='BFGS', jac=self.soft_loss_grad,options={'maxiter':100})
                cons_theta = res.x
            
            if (self.plot):
                ind = np.argsort(self.trainx[:,0])
                plt.plot(self.trainx[ind,0],self.N.dot(cons_theta.reshape((-1,1)))[ind,0],'red')
                plt.plot(self.trainx[ind,0],self.trainy[ind],'blue')
                plt.show()
            
            self.cons_theta_dic[str(i)] = cons_theta
            prediction_list.append(self.newN.dot(theta.reshape((-1,1))))
        
        self.prediction = np.concatenate(tuple(prediction_list))
        
    # function evaluation: evaluate on newx (test)
    def evaluation(self):
        self.lsq_fit_all()
        self.cons_fit_all()
        mse = np.average((np.array(self.newx['totalvar']) - self.prediction)**2)
        c5 = 0
        temp = self.cons_newx
        self.cons_newx = True
        for i in range(len(self.unique_t)):
            self.init_every(i)
            c5 = c5 + self.nl_f_s(self.cons_theta_dic[str(i)])
        self.cons_newx = temp
        return((mse,c5))
            
            
        
        

if __name__ == '__main__':

    m = smoothing_splines(data,newx=data.sample(n=10,random_state=1),cons_newx = False, plot = False, even_knots = True,
                          smooth_vec=np.array([0.01,0.01,0.01,0.2,0.2,0.2]),cv=False,penalty=1)
    m.evaluation()
    