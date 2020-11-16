# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 12:58:57 2020

@author: Tianyang
"""

import os
os.chdir('C:/Users/Tianyang/Documents/Option Project/source')
exec(open('initialization.py').read())

# -------------------------------------------------------------------------------------
# define class for local quadratic regression

class local_quad_reg():
    # class object for local quadratic parsi-regression. Quadratic on 'moneyness', linear on 'T2M'
    # Input: "data": [n x 3], DataFrame; "newx": [n x 2], DataFrame
    # Parameters: "H": bandwidth of kernel
    # Output: "prediction": [n x 1], DataFrame
    
    # function __init__: initiate core data (nc=['none','hard','soft'])
    def __init__(self,data,test,newx,
                 H=np.array(([1,0],[0,1])),
                 nc='none',penalty=1):
        self.trainx = np.array(data[['moneyness','busT2M']])
        self.trainy = np.array(data['totalvar']).reshape((-1,1))
        self.test = np.array(test)
        self.newx = np.array(newx)
        self.H = H
        self.nc = nc
        self.penalty = penalty
        warnings.filterwarnings("ignore")
        
    # function gaussian_kernel_d2: compute kernel density at one grid. 
    #   The input is the distance between two points.
    def guassian_kernel_d2(self,X):
        K = (2*np.pi)**(-1)*(np.linalg.det(self.H)**(-1/2))*np.exp(
                -1/2*(X.T.dot(np.linalg.inv(self.H)).dot(X)))
        return K
    
    # function gaussian_kernel_d2_v: evaluate all kernel values of trainx based on sub_newx. 
    #   Need update for each sub_newx iteration.
    def gaussian_kernel_d2_v(self):
        K = np.zeros((self.trainx.shape[0],1))
        for i in range(self.trainx.shape[0]):
            sub_X = self.trainx[i,:].T - self.sub_newx.T
            sub_K = self.guassian_kernel_d2(sub_X)
            K[i,0] = sub_K
        self.K = K/K.sum()
    
    # function feature: evaluate feature matrix of trainx based on sub_newx.
    #   Need update for each sub_newx iteration. F = [1e, Ki-K, Ti-T, (Ki-K)^2, (Ki-K)(Ti-T)], n x 5 matrix.
    def feature(self):
        F = np.zeros((self.trainx.shape[0],5))
        F[:,0] = 1
        F[:,1] = (self.trainx - self.sub_newx)[:,0]
        F[:,2] = (self.trainx - self.sub_newx)[:,1]
        F[:,3] = (self.trainx - self.sub_newx)[:,0]**2
        F[:,4] = (self.trainx - self.sub_newx)[:,0] * (
                self.trainx - self.sub_newx)[:,1]
        self.F = F
    
    # function w: compute total variance. The prediction model.
    def w(self,beta):
        return(beta[0])
    
    # function w_loss: compute loss function. The objective function of the optimization. 
    #   This is actually a weighted least square formulation.
    def w_loss(self,beta):
        return self.K.T.dot((self.trainy-self.F.dot(beta.reshape(-1,1)))**2)[0,0]
    
    # function w_grad: compute gradient
    def w_grad(self,beta):
        beta = np.array(beta)
        w_delta = np.zeros((5))
        for i in range(5):
            w_delta[i] = -2*self.K.T.dot((self.trainy - self.F.dot(beta.reshape((-1,1))))*self.F[:,i].reshape((-1,1)))[0,0]
        return(w_delta)
        
    # function w_hess: compute hessian
    def w_hess(self,beta):
        w_dd = np.zeros((5,5))
        for i in range(5):
            for j in range(5):
                w_dd[i,j] = 2*self.K.T.dot(self.F[:,i]*self.F[:,j])
        return(w_dd)
        
    # function nl_cons_f: LHS of Durrleman’s condition itself.
    def nl_cons_f(self,beta):
        return [((1 - self.sub_newx[0]*beta[1]/(2*self.w(beta)))**2 - beta[1]/4*
                (1/self.w(beta) + 1/4) + beta[4])]
        
    # function nl_cons_grad: gradient of LHS of Durrleman's condition.
    def nl_cons_grad(self,beta):
        beta = np.array(beta)
        grad = np.zeros((5))
        grad[0] = 2*(1 - self.sub_newx[0]*beta[1]/(2*beta[0]))*(self.sub_newx[0]*beta[1]/(2*beta[0]**2)) + (
                beta[1]/4/beta[0**2])
        grad[1] = 2*(1 - self.sub_newx[0]*beta[1]/(2*beta[0]))*(- self.sub_newx[0]*(2*beta[0])) - 1/4*(
                1/beta[0] + 1/4)
        grad[2] = 0
        grad[3] = 1
        grad[4] = 0
        return grad
    
    # function nl_cons_hess: hessian of LHS of Durrleman's condition.
    def nl_cons_hess(self,beta,v):
        beta = np.array(beta)
        hess = np.zeros((5,5))
        hess[0,0] = 2*(-self.sub_newx[0]*beta[1]/beta[0]**3 + 3/4*(self.sub_newx[0]**2)*(beta[1]**2)/(beta[0]**4)) - (
                1/2*beta[1]/beta[0]**3)
        hess[1,0] = 2*(self.sub_newx[0]/(2*beta[0]**2)-(self.sub_newx[0]**2)*beta[1]/(2*beta[0]**3)) + (
                1/4/beta[0]**2)
        hess[0,1] = hess[1,0]
        hess[1,1] = (self.sub_newx[0]/beta[0])**2/2
        hess = hess*v[0]
        return(hess)
    
    # function w_loss_soft: compute loss function with nonlinear constraint penalty. 
    def w_loss_soft(self,beta):
        c5 = -self.nl_cons_f(beta)[0]
        if c5<=0:
            c5=0
        return self.w_loss(beta) + self.penalty*c5
    
    # function w_grad_soft: compute gradient with nonlinear constraint penalty.
    def w_grad_soft(self,beta):
        if (self.nl_cons_f(beta)[0]<=0):
            return self.w_grad(beta) + self.penalty*np.array(-self.nl_cons_grad(beta)).reshape((5))
        else:
            return self.w_grad(beta)
    
    # function w_hess_soft: compute hessian matrix with nonlinear constraint penalty.
    def w_hess_soft(self,beta):
        if (self.nl_cons_f(beta)[0]<=0):
            nl_cons_hess_new = np.zeros((5,5))
        else:
            nl_cons_hess_raw = -self.penalty*self.nl_cons_hess(beta,[1])
            _index = np.where(self.nl_cons_grad(beta)==0)[0]
            nl_cons_hess_new = nl_cons_hess_raw
            nl_cons_hess_new[_index,:] = 0
            nl_cons_hess_new[:,_index] = 0
        return self.w_hess(beta) + nl_cons_hess_new
    
    # function fit: fit model for one sub_newx. Return the value as output
    def fit(self):
        beta = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        res = minimize(self.w_loss, beta, method='trust-constr', jac=self.w_grad, hess=self.w_hess,
               options={'verbose': 0})
        beta = res.x
        
        if ((self.nl_cons_f(beta)[0]>=0) & ((beta[0]<0) | (beta[2]<0))):
            bounds = [(0,None),(None,None),(0,None),(None,None),(None,None)]
            res = minimize(self.w_loss, beta, method='SLSQP', jac=self.w_grad, hess=self.w_hess,
                    bounds=bounds,
                   options={'verbose': 0,'maxiter':3000,'gtol':1e-07})
            beta = res.x
        if ((self.nl_cons_f(beta)[0]<0) & (self.nc=='hard')):
            bounds = [(0,None),(None,None),(0,None),(None,None),(None,None)]
            nonlinear_constraint = NonlinearConstraint(self.nl_cons_f, 0, np.inf, 
                                                       jac=self.nl_cons_grad, hess=self.nl_cons_hess)
            res = minimize(self.w_loss, beta, method='trust-constr', jac=self.w_grad, hess=self.w_hess,
                   constraints=nonlinear_constraint,
                    bounds=bounds,
                   options={'verbose': 0,'maxiter':3000,'gtol':1e-08})
            beta = res.x
        if ((self.nl_cons_f(beta)[0]<0) & (self.nc=='soft')):
            res = minimize(self.w_loss_soft, beta, method='Newton-CG', jac=self.w_grad_soft, hess=self.w_hess_soft)
            beta = res.x
            
        sub_pred = beta[0]
        self.beta = beta
        return(sub_pred)
        
    # function fit_all: fit model for all newx. Store prediction in self.
    def fit_all(self):
        self.prediction = np.zeros(self.newx.shape[0])
        self.c4 = np.zeros(self.newx.shape[0])
        self.c5 = np.zeros(self.newx.shape[0])
        for i in range(self.newx.shape[0]):
            self.sub_newx = self.newx[i,:]
            self.gaussian_kernel_d2_v()
            self.feature()
            sub_pred = self.fit()
            self.prediction[i] = sub_pred
            self.c4[i] = max(-self.beta[2],0)
            self.c5[i] = max(-self.nl_cons_f(self.beta)[0],0)
#            print('Fitting process completed for: ------ ' + str(i+1) +'|' + str(self.newx.shape[0]))
            
    # function evaluation: evaluate based on the test data
    def evaluation(self):
        self.fit_all()
        mse = np.average((self.prediction - self.test[:,2])**2)
        c4 = np.sum(self.c4)
        c5 = np.sum(self.c5)
        return((mse,c4,c5))
        
        
        
        
        
