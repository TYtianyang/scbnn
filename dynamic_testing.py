# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 14:47:04 2020

@author: Tianyang
"""

import os
os.chdir('C:/Users/Tianyang/Documents/Option Project/source')
exec(open('initialization.py').read())

from model_B_splines import B_splines
from model_local_quad_reg import local_quad_reg
from model_smoothing_splines import smoothing_splines
from model_net import net
from model_bnet_backprop import Gaussian, ScaleMixtureGaussian, BayesianLinear, BayesianNetwork

df = pd.read_csv('modified_df.csv',sep=',')
TimeInt = df['TimeInt'].unique()

# -------------------------------------------------------------------------------------
# single testing

sub_df = df[df['TimeInt']==TimeInt[3]][['moneyness','busT2M','totalvar']]
data = sub_df.groupby(['moneyness','busT2M']).mean().reset_index()

# local_quad_reg
start_time = time.time()
model = local_quad_reg(data,test=data.sample(100,random_state=3),newx=data.sample(100,random_state=3)[['moneyness','busT2M']],
                       nc='hard',penalty=10)
print(model.evaluation())
print(time.time()-start_time)

# smoothing_splines
start_time = time.time()
model = smoothing_splines(data,newx=data.sample(n=10,random_state=1),cons_newx = False, plot = False, even_knots = True,
                          smooth_vec=np.array([0.01,0.01,0.01,0.2,0.2,0.2]),cv=False,penalty=1)
print(model.evaluation())
print(time.time()-start_time)

# B_splines
start_time = time.time()
model = B_splines(data,newx=data.sample(n=200,random_state=1),cons_newx=False,penalty=1,auto_tic=True,tic=1,order=3,
                     plot=False)
print(model.evaluation())
print(time.time()-start_time)

# net
start_time = time.time()
model = net(data,penalty=1,test=data.sample(100,random_state=3))
model.load_state_dict(torch.load('pretrain'))
model.fit()
print(model.evaluation())
print(time.time()-start_time)

# bnet
start_time = time.time()
torch.manual_seed(4)
PI = 0.5
SIGMA_1 = torch.FloatTensor([math.exp(-0)])
SIGMA_2 = torch.FloatTensor([math.exp(-6)])
model = BayesianNetwork(train, test, bayes_step = 500, bayes_lr = 1e-4)
model.sample_mode = "mean"
model.bayes_fit()
model.sample_mode = "sample"
model.bayes_step = 1000
model.bayes_fit()
print(model.evaluation())
print(time.time()-start_time)

# -------------------------------------------------------------------------------------
# dynamic testing

# set up the train test split
seed = np.arange(len(TimeInt)) + 1
train_list = []
test_list = []
for i in range(len(TimeInt)):
    sub_df = df[df['TimeInt']==TimeInt[i]][['moneyness','busT2M','totalvar']]
    data = sub_df.groupby(['moneyness','busT2M']).mean().reset_index()
    np.random.seed(seed[i])
    msk = np.random.rand(len(data)) < 0.95
    train = data[msk]
    test = data[~msk]
    train_list.append(train)
    test_list.append(test)

# Case 1: smoothing splines
case1_mse = []
case1_c5 = []
case1_time = []
for i in range(len(TimeInt)):
    start_time = time.time()
    train = train_list[i]
    test = test_list[i]
    model = smoothing_splines(train,newx=test,cons_newx = False, plot = False, even_knots = True,
                              smooth_vec=np.array([0.01,0.01,0.01,0.2,0.2,0.2]),cv=False,penalty=0)
    mse, c5 = model.evaluation()
    time_ = time.time()-start_time
    case1_mse.append(mse)
    case1_c5.append(c5)
    case1_time.append(time_)

# Case 2: smoothing splines + soft constraints (penalty=1)
case2_mse = []
case2_c5 = []
case2_time = []
for i in range(len(TimeInt)):
    start_time = time.time()
    train = train_list[i]
    test = test_list[i]
    model = smoothing_splines(train,newx=test,cons_newx = False, plot = False, even_knots = True,
                              smooth_vec=np.array([0.01,0.01,0.01,0.2,0.2,0.2]),cv=False,penalty=1)
    mse, c5 = model.evaluation()
    time_ = time.time()-start_time
    case2_mse.append(mse)
    case2_c5.append(c5)
    case2_time.append(time_)
    
# Case 3: B-splines
case3_mse = []
case3_c5 = []
case3_time = []
for i in range(len(TimeInt)):
    start_time = time.time()
    train = train_list[i]
    test = test_list[i]
    model = B_splines(train,newx=test,cons_newx=False,penalty=0,auto_tic=True,tic=1,order=3,
                         plot=False)
    mse, c5 = model.evaluation()
    time_ = time.time()-start_time
    case3_mse.append(mse)
    case3_c5.append(c5)
    case3_time.append(time_)
    
# Case 4: B-splines + soft constraints (penalty=1)
case4_mse = []
case4_c5 = []
case4_time = []
for i in range(len(TimeInt)):
    start_time = time.time()
    train = train_list[i]
    test = test_list[i]
    model = B_splines(train,newx=test,cons_newx=False,penalty=1,auto_tic=True,tic=1,order=3,
                         plot=False)
    mse, c5 = model.evaluation()
    time_ = time.time()-start_time
    case4_mse.append(mse)
    case4_c5.append(c5)
    case4_time.append(time_)
    
# Case 5: local-quad-regression
case5_mse = []
case5_c4 = []
case5_c5 = []
case5_time = []
for i in range(len(TimeInt)):
    start_time = time.time()
    train = train_list[i]
    test = test_list[i]
    model = local_quad_reg(train,test=test,newx=test[['moneyness','busT2M']],
                           nc='none',penalty=1)
    mse,c4,c5 = model.evaluation()
    time_ = time.time()-start_time
    case5_mse.append(mse)
    case5_c4.append(c4)
    case5_c5.append(c5)
    case5_time.append(time_)
    
# Case 6: local-quad-regression + hard constraints
case6_mse = []
case6_c4 = []
case6_c5 = []
case6_time = []
for i in range(len(TimeInt)):
    start_time = time.time()
    train = train_list[i]
    test = test_list[i]
    model = local_quad_reg(train,test=test,newx=test[['moneyness','busT2M']],
                           nc='hard',penalty=1)
    mse,c4,c5 = model.evaluation()
    time_ = time.time()-start_time
    case6_mse.append(mse)
    case6_c4.append(c4)
    case6_c5.append(c5)
    case6_time.append(time_)
    
# Case 7: local-quad-regression + soft constraints (penalty=1)
case7_mse = []
case7_c4 = []
case7_c5 = []
case7_time = []
for i in range(len(TimeInt)):
    start_time = time.time()
    train = train_list[i]
    test = test_list[i]
    model = local_quad_reg(train,test=test,newx=test[['moneyness','busT2M']],
                           nc='soft',penalty=1)
    mse,c4,c5 = model.evaluation()
    time_ = time.time()-start_time
    case7_mse.append(mse)
    case7_c4.append(c4)
    case7_c5.append(c5)
    case7_time.append(time_)
    
# Case 8: net
case8_mse = []
case8_c4 = []
case8_c5 = []
case8_time = []
for i in range(len(TimeInt)):
    start_time = time.time()
    train = train_list[i]
    test = test_list[i]
    model = net(train,penalty=1,test=test)
    model.load_state_dict(torch.load('pretrain'))
    model.fit_0()
    mse,c4,c5 = model.evaluation()
    time_ = time.time()-start_time
    case8_mse.append(mse)
    case8_c4.append(c4)
    case8_c5.append(c5)
    case8_time.append(time_)
    
# Case 9: net + soft constraints (penalty=1)
case9_mse = []
case9_c4 = []
case9_c5 = []
case9_time = []
for i in range(len(TimeInt)):
    start_time = time.time()
    train = train_list[i]
    test = test_list[i]
    model = net(train,penalty=1,test=test)
    model.load_state_dict(torch.load('pretrain'))
    model.fit()
    mse,c4,c5 = model.evaluation()
    time_ = time.time()-start_time
    case9_mse.append(mse)
    case9_c4.append(c4)
    case9_c5.append(c5)
    case9_time.append(time_)
    
# Case 10: online net
case10_mse = []
case10_c4 = []
case10_c5 = []
case10_time = []
for i in range(len(TimeInt)):
    if i==0:
        start_time = time.time()
        train = train_list[i]
        test = test_list[i]
        model = net(train,penalty=1,test=test)
        model.load_state_dict(torch.load('pretrain'))
        model.fit()
        mse,c4,c5 = model.evaluation()
        time_ = time.time()-start_time
        case10_mse.append(mse)
        case10_c4.append(c4)
        case10_c5.append(c5)
        case10_time.append(time_)
        last_model = model.hidden
    else:
        start_time = time.time()
        train = train_list[i]
        test = test_list[i]
        model = net(train,penalty=1,test=test,constr_step=100)
        model.hidden = last_model
        model.fit()
        mse,c4,c5 = model.evaluation()
        time_ = time.time()-start_time
        case10_mse.append(mse)
        case10_c4.append(c4)
        case10_c5.append(c5)
        case10_time.append(time_)
        last_model = model.hidden

# Case 11: Bayes net
case11_mse = []
case11_c4 = []
case11_c5 = []
case11_time = []
for i in range(len(TimeInt)):
    start_time = time.time()
    train = train_list[i]
    test = test_list[i]
    torch.manual_seed(4)
    model = BayesianNetwork(train, test, bayes_step = 500, bayes_lr = 1e-4)
    model.sample_mode = "mean"
    model.bayes_fit()
    model.sample_mode = "sample"
    model.bayes_step = 1000
    model.bayes_fit()
    mse,c4,c5, expected_theta_len = model.evaluation()
    time_ = time.time()-start_time
    case11_mse.append(mse)
    case11_c4.append(c4)
    case11_c5.append(c5)
    case11_time.append(time_)

# -------------------------------------------------------------------------------------
# others

#zdata = np.asarray(sub_df['totalvar'])
#xdata = np.asarray(sub_df['moneyness'])
#ydata = np.asarray(sub_df['busT2M'])
#xnew = np.linspace(np.min(xdata),np.max(xdata),num=10)
#ynew = np.linspace(np.min(ydata),np.max(ydata),num=10)
#xnew, ynew = np.meshgrid(xnew,ynew)
#
#panel_new = np.concatenate((xnew.reshape((-1,1)),ynew.reshape((-1,1))),axis=1)
#m = local_quad_reg(sub_df,panel_new,H=np.array([[0.05,0],[0,0.2]]),nc='soft') # [[0.05,1],[1,0.2]]
#m.fit_all()
#znew = m.prediction.reshape((-1,1))
#panel_new = np.hstack((panel_new,znew))
#xnew = panel_new[:,0].reshape((10,10))
#ynew = panel_new[:,1].reshape((10,10))
#znew = panel_new[:,2].reshape((10,10))
#
#fig = plt.figure()
#ax = plt.axes(projection='3d')
#ax.scatter3D(xdata, ydata, zdata, c='red',marker='.')
#ax.set_xlabel('Moneyness')
#ax.set_ylabel('T2M')
#ax.set_zlabel('Total Variance')
#ax.locator_params(axis='y',nbins=5)
#ax.locator_params(axis='x',nbins=5)
#ax.view_init(elev = None, azim=20)
#plt.show()
#
#fig = plt.figure()
#bx = plt.axes(projection='3d')
#bx.scatter3D(xnew, ynew, znew, c='blue',marker='.')
#bx.set_xlabel('Moneyness')
#bx.set_ylabel('T2M')
#bx.set_zlabel('Total Variance')
#bx.locator_params(axis='y',nbins=5)
#bx.locator_params(axis='x',nbins=5)
#bx.view_init(elev = None, azim=20)
#bx.set_zlim(ax.get_zlim())
#bx.set_ylim(ax.get_ylim())
#bx.set_xlim(ax.get_xlim())
#plt.show()
#
#fig = plt.figure()
#bx = plt.axes(projection='3d')
#bx.contour3D(xnew, ynew, znew, 50, cmap='binary')
#bx.set_xlabel('Moneyness')
#bx.set_ylabel('T2M')
#bx.set_zlabel('Total Variance')
#bx.locator_params(axis='y',nbins=5)
#bx.locator_params(axis='x',nbins=5)
#bx.view_init(elev = None, azim=20)
#bx.set_zlim(ax.get_zlim())
#bx.set_ylim(ax.get_ylim())
#bx.set_xlim(ax.get_xlim())
#plt.show()

for ind in TimeInt:

    sub_df = df[df['TimeInt']==ind][['moneyness','busT2M','totalvar']]
    
    zdata = np.asarray(sub_df['totalvar'])
    xdata = np.asarray(sub_df['moneyness'])
    ydata = np.asarray(sub_df['busT2M'])
    xnew = np.linspace(np.min(xdata),np.max(xdata),num=100)
    ynew = np.linspace(np.min(ydata),np.max(ydata),num=100)
    xnew, ynew = np.meshgrid(xnew,ynew)
    
#    model = interpolate.SmoothBivariateSpline(xdata,ydata,zdata,kx=3,ky=3)
#    znew = model.ev(xnew, ynew)
#    print(np.sqrt(model.get_residual()/sub_df.shape[0]))
#    znew = interpolate.griddata(np.concatenate((xdata.reshape(len(xdata),1),ydata.reshape(len(ydata),1)),axis=1), 
#                                zdata, (xnew, ynew), method='nearest')
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(xdata, ydata, zdata, c='red',marker='.')
    ax.set_xlabel('Moneyness')
    ax.set_ylabel('T2M')
    ax.set_zlabel('Total Variance')
    ax.locator_params(axis='y',nbins=5)
    ax.locator_params(axis='x',nbins=5)
    ax.view_init(elev = None, azim=20)
    plt.show()
    
#    fig = plt.figure()
#    bx = plt.axes(projection='3d')
#    bx.contour3D(xnew, ynew, znew, 50, cmap='binary')
#    bx.set_xlabel('Moneyness')
#    bx.set_ylabel('T2M')
#    bx.set_zlabel('Total Variance')
#    bx.locator_params(axis='y',nbins=5)
#    bx.locator_params(axis='x',nbins=5)
#    bx.view_init(elev = None, azim=20)
#    bx.set_zlim(ax.get_zlim())
#    bx.set_ylim(ax.get_ylim())
#    bx.set_xlim(ax.get_xlim())
#    plt.show()

    


