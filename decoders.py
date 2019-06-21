from scipy.stats import spearmanr
import scipy.sparse.linalg
from scipy import io
import numpy as np
from matplotlib import pyplot as plt
import os
from scipy.stats import zscore
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from sklearn.manifold import isomap
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso, Lars
from sklearn.linear_model import orthogonal_mp
from scipy.stats import linregress
from multiprocessing import Pool

import utils

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    
    class TwoLayer(nn.Module):
        def __init__(self, D_in, H, D_out):
            super(TwoLayer, self).__init__()
            self.sigmoid = nn.Sigmoid()
            self.lil = nn.Sequential(
                nn.Linear(D_in, H),
                nn.ReLU(inplace=True),
                nn.Linear(H, H),
                nn.ReLU(inplace=True),
                nn.Linear(H, D_out),
            )
        def forward(self, x):
            xf = self.lil(x)
            xf = self.sigmoid(xf)
            return xf
except:
    print('torch not installed, cannot run neural-network decoder')

def fast_ridge(X, y, lam=1):
    N, M = X.shape
    lam = lam * M
    if N<M:
        XXt = X @ X.T
        w = np.linalg.solve(XXt + lam * np.eye(N), X @ y)
    else:
        XtX = X.T @ X
        w = 1/lam * X @ (y - np.linalg.solve(lam * np.eye(M) + XtX, XtX @ y))
    #w = np.squeeze(w)
    return w

def independent_decoder(sresp, istim, itrain, itest, nbase = 10):
    nangle =  2*np.pi
    A, B, D, rez = fit_indep_model(sresp[:, itrain], istim[itrain], nbase)
    apred, logL, B2, Kup = test_indep_model(sresp[:, itest], A, nbase)
    
    # single neuron tuning curves
    ypred = A.T @ B
    # SNR
    SNR = np.var(ypred, axis=1) / np.var(rez, axis=1)
    
    # preferred stimulus for each neuron
    theta_pref = istim[itrain][np.argmax(ypred, axis=1)]
    
    error = istim[itest] - apred
    error = np.remainder(error,nangle)
    error[error > nangle/2] = error[error > nangle/2] - nangle
    
    return apred, error, ypred, logL, SNR, theta_pref

def test_indep_model(X, A, nbase, xcoef = None):
    # use GPU for optimization
    nodes = 32
    theta = np.linspace(0, 2*np.pi, nodes+1)[:-1]

    bubu = np.arange(0,nbase)[:, np.newaxis]
    F1 = np.cos(theta * bubu)
    F2 = np.sin(theta * bubu)
    B = np.concatenate((F2,F1), axis=0)
    D = np.concatenate((F1*bubu, -F2*bubu), axis=0)
    B = B[1:, :]
    D = D[1:, :]

    logL = np.zeros((X.shape[1], nodes))
    for k in range(nodes):
        rez = X - (A.T @ B[:, k])[:, np.newaxis]
        logL[:, k] = -np.mean(rez**2, axis=0)
  
    Kup = utils.upsampling_mat(nodes, int(3200/nodes), nodes/32)
    yup = logL @ Kup.T
    apred = np.argmax(yup, axis=1) / yup.shape[1] * 2 * np.pi

    return apred, logL, B, Kup

def fit_indep_model(X, istim, nbase):
    theta = istim.astype(np.float32)
    bubu = np.arange(0,nbase)[:, np.newaxis]
    F1 = np.cos(theta * bubu)
    F2 = np.sin(theta * bubu)
    B = np.concatenate((F2,F1), axis=0)
    D = np.concatenate((F1*bubu, -F2*bubu), axis=0)
    B = B[1:, :]
    D = D[1:, :]
    A = np.linalg.solve(B @ B.T, B @ X.T)
    rez = X - A.T @ B
    vexp = 1 - np.mean(rez**2, axis=1)

    return A, B, D, rez

def rez_proj_diff(dTheta, rez, niter = 1):
    NN = rez.shape[0]
    xcoef = np.ones(NN)
    for j in range(niter):
        X0 = dTheta * xcoef[:, np.newaxis]
        ddT = np.sum(X0 * rez, axis=0)/np.sum(X0**2, axis=0)
        X1 = ddT * dTheta
        xcoef = np.sum(X1 * rez,axis = 1)/np.sum(X1**2, axis=1)
        xcoef = xcoef/np.median(xcoef)
    return xcoef, ddT

def orthmp_decoder(inputs):
    alpha, X, y = inputs
    w = orthogonal_mp(X.T, y, n_nonzero_coefs=int(np.ceil(alpha * X.shape[0])))
    return w

def lars_decoder(inputs):
    alpha, X, y = inputs
    model = Lars(fit_intercept=False,n_nonzero_coefs=int(np.ceil(alpha * X.shape[0]))).fit(X.T, y)
    w = model.coef_
    return w

def lasso_decoder(inputs):
    alpha, X, y = inputs
    model = Lasso(alpha = alpha).fit(X.T, y)
    w = model.coef_
    return w

def best_theta(sresp, istim, sigma = 0.05):
    theta_pref = np.linspace(0,2*np.pi,101)[:-1]
    theta0     = istim[:,np.newaxis] - theta_pref[np.newaxis,:]
    y          = np.exp((np.cos(theta0)-1) / sigma)
    y  -= y.mean()

    cc     = sresp @ y / (y**2).sum(axis=0)
    ith    = np.argmax(cc, axis=1)
    theta0 = theta_pref[ith]
    A0     = np.max(cc, axis=1)

    device      = torch.device("cuda")
    xtrain_gpu  = torch.from_numpy(sresp.astype(np.float32)).to(device)
    theta_gpu   = torch.from_numpy(istim.astype(np.float32)).to(device)
    A_gpu       = torch.from_numpy(A0[:,np.newaxis].astype(np.float32)).to(device)
    theta0_gpu  = torch.from_numpy(theta0[:,np.newaxis].astype(np.float32)).to(device)

    theta0_gpu.requires_grad=True
    learning_rate = 1000
    for it in range(200):
        F = A_gpu * (torch.exp((torch.cos(( theta_gpu - theta0_gpu))-1) / 0.05 ) - 0.089780316)
        loss = ((F - xtrain_gpu) ** 2).mean()
        loss.backward()
        with torch.no_grad():
            theta0_gpu -= learning_rate * theta0_gpu.grad
            theta0_gpu.grad.zero_()
        if it%100==0:
            print(loss)
    t0 = theta0_gpu.cpu().detach().numpy().flatten()
    return t0, A0

def get_pops(t0, A0, sresp, nth = 30):
    theta_bins = np.linspace(0,2*np.pi,nth+1)
    t0         = t0%(2*np.pi)
    pops       = np.zeros((nth, sresp.shape[1]))
    for j in range(nth):
        ix = np.logical_and(t0>=theta_bins[j], t0<theta_bins[j+1])
        pops[j, :] = A0[ix].T @ sresp[ix, :]
        #pops[j, :] = np.mean(sresp[ix, :], axis=0)
    return pops

def vonmises_decoder(sresp, istim, itrain, itest, nangle=2*np.pi, lam=1, dcdtype='L2'):
    ''' stim ids istim, neural responses sresp (NNxnstim)'''
    ''' nangle = np.pi if orientations'''
    nth = 48
    sigma = 0.05 * 2

    theta_pref = np.linspace(0,2*np.pi,nth+1)[:-1]
    theta0 = 2 * np.pi / nangle * istim[itrain,np.newaxis] - theta_pref[np.newaxis,:]

    # von mises
    y = np.exp((np.cos(theta0)-1) / sigma)

    # add little bump in opposite direction
    #if nangle > np.pi:
    #    theta1 = 2 * np.pi / nangle * istim[itrain,np.newaxis] - (theta_pref[np.newaxis,:] - np.pi)%(2*np.pi)
    #    y += 0.25 * np.exp((np.cos(theta1)-1) / sigma)

    y = zscore(y, axis=1)

    NN = sresp.shape[0]
    ntot = y.shape[0]

    X = sresp[:,itrain]
    if dcdtype=='L2':
        XtX = X @ X.T
        A = fast_ridge(X, y, lam=lam)
        #A = np.linalg.solve(XtX + X.shape[1] * lam*np.eye(NN), X @ y)
    else:
        dsplit = []
        for i in range(nth):
            dsplit.append([lam, X, y[:,i]])
        if dcdtype=='L1':
            with Pool(10) as p:
                results = p.map(lasso_decoder, dsplit)
        elif dcdtype=='OMP':
            with Pool(10) as p:
                results = p.map(orthmp_decoder, dsplit)
        elif dcdtype=='Lars':
            with Pool(10) as p:
                results = p.map(lars_decoder, dsplit)
        A = np.zeros((NN, nth))
        for i in range(len(results)):
            A[:,i] = results[i]

    ypred = sresp[:,itest].T @ A
    
    # circular interpolation of ypred
    Kup = utils.upsampling_mat(y.shape[1])
    yup = ypred @ Kup.T

    apred = np.argmax(yup, axis=1) / yup.shape[1] * nangle
    error = istim[itest] - apred
    error = np.remainder(error, nangle)
    error[error > nangle/2] = error[error > nangle/2] - nangle

    return apred, error, ypred, A

def derivative_decoder(istim, sresp, itrain, itest, lam=1, theta_pref = np.linspace(0,2*np.pi,33)[:-1], dcdtype='regression'):
    ''' stim ids istim, neural responses sresp (NNxnstim)'''
    ''' nangle = np.pi if orientations'''

    theta_pref = np.array(theta_pref)
    nth = len(theta_pref)
    sigma = 0.05 * 2
    dt = np.pi/32

    theta0 = istim[itrain,np.newaxis] - theta_pref[np.newaxis,:]

    y = np.exp(np.cos(theta0- dt) / sigma) - np.exp(np.cos(theta0 + dt) / sigma)
    NN = sresp.shape[0]
    ntot = y.shape[0]
    X = sresp[:,itrain]

    y = zscore(y, axis=0) # changed this from axis=0
    if dcdtype is 'regression':
        A = fast_ridge(X, y, lam = lam)
    else:
        A = X @ y
    ypred = A.T @ sresp[:, itest]

    D = np.zeros((0,))
    dy = np.zeros((0,))
    for j in range(nth):
        ds = (istim[itest] - theta_pref[j])%(2*np.pi)
        ds[ds>np.pi] = ds[ds>np.pi] - 2*np.pi
        D = np.concatenate((D, ds), axis=0)
        dy = np.concatenate((dy, ypred[j,:]), axis=0)

    return D, dy, A

def rf_discriminator(xtrain, ytrain, xtest, ytest):
    clf = RandomForestClassifier(n_estimators=1000,max_depth=None)
    clf.fit(xtrain, ytrain)
    ypred = clf.predict(xtest)

    print('accuracy: %2.2f'%(1 - np.abs((ypred>0.5).astype(int) - ytest.astype(int)).mean()))
    
    # convert to -1 / +1
    ychoice = ((ypred>0.5) - 0.5) * 2
    return ychoice
    
def nn_discriminator(xtrain, ytrain, xtest, ytest):
    D_in  = xtrain.shape[1]
    H = 100
    D_out = 1
    model = TwoLayer(D_in, H, D_out)

    device = torch.device("cuda")
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    loss_function = nn.BCELoss()

    xtrain_gpu  = torch.from_numpy(xtrain.astype(np.float32)).to(device)
    ytrain_gpu = torch.from_numpy(ytrain.astype(np.float32)).to(device)    

    for it in range(int(5e4)):
        output  = model(xtrain_gpu)
        loss    = loss_function(output[:,0], ytrain_gpu)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #if it%10000==0:
        #    print(it, loss.mean().item())
            
    ypred = model(torch.from_numpy(xtest.astype(np.float32)).to(device)).cpu().detach().numpy()
    print('accuracy: %2.2f'%(1 - np.abs((ypred[:,0]>0.5).astype(int) - ytest.astype(int)).mean()))

    # convert to -1 / +1
    ychoice = ((ypred[:,0]>0.5) - 0.5) * 2
    return ychoice

def dense_discrimination(fs):
    ''' discriminate between +/- 2 degrees trials and as a function of # of neurons and stims '''
    nskipstim = 2**np.linspace(0, 10, 21)
    nstim     = np.zeros((len(nskipstim), len(fs)), 'int')
    nskip     = 2**np.linspace(0, 10, 21)
    npop      = np.zeros((len(nskip), len(fs)), 'int')

    nth = 1
    lam = 1
    theta_pref = np.array([np.pi/4])
    dd = 1/10
    drange2 = np.arange(-2, 2.01, dd*2)    
    P = np.zeros((len(nskipstim), len(drange2), len(fs)), np.float32)
    P2 = np.zeros((len(nskip), len(drange2), len(fs)), np.float32)

    for t,f in enumerate(fs):
        print(os.path.basename(f))
        dat = np.load(f, allow_pickle=True).item()

        sresp, istim, itrain, itest = utils.compile_resp(dat)    
        theta0 = istim[itrain,np.newaxis] - theta_pref
        y = theta0
        NN = sresp.shape[0]
        ntot = y.shape[0]
        X = sresp[:, itrain]
        Xtest = sresp[:, itest]

        nstim[:,t] = (itrain.size/nskipstim).astype('int')   
        y = zscore(y, axis=0) # changed this from axis=0
        np.random.seed(seed = 101)
        rperm2 = np.random.permutation(itrain.size)    
        for m in range(len(nskipstim)): 
            iSS = rperm2[:nstim[m,t]]
            A = fast_ridge(X[:, iSS], y[iSS], lam = 1)
            ypred = (A.T @ Xtest).flatten()
            D = np.zeros((0,))
            dy = np.zeros((0,))
            ds = (istim[itest] - theta_pref[0])%(2*np.pi)
            ds[ds>np.pi] = ds[ds>np.pi] - 2*np.pi
            D = np.concatenate((D, ds), axis=0)
            dy = np.concatenate((dy, ypred), axis=0)
            for j,deg in enumerate(drange2):
                ix = np.logical_and(D>np.pi/180 * (deg-dd), D<np.pi/180 * (deg+dd))
                P[m, j, t] = np.mean(dy[ix]>0)

        np.random.seed(seed = 101)
        npop[:, t] = (NN/nskip).astype('int')        
        rperm = np.random.permutation(NN)    
        for k in range(len(nskip)):
            iNN = rperm[:npop[k,t]]        
            A = fast_ridge(X[iNN,:], y, lam=1)
            ypred = (A.T @ Xtest[iNN,:]).flatten()
            D = np.zeros((0,))
            dy = np.zeros((0,))

            ds = (istim[itest] - theta_pref[0])%(2*np.pi)
            ds[ds>np.pi] = ds[ds>np.pi] - 2*np.pi
            D = np.concatenate((D, ds), axis=0)
            dy = np.concatenate((dy, ypred), axis=0)

            for j,deg in enumerate(drange2):
                ix = np.logical_and(D>np.pi/180 * (deg-dd), D<np.pi/180 * (deg+dd))
                P2[k, j, t] = np.mean(dy[ix]>0)
                
    return npop, nstim, P, P2, drange2

def run_discrimination(fs, decoder='linear'):
    drange = np.arange(-29,30)
    P = np.zeros((len(fs),len(drange)))
    d75 = np.zeros((len(fs),))
    ithres = np.pi/4
    for t,f in enumerate(fs):
        print(os.path.basename(f))
        dat = np.load(f, allow_pickle=True).item()
        sresp, istim, itrain, itest = utils.compile_resp(dat)
        
        if decoder=='linear':
            D, dy, A = derivative_decoder(istim, sresp, itrain, itest)
            for j,deg in enumerate(drange):
                ix = np.logical_and(D>np.pi/180 * (deg-.5), D<np.pi/180 * (deg+.5))
                P[t,j] = np.nanmean(dy[ix]>0)
        else:
            if decoder=='deep_net':
                ## compute PC's
                pca = PCA(n_components=256).fit(sresp)
                x = pca.components_.T
                sv = pca.singular_values_
                x *= sv
            else:
                x = sresp.T.copy()
                del sresp
            imin = ithres - np.pi/6
            imax = ithres + np.pi/6
            gstims = np.logical_and(istim>=imin, istim<=imax).nonzero()[0]
            xtrain = x[gstims[::2], :]
            xtest = x[gstims[1::2], :]
            ytrain = (istim[gstims[::2]] - ithres) > 0
            ytest = (istim[gstims[1::2]] - ithres) > 0
            atest = istim[gstims[1::2]] - ithres

            Pk = np.zeros((len(drange),5))
            for k in range(5):
                if decoder=='random_forest':
                    ychoice = rf_discriminator(xtrain, ytrain, xtest, ytest)
                elif decoder=='deep_net':
                    ychoice = nn_discriminator(xtrain, ytrain, xtest, ytest)

                P0 = np.zeros(drange.shape)
                for j,deg in enumerate(drange):
                    ix = np.logical_and(atest>np.pi/180 * (deg-.5), atest<np.pi/180 * (deg+.5))
                    P0[j] = np.mean(ychoice[ix]>0)
                P0 = (P0 + 1 - P0[::-1]) / 2 
                d750 = utils.discrimination_threshold(P0, drange)[0]
                print('discrimination threshold %2.2f'%d750)
                Pk[:,k] = P0
            P[t] = Pk.mean(axis=-1)
        d75[t] = utils.discrimination_threshold(P[t], drange)[0]
        print('--- discrimination threshold %2.2f'%d75[t])
    return P, d75, drange

def run_decoder(fs, linear=True, npc=0):
    E = np.zeros((len(fs),))
    errors = []
    stims = []
    snrs = []
    theta_prefs = []
    for t,f in enumerate(fs):
        dat = np.load(f, allow_pickle=True).item()

        sresp, istim, itrain, itest = utils.compile_resp(dat, npc=npc)
        SNR = []
        theta_pref = []
        if linear:
            apred, error, _, _ = vonmises_decoder(sresp, istim, itrain, itest)
        else:
            apred, error, _, _, SNR, theta_pref = independent_decoder(sresp, istim, itrain, itest)

        # save error and stimulus
        errors.append(error)
        stims.append(istim[itest])
        snrs.append(SNR)
        theta_prefs.append(theta_pref)
        E[t] = np.median(np.abs(error)) * 180/np.pi
        print(os.path.basename(f), E[t])
        
    return E, errors, stims, snrs, theta_prefs

def runspeed_discrimination(fs, all_running):
    ntesthalf = 1000
    drange = np.arange(-29, 30, 1)
    P0 = np.zeros((len(fs), len(drange), 2), np.float32)
    d75 = np.zeros((len(fs), 2), np.float32)
    for t,f in enumerate(fs):
        print(os.path.basename(f))
        dat = np.load(f, allow_pickle=True).item()

        sresp, istim, itrain, itest = utils.compile_resp(dat)

        rperm = np.random.permutation(istim.size)
        runsp = all_running[t]
        itest1 = (runsp[rperm]<2).nonzero()[0]
        itest1 = rperm[itest1[:ntesthalf]]
        itest2 = (runsp[rperm]>10).nonzero()[0]
        itest2 = rperm[itest2[:ntesthalf]]
        itest = np.concatenate((itest1, itest2), axis=0)
        itrain = np.ones(istim.size, 'Bool')
        itrain[itest] = False

        D, dy, A = derivative_decoder(istim, sresp, itrain[::1], itest1, lam = 1)    
        for j,deg in enumerate(drange):
            ix = np.logical_and(D>np.pi/180 * (deg-.5), D<np.pi/180 * (deg+.5))
            P0[t,j,0] = np.mean(dy[ix]>0)

        D, dy, A = derivative_decoder(istim, sresp, itrain[::1], itest2, lam = 1)    
        for j,deg in enumerate(drange):
            ix = np.logical_and(D>np.pi/180 * (deg-.5), D<np.pi/180 * (deg+.5))
            P0[t,j,1] = np.mean(dy[ix]>0)
        d75[t,0] = utils.discrimination_threshold(P0[t,:,0], drange)[0]
        d75[t,1] = utils.discrimination_threshold(P0[t,:,1], drange)[0]
        print('--- discrimination threshold passive %2.2f, running %2.2f'%(d75[t,0], d75[t,1]))
        
    return P0, d75, drange


def layer_discrimination(fs, all_depths):
    drange = np.arange(-29, 30, 1)
    P0 = np.zeros((len(fs), len(drange), 2), np.float32)
    d75 = np.zeros((len(fs), 2), np.float32)
    nangle=2*np.pi
    for t,f in enumerate(fs):
        print(os.path.basename(f))
        dat = np.load(f, allow_pickle=True).item()
        depths = all_depths[t]

        sresp, istim, itrain, itest = utils.compile_resp(dat)
        nstim = sresp.shape[1]

        upper = depths < depths.min() + 100
        lower = depths > depths.max() - 100

        D, dy0, A = derivative_decoder(istim, sresp[upper], itrain, itest, lam = 1)
        D, dy1, A = derivative_decoder(istim, sresp[lower], itrain, itest, lam = 1)
        for j,deg in enumerate(drange):
            ix = np.logical_and(D>np.pi/180 * (deg-.5), D<np.pi/180 * (deg+.5))
            P0[t,j,0] = np.mean(dy0[ix]>0)
            P0[t,j,1] = np.mean(dy1[ix]>0)

        d75[t,0] = utils.discrimination_threshold(P0[t,:,0], drange)[0]
        d75[t,1] = utils.discrimination_threshold(P0[t,:,1], drange)[0]
        print('--- discrimination threshold L2/3 %2.2f, L4 %2.2f'%(d75[t,0], d75[t,1]))
        
    return P0, d75, drange

def chron_discrimination(fs, all_depths):
    drange = np.arange(-29, 30, 1)
    P0 = np.zeros((len(fs), len(drange), 2), np.float32)
    d75 = np.zeros((len(fs), 2), np.float32)
    nangle=2*np.pi
    for t,f in enumerate(fs):
        print(os.path.basename(f))
        dat = np.load(f, allow_pickle=True).item()
        
        sresp, istim, itrain, itest = utils.compile_resp(dat)
        nstim = sresp.shape[1]

        D0, dy0, A = derivative_decoder(istim, sresp, itrain, itest, lam = 1)

        # use 75% vs 25%
        itrain = np.zeros((nstim,), np.bool)
        itest = np.zeros((nstim,), np.bool)
        itrain[:int(nstim*.75)] = True
        itest[int(nstim*.75):] = True
        D1, dy1, A = derivative_decoder(istim, sresp, itrain, itest, lam = 1)

        for j,deg in enumerate(drange):
            ix = np.logical_and(D0>np.pi/180 * (deg-.5), D0<np.pi/180 * (deg+.5))
            P0[t,j,0] = np.mean(dy0[ix]>0)

            ix = np.logical_and(D1>np.pi/180 * (deg-.5), D1<np.pi/180 * (deg+.5))
            P0[t,j,1] = np.mean(dy1[ix]>0)
        d75[t,0] = utils.discrimination_threshold(P0[t,:,0], drange)[0]
        d75[t,1] = utils.discrimination_threshold(P0[t,:,1], drange)[0]
        print('--- discrimination threshold original %2.2f, chronological %2.2f'%(d75[t,0], d75[t,1]))
        
    return P0, d75, drange

def asymptotics(fs, linear=True):
    nskip = 2**np.linspace(0, 10, 21)
    nskipstim = 2**np.linspace(0, 10, 21)
    E = np.zeros((len(nskip),2, len(fs)))
    E2 = np.zeros((len(nskipstim), len(fs)))

    ccE = np.zeros((len(nskip), 2, len(fs)))
    nsplit = np.zeros((len(nskip), len(fs)), 'int')
    npop = np.zeros((len(nskip), len(fs)), 'int')
    nstim = np.zeros((len(nskipstim), len(fs)), 'int')

    for t,f in enumerate(fs):
        print('asymp for: ', os.path.basename(f))
        dat = np.load(f, allow_pickle=True).item()

        sresp, istim, itrain, itest = utils.compile_resp(dat)
        ypos = np.array([dat['stat'][j]['med'][0] for j in range(len(dat['stat']))])

        # split neurons for decoder into strips (no Z overlap between two sets)
        nstrips = 8

        NN = sresp.shape[0]
        npop[:, t] = (NN/nskip).astype('int')

        np.random.seed(seed = 101)
        rperm = np.random.permutation(NN)    
        for k in range(len(nskip)):
            iNN = rperm[:npop[k,t]]
            if linear:
                error = vonmises_decoder(sresp[iNN], istim, itrain, itest)[1]
            else:
                error = independent_decoder(sresp[iNN, :], istim, itrain, itest)[1]
                
            E[k,0,t] = np.median(np.abs(error)) * 180/np.pi

            n1, n2 = utils.stripe_split(ypos[iNN], nstrips)
            if linear:
                err1 = vonmises_decoder(sresp[iNN[n1]], istim, itrain, itest)[1]
                err2 = vonmises_decoder(sresp[iNN[n2]], istim, itrain, itest)[1]
            else:
                err1 = independent_decoder(sresp[iNN[n1]], istim, itrain, itest)[1]
                err2 = independent_decoder(sresp[iNN[n2]], istim, itrain, itest)[1]
                
            E[k,1,t] = np.abs(np.median(err1*err2))**.5 * 180/np.pi

            ccE[k,0,t] = np.corrcoef(err1, err2)[0,1]
            ccE[k,1,t] = spearmanr(err1, err2)[0]
            nsplit[k,t] = len(n1)

        nstim[:,t] = (itrain.size/nskipstim).astype('int')    
        np.random.seed(seed = 101)
        rperm = np.random.permutation(itrain.size)    
        for k in range(len(nskipstim)):    
            iSS = rperm[:nstim[k,t]]
            if linear:
                error = vonmises_decoder(sresp, istim, itrain[iSS], itest)[1]
            else:
                error = independent_decoder(sresp, istim, itrain[iSS], itest)[1]
            E2[k, t] = np.median(np.abs(error)) * 180/np.pi
    
    return E, ccE, nsplit, npop, nstim, E2

def pc_decoding(fs, nPC):
    ''' linearly decode from PCs of data '''

    errors = np.zeros((len(fs), len(nPC)))
    for t,f in enumerate(fs):
        print(os.path.basename(f))
        dat = np.load(f, allow_pickle=True).item()

        sresp, istim, itrain, itest = utils.compile_resp(dat)    

        pca = PCA(n_components=nPC[-1]).fit(sresp)
        u = pca.components_.T
        sv = pca.singular_values_
        u *= sv

        for k,pc in enumerate(nPC):
            apred, error, _, _ = vonmises_decoder(u[:,:pc].T, istim, itrain, itest)
            errors[t,k] = np.median(np.abs(error)) * 180 / np.pi
            if t==0:
                if k==0:
                    apreds = np.zeros((len(nPC), len(itest)))
                    atrues = np.zeros((len(nPC), len(itest)))
                apreds[k] = apred * 180 / np.pi
                atrues[k] = istim[itest] * 180 / np.pi
    return errors, apreds, atrues
