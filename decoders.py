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


def log_prob(a, x, y, rcos, rsin):
    atx = a[0].T @ x
    btx = a[1].T @ x

    Z = atx * rcos[:, np.newaxis] + btx * rsin[:, np.newaxis]
    Zmax = np.max(Z, axis=0)

    Z = np.exp(Z-Zmax)
    Zsum = np.sum(Z,axis=0)

    logL = np.mean(atx * np.cos(y) + btx * np.sin(y) - np.log(Zsum) - Zmax)

    Zcos = rcos.T @ Z / Zsum
    Zsin = rsin.T @ Z / Zsum

    da = (x @ (np.cos(y) - Zcos))#/x.shape[1]
    db = (x @ (np.sin(y) - Zsin))#/x.shape[1]

    return logL, np.stack((da,db))

def log2d(fs, npc=0):
    merror = np.zeros((len(fs),), np.float32)
    for t,f in enumerate(fs):
        dat = np.load(f, allow_pickle=True).item()
        sresp, istim, itrain, itest = utils.compile_resp(dat, npc=npc)

        x = sresp[:, itrain]
        y = istim[itrain]
        NN, NT = np.shape(sresp)

        th_range = np.arange(0, 2*np.pi, 2*np.pi/360)
        rcos = np.cos(th_range)
        rsin = np.sin(th_range)

        a = np.random.randn(2, NN)/1e7 # initializdr with very small values

        eps0 = 0.05 # learning rate
        niter = 801
        lam = .0 # regularization parameter, makes fitting unstable

        logL = np.zeros(niter,)
        pa = np.zeros(a.shape)

        for it in range(niter):
            logL[it], da = log_prob(a, x, y, rcos, rsin)
            pa = .95 * pa + .05 * (da - lam * a)
            if it<20:
                eps = eps0/(20-it)
            a += eps * pa
            #if it%100==0:
            #    print(logL[it])

        dx = a[0].T @ sresp[:, itest]
        dy = a[1].T @ sresp[:, itest]

        apred = np.angle(dx + 1j * dy)
        apred[apred<0] = apred[apred<0] + 2*np.pi

        nangle = 2*np.pi
        error = istim[itest] - apred
        error = np.remainder(error, nangle)
        error[error > nangle/2] = error[error > nangle/2] - nangle
        merror[t] = np.median(np.abs(error)) * 180/np.pi
        print(t, merror[t])
        if t==0:
            errors_ex = error
            stims_ex = istim[itest]

    return merror, errors_ex, stims_ex


def lowstim_decoder(sresp, istim, itrain, itest):
    # allen institute uses np.arange(0.0, 180.0, 30.0) stimuli
    stims = np.arange(0.0, 180.0, 30.0)
    irange = np.arange(-15.0, 180.0, 15.0)

    return stims

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

def independent_decoder(sresp, istim, itrain, itest, nbase=10, nangle=2*np.pi, fitgain=False):
    if nangle==np.pi and istim.max() > np.pi:
        istim = np.remainder(istim.copy(), np.pi)

    A, B, vv, SNR, ypred = fit_indep_model(sresp[:, itrain], istim[itrain], nbase, nangle=nangle, fitgain=fitgain)
    apred, logL, B2, Kup = test_indep_model(sresp[:, itest], A, vv, nbase, nangle=nangle, fitgain=fitgain)


    # preferred stimulus for each neuron
    theta_pref = istim[itrain][np.argmax(ypred, axis=1)]

    error = istim[itest] - apred
    error = np.remainder(error,nangle)
    error[error > nangle/2] = error[error > nangle/2] - nangle

    return apred, error, ypred, logL, SNR, theta_pref, A, B, B2


def test_indep_model(X, A, vv, nbase, xcoef=None, nangle=2*np.pi, fitgain=False):
    # use GPU for optimization
    nodes = 32
    theta = np.linspace(0, nangle, nodes+1)[:-1]

    bubu = np.arange(0,nbase)[:, np.newaxis]
    F1 = np.cos(theta * bubu * (2*np.pi) / nangle)
    F2 = np.sin(theta * bubu * (2*np.pi) / nangle)
    B = np.concatenate((F2,F1), axis=0)
    D = np.concatenate((F1*bubu, -F2*bubu), axis=0)
    B = B[1:, :]
    D = D[1:, :]

    logL = np.zeros((X.shape[1], nodes))
    for k in range(nodes):
        ypred = (A.T @ B[:, k])[:, np.newaxis]
        if fitgain:
            g     = np.sum(ypred * X, axis=0) / np.sum(ypred**2, axis=0)
            ypred = g * ypred
        rez = X - ypred
        logL[:, k] = -np.mean(rez**2/vv[:, np.newaxis], axis=0)

    Kup = utils.upsampling_mat(nodes, int(3200/nodes), nodes/32)
    yup = logL @ Kup.T
    apred = np.argmax(yup, axis=1) / yup.shape[1] * nangle

    return apred, logL, B, Kup


def fit_indep_model(X, istim, nbase, nangle=2*np.pi, lam = .001, fitgain=False):
    theta = istim.astype(np.float32)
    bubu = np.arange(0,nbase)[:, np.newaxis]
    F1 = np.cos(theta * bubu * (2*np.pi) / nangle)
    F2 = np.sin(theta * bubu * (2*np.pi) / nangle)
    B = np.concatenate((F2,F1), axis=0)
    D = np.concatenate((F1*bubu, -F2*bubu), axis=0)
    B = B[1:, :]
    D = D[1:, :]
    A = np.linalg.solve(B @ B.T, B @ X.T)
    #rez = X - A.T @ B

    ypred = A.T @ B
    if fitgain:
        g = np.sum(ypred * X, axis=0) / np.sum(ypred**2, axis=0)
        ypred = g * ypred
    rez = X - ypred
    vv = lam + 1.*np.var(rez, axis=1)

    SNR = np.var(ypred, axis=1) / np.var(rez, axis=1)

    return A, B, vv, SNR, ypred

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

def nbasis_linear(fs, npc=0):
    """ how the decoding varies as a function of the number of basis functions """
    nbasis = [2, 5, 8, 10, 15, 20, 30, 48, 100]
    ntt = [2.5, 5, 7.5, 10]
    errors = np.zeros((len(ntt), len(nbasis), len(fs)), np.float32)
    for i,f in enumerate(fs):
        print('dataset %d'%i)
        dat = np.load(f, allow_pickle=True).item()
        sresp, istim, itrain, itest = utils.compile_resp(dat)

        lam = 1
        nangle = 2 * np.pi
        X = sresp[:,itrain]
        XtX = X @ X.T
        for j,nt in enumerate(ntt):
            for k,nth in enumerate(nbasis):
                if nth>2:
                    sigma = nt / nth

                    # von mises
                    theta_pref = np.linspace(0,2*np.pi,nth+1)[:-1]
                    theta0 = 2 * np.pi / nangle * istim[itrain,np.newaxis] - theta_pref[np.newaxis,:]
                    y = np.exp((np.cos(theta0)-1) / sigma)
                    y = zscore(y, axis=1)
                else:
                    # cosine decoding
                    theta_pref = np.array([0.0])
                    theta0 = 2 * np.pi / nangle * istim[itrain,np.newaxis] - theta_pref[np.newaxis,:]
                    y = np.concatenate((np.cos(theta0[:,:1]), np.sin(theta0[:,:1])), axis=-1)

                ntot = y.shape[0]

                A = fast_ridge(X, y, lam=lam)
                ypred = sresp[:,itest].T @ A

                # circular interpolation of ypred
                if nth>2:
                    Kup = utils.upsampling_mat(y.shape[1])
                    yup = ypred @ Kup.T
                    apred = np.argmax(yup, axis=1) / yup.shape[1] * nangle
                else:
                    apred = np.arctan2(ypred[:,1], ypred[:,0])
                error = istim[itest] - apred
                error = np.remainder(error, nangle)
                error[error > nangle/2] = error[error > nangle/2] - nangle
                errors[j,k,i] = np.median(np.abs(error)) * 180/np.pi
                print(errors[j,k,i])

    return errors, nbasis

def linear_2d(fs, npc=0, lam=5):
    """ cosine / sine decoding """
    errors = np.zeros((len(fs),), np.float32)
    for i,f in enumerate(fs):
        print('dataset %d'%i)
        dat = np.load(f, allow_pickle=True).item()
        sresp, istim, itrain, itest = utils.compile_resp(dat)

    # von mises
    y = np.exp((np.cos(theta0)-1) / sigma)

    # cosine decoding
    theta_pref = np.array([0.0])
    theta0 = 2 * np.pi / nangle * istim[itrain,np.newaxis] - theta_pref[np.newaxis,:]
    y = np.concatenate((np.cos(theta0[:,:1]), np.sin(theta0[:,:1])), axis=-1)

    A = fast_ridge(X, y, lam=lam)
    ypred = sresp[:,itest].T @ A

    apred = np.arctan2(ypred[:,1], ypred[:,0])
    error = istim[itest] - apred
    error = np.remainder(error, nangle)
    error[error > nangle/2] = error[error > nangle/2] - nangle
    errors[i] = np.median(np.abs(error)) * 180/np.pi
    print(errors[i])

    return errors


def vonmises_decoder(sresp, istim, itrain, itest, nth=48, nangle=2*np.pi,
                     lam=1, dcdtype='L2'):
    """ stim ids istim, neural responses sresp (NNxnstim)
        nangle = np.pi if orientations """
    if nangle==np.pi and istim.max() > np.pi:
        istim = np.remainder(istim.copy(), np.pi)

    # von mises
    sigma = 0.1 / (nangle / (2*np.pi))
    theta_pref = np.linspace(0, nangle, nth+1)[:-1]
    theta0 = 2*np.pi/nangle * (istim[itrain,np.newaxis] - theta_pref[np.newaxis,:])
    y = np.exp((np.cos(theta0)-1) / sigma)
    y = zscore(y, axis=1)

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

def derivative_decoder(istim, sresp, itrain, itest, lam=1, nangle=2*np.pi,
                       dcdtype='regression'):
    """ stim ids istim, neural responses sresp (NNxnstim)
        nangle = np.pi if orientations """
    if nangle==np.pi and istim.max() > np.pi:
        istim = np.remainder(istim.copy(), np.pi)

    # difference of vonmises
    sigma = 0.1 / (nangle / (2*np.pi))
    dt = np.pi/32 # difference
    theta_pref = np.linspace(0, nangle, 33)[:-1]
    theta0 = 2*np.pi/nangle * (istim[itrain,np.newaxis] - theta_pref[np.newaxis,:])
    y = np.exp(np.cos(theta0- dt) / sigma) - np.exp(np.cos(theta0 + dt) / sigma)
    y = zscore(y, axis=1)

    X = sresp[:,itrain]

    if dcdtype is 'regression':
        A = fast_ridge(X, y, lam = lam)
    else:
        A = X @ y
    ypred = A.T @ sresp[:, itest]

    D = np.zeros((0,))
    dy = np.zeros((0,))
    nth = len(theta_pref)
    for j in range(nth):
        ds = (istim[itest] - theta_pref[j])%(nangle)
        ds[ds > nangle/2] = ds[ds > nangle/2] - nangle
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

def dense_discrimination(fs, npc=0):
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
    P = np.zeros((len(nskipstim), len(nskip), len(drange2), len(fs)), np.float32)
    #P2 = np.zeros((len(nskip), len(drange2), len(fs)), np.float32)

    for t,f in enumerate(fs):
        print(os.path.basename(f))
        dat = np.load(f, allow_pickle=True).item()

        sresp, istim, itrain, itest = utils.compile_resp(dat, npc=npc)
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
        np.random.seed(seed = 101)
        npop[:, t] = (NN/nskip).astype('int')
        rperm = np.random.permutation(NN)
        for m in range(len(nskipstim)):
            iSS = rperm2[:nstim[m,t]]
            for k in range(len(nskip)):
                iNN = rperm[:npop[k,t]]
                A = fast_ridge(X[np.ix_(iNN, iSS)], y[iSS], lam = 1)
                ypred = (A.T @ Xtest[iNN]).flatten()
                D = np.zeros((0,))
                dy = np.zeros((0,))
                ds = (istim[itest] - theta_pref[0])%(2*np.pi)
                ds[ds>np.pi] = ds[ds>np.pi] - 2*np.pi
                D = np.concatenate((D, ds), axis=0)
                dy = np.concatenate((dy, ypred), axis=0)
                for j,deg in enumerate(drange2):
                    ix = np.logical_and(D>np.pi/180 * (deg-dd), D<np.pi/180 * (deg+dd))
                    P[m, k, j, t] = np.mean(dy[ix]>0)

    return npop, nstim, P, drange2

def run_discrimination(fs, nangles=None, decoder='linear', npc=0):
    if nangles is None:
        nangles = 2*np.pi * np.ones((len(fs),))
    drange = np.arange(-29,30)
    P = np.zeros((len(fs),len(drange)))
    d75 = np.zeros((len(fs),))
    ithres = np.pi/4
    for t,f in enumerate(fs):
        print(os.path.basename(f))
        dat = np.load(f, allow_pickle=True).item()
        sresp, istim, itrain, itest = utils.compile_resp(dat, npc=npc)

        if decoder=='linear':
            D, dy, A = derivative_decoder(istim, sresp, itrain, itest, nangle=nangles[t])
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

def run_decoder(fs, linear=True, nangles=None, npc=0):
    if nangles is None:
        nangles = 2*np.pi * np.ones((len(fs),))
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
            d = vonmises_decoder(sresp, istim, itrain, itest, nangle=nangles[t])
            apred, error = d[0], d[1]
        else:
            d = independent_decoder(sresp, istim, itrain, itest, nangle=nangles[t])
            apred, error, SNR, theta_pref = d[0], d[1], d[4], d[5]

        # save error and stimulus
        errors.append(error)
        stims.append(istim[itest])
        snrs.append(SNR)
        theta_prefs.append(theta_pref)
        E[t] = np.median(np.abs(error)) * 180/np.pi
        print(os.path.basename(f), E[t])

    return E, errors, stims, snrs, theta_prefs

def runspeed_discrimination(fs, all_running, npc=0):
    ntesthalf = 1000
    drange = np.arange(-29, 30, 1)
    P0 = np.zeros((len(fs), len(drange), 2), np.float32)
    d75 = np.zeros((len(fs), 2), np.float32)
    for t,f in enumerate(fs):
        print(os.path.basename(f))
        dat = np.load(f, allow_pickle=True).item()

        sresp, istim, itrain, itest = utils.compile_resp(dat, npc=npc)

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


def layer_discrimination(fs, all_depths, npc=0):
    drange = np.arange(-29, 30, 1)
    P0 = np.zeros((len(fs), len(drange), 2), np.float32)
    d75 = np.zeros((len(fs), 2), np.float32)
    nangle=2*np.pi
    for t,f in enumerate(fs):
        print(os.path.basename(f))
        dat = np.load(f, allow_pickle=True).item()
        depths = all_depths[t]

        sresp, istim, itrain, itest = utils.compile_resp(dat, npc=npc)
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

def chron_discrimination(fs, all_depths, npc=0):
    drange = np.arange(-29, 30, 1)
    P0 = np.zeros((len(fs), len(drange), 2), np.float32)
    d75 = np.zeros((len(fs), 2), np.float32)
    nangle=2*np.pi
    for t,f in enumerate(fs):
        print(os.path.basename(f))
        dat = np.load(f, allow_pickle=True).item()

        sresp, istim, itrain, itest = utils.compile_resp(dat, npc=npc)
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

def dense_decoder(sresp, istim, itrain, itest, lam=1):
    y = istim[itrain]
    X = sresp[:,itrain]
    XtX = X @ X.T
    A = fast_ridge(X, y, lam=lam)
    apred = sresp[:,itest].T @ A
    error = istim[itest] - apred
    return apred, error

def dense_asymptotics(fs, lam=1, npc=0):
    """ linear decoding of densely presented stims as a fcn of neurons and trials """
    nskip = 2**np.linspace(0, 10, 21)
    nskipstim = 2**np.linspace(0, 10, 21)
    Eneur = np.zeros((len(nskip), len(fs)))
    Estim = np.zeros((len(nskipstim), len(fs)))

    npop = np.zeros((len(nskip), len(fs)), 'int')
    nstim = np.zeros((len(nskipstim), len(fs)), 'int')

    errors = []
    stims = []
    snrs = []
    theta_prefs = []
    for t,f in enumerate(fs):
        print('dataset %d'%t)
        dat = np.load(f, allow_pickle=True).item()
        sresp, istim, itrain, itest = utils.compile_resp(dat, npc=npc)
        istim -= istim.mean()

        NN = sresp.shape[0]
        npop[:, t] = (NN/nskip).astype('int')
        np.random.seed(seed = 101)
        rperm = np.random.permutation(NN)
        for k in range(len(nskip)):
            iNN = rperm[:npop[k,t]]
            error = dense_decoder(sresp[iNN], istim, itrain, itest, lam=lam)[1]
            Eneur[k,t] = np.mean((error * 180/np.pi) ** 2)
            #if k==0:
            #print(np.median(np.abs(error))* 180/np.pi)
            #print(k,t,Eneur[k,t])

        nstim[:,t] = (itrain.size/nskipstim).astype('int')
        np.random.seed(seed = 101)
        rperm = np.random.permutation(itrain.size)
        for k in range(len(nskipstim)):
            iSS = rperm[:nstim[k,t]]
            error = dense_decoder(sresp, istim, itrain[iSS], itest, lam=lam)[1]
            Estim[k,t] = np.mean((error * 180/np.pi)**2)
            #if k==0:
            #print(k,t,Estim[k,t])


    return Eneur, Estim, npop, nstim

def run_independent_and_gain(fs, npc=0):
    E = np.zeros((2, len(fs)))
    ccE = np.zeros((2, 2, len(fs)))
    nsplit = np.zeros((len(fs),), 'int')
    nstrips = 8

    for t,f in enumerate(fs):
        dat = np.load(f, allow_pickle=True).item()

        sresp, istim, itrain, itest = utils.compile_resp(dat, npc=npc)
        ypos = np.array([dat['stat'][j]['med'][0] for j in range(len(dat['stat']))])

        # split neurons for decoder into strips (no Z overlap between two sets)
        NN = sresp.shape[0]
        np.random.seed(seed = 101)
        iNN = np.random.permutation(NN)
        
        for fitgain in [0,1]:
            error = independent_decoder(sresp[iNN, :], istim, itrain, itest, fitgain=fitgain)[1]
            E[fitgain,t] = np.median(np.abs(error)) * 180/np.pi
            print('%s error=%2.2f'%(os.path.basename(f), E[fitgain,t] ))
            n1, n2 = utils.stripe_split(ypos[iNN], nstrips)
            err1 = independent_decoder(sresp[iNN[n1]], istim, itrain, itest, fitgain=fitgain)[1]
            err2 = independent_decoder(sresp[iNN[n2]], istim, itrain, itest, fitgain=fitgain)[1]

            ccE[fitgain,0,t] = np.corrcoef(err1, err2)[0,1]
            ccE[fitgain,1,t] = spearmanr(err1, err2)[0]
            print(ccE[fitgain,1,t])
            
    return E, ccE


def asymptotics(fs, linear=True, npc=0):
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

        sresp, istim, itrain, itest = utils.compile_resp(dat, npc=npc)
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

def pc_decoding(fs, nPC, npc=0):
    ''' linearly decode from PCs of data '''

    errors = np.zeros((len(fs), len(nPC)))
    for t,f in enumerate(fs):
        print(os.path.basename(f))
        dat = np.load(f, allow_pickle=True).item()

        sresp, istim, itrain, itest = utils.compile_resp(dat, npc=npc)

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
