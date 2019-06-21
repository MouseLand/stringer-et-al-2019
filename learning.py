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
import utils, decoders
        
def weak_learning(data, istim, itrain, itest, dcdtype='best_neuron'):        
    y = np.sign(istim)    
    NN = data.shape[0]
    ntot = y.shape[0]    
    X = data[:,itrain]   
    mu = np.mean(X, axis=1)
    sd = np.std(X, axis=1)
    X -= mu[:,np.newaxis]
    X /=sd[:,np.newaxis]
    
    xlist = []    
    if dcdtype is 'best_neuron':
        A = np.zeros(NN,)
        cc = X @ zscore(y[itrain])/itrain.size
        ix = np.argmax(np.abs(cc))
        A[ix] = np.sign(cc[ix])       
        xlist.append(cc)
    elif dcdtype is 'one_shot':
        ipos = itrain[y[itrain]>4*np.pi/36]
        ineg = itrain[y[itrain]<-4*np.pi/36]
        ipos = np.random.choice(ipos, (1,))
        ineg = np.random.choice(ineg, (1,))
        A = data[:,  ipos] - data[:,  ineg]
        A = np.squeeze(A)        
        xlist.append(data[:,ipos])
        xlist.append(data[:,ineg])
    elif dcdtype is 'random_projection':
        A = np.random.randn(NN, 100)
        xproj = A.T @ X
        xproj = zscore(xproj, axis=1)
        cc = np.sum(xproj * y[itrain], axis=1)
        imax = np.argmax(np.abs(cc))
        A = A[:, imax] * np.sign(cc[imax])
        A = np.squeeze(A)                
    
    zdata = (data[:, itest] - mu[:,np.newaxis])/sd[:,np.newaxis]
    ypred = A.T @ zdata
    
    return ypred, xlist


def perceptron_learning(sresp, istim, itrain, itest, Ltype = 'sign', lam = 0, eta = 4e-5):
    X = sresp[:,itrain]    
    y = istim[itrain,np.newaxis]    
    y = zscore(y, axis=0)
    y = np.squeeze(y)
    ylabel = np.sign(y)    
    
    NN = X.shape[0]        
    w = 1e-6 * np.random.randn(NN,)
            
    Pcorrect = []
    nstim = 2**np.linspace(4, np.log2(X.shape[1]-1), 21)    
    nstim = np.concatenate(([0, 1, 2, 3, 4, 6, 10], nstim))
    nstim = np.array(nstim)
    nstim = nstim.astype('int')           
    
    D = istim[itest]
    
    for j in range(X.shape[1]):                
        if Ltype is 'regression':
            if np.isin(j, nstim):                
                w = decoders.fast_ridge(X[:, :j+1], ylabel[:j+1], lam=lam)        
        if Ltype is not 'regression':
            ypred = w.T @ X[:,j]
            #ypred = 1./(1+np.exp(-ypred))
            #dsigm  = ypred * (1-ypred)    

            if Ltype is 'Hebb':
                err = ylabel[j]
            elif Ltype is 'full':
                err = ylabel[j] - ypred
            else:            
                err = (ylabel[j] - np.sign(ypred))/2

            dw = err * X[:,j] - lam * w        
            w = w + eta * dw
        if np.isin(j, nstim):
            dy = w.T @ sresp[:, itest]
            pc = (np.mean(dy[D>0]>0) + np.mean(dy[D<0]<0))/2
            Pcorrect.append(pc)
            
    Pcorrect = np.array(Pcorrect)
    return nstim, Pcorrect
    
def train_perceptrons(fs, task_type='hard'):
    thmax = np.pi/6
    if task_type=='hard':
        theta_pref = np.pi/4
        all_thetas = [theta_pref]
    else:    
        all_thetas = np.linspace(0, 2*np.pi, 33)[:-1]
    nstim = np.zeros((len(fs),len(all_thetas), 28))
    perf = np.zeros((4, len(fs), len(all_thetas), 28))
        
    for t,f in enumerate(fs):
        print(os.path.basename(f))
        dat = np.load(f, allow_pickle=True).item()
        sresp, istim, itrain, itest = utils.compile_resp(dat)    

        for j, theta_pref in enumerate(all_thetas):    
            dy = istim - theta_pref
            itrain0=itrain
            itest0=itest
            if task_type!='hard':
                dy = dy%(2*np.pi)
                dy[dy>np.pi] = dy[dy>np.pi] - 2*np.pi
                ix = np.logical_and(np.abs(dy) < thmax, np.abs(dy) > thmax/6)

                itrain0 = itrain[ix[itrain]]
                itest0 = itest[ix[itest]]

            nstim[t,j], perf[0,t,j] = perceptron_learning(sresp, dy, itrain0, itest0,  Ltype='regression', lam = 1)
            _,perf[1,t,j] = perceptron_learning(sresp, dy, itrain0, itest0, Ltype='basic')
            _,perf[2,t,j] = perceptron_learning(sresp, dy, itrain0, itest0, Ltype='full')
            _,perf[3,t,j] = perceptron_learning(sresp, dy, itrain0, itest0, Ltype='Hebb')
            
    return nstim,perf

def train_weak_learners(fs):

    nstim = np.zeros((len(fs), 32))
    perf = np.zeros((3, len(fs), 32))
    theta_pref = np.pi/4
    thmax = np.pi/6
    all_thetas = np.linspace(0, 2*np.pi, 33)[:-1]

    D = np.zeros((0,))
    dy = np.zeros((0,3))

    for t,f in enumerate(fs):
        print(os.path.basename(f))
        dat = np.load(f, allow_pickle=True).item()
        sresp, istim, itrain, itest = utils.compile_resp(dat)        

        for j, theta_pref in enumerate(all_thetas):    
            ds = istim - theta_pref
            ds = ds%(2*np.pi)
            ds[ds>np.pi] = ds[ds>np.pi] - 2*np.pi
            ix = np.logical_and(np.abs(ds) < thmax, np.abs(ds) > thmax/6)

            itrain0 = itrain[ix[itrain]]
            itest0 = itest[ix[itest]]

            ypred1,xlist = weak_learning(sresp, ds, itrain0, itest0, dcdtype = 'best_neuron')
            ypred2,_ = weak_learning(sresp, ds, itrain0, itest0, dcdtype = 'one_shot')
            ypred3,_ = weak_learning(sresp, ds, itrain0, itest0, dcdtype = 'random_projection')

            D = np.concatenate((D, ds[itest0]), axis=0)
            dy = np.concatenate((dy, np.vstack((ypred1,ypred2, ypred3)).T ), axis=0)        

    drange = np.concatenate((np.arange(-29, -4), np.arange(5, 30)))
    P = np.zeros((len(drange),3))
    dd = .5
    for j,deg in enumerate(drange):
        ix = np.logical_and(D>np.pi/180 * (deg-dd), D<np.pi/180 * (deg+dd))
        P[j, :] = np.mean(dy[ix, :]>0, axis=0)
        
    return P, drange, xlist
