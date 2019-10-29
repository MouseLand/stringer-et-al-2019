import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

def fit_asymptote(x, y, xall, fitexp=False):
    ''' fit y = alpha + beta / sqrt(x)'''
    xi = x.copy()**-0.5
    if xi.ndim < 2:
        xi = xi[:,np.newaxis]
        xall = xall[:,np.newaxis]
    reg = LinearRegression().fit(xi, y)
    beta = reg.coef_
    alpha = reg.intercept_
    r2 = reg.score(xi, y)
    if not fitexp:
        ypred = alpha + np.dot(xall**-0.5, beta)
        par = [alpha]
        for b in beta:
            par.append(b)
        return par, r2, ypred
    if xi.shape[1]==1:
        par0 = [alpha, beta[0], 0.5]
        f = asymp
    else:
        par0 = [alpha, beta[0], beta[1], 0.5, 0.5]
        f = asymp2
    par, mcov = curve_fit(f, x, y, par0)

    if xi.shape[1]==1:
        ypred = asymp(x, par[0], par[1], par[2])
    else:
        ypred = asymp2(x.T, par[0], par[1], par[2], par[3], par[4])
    r2 = np.corrcoef(ypred, y)[0,1]
    print(par, r2)
    if xi.shape[1]==1:
        ypred = asymp(xall, par[0], par[1], par[2])
    else:
        ypred = asymp2(xall.T, par[0], par[1], par[2], par[3], par[4])
    return par, r2**2, ypred

def asymp(x, alpha, beta, t1):
    y = alpha + beta / x**t1
    return y

def asymp2(x, alpha, beta, gamma, t1, t2):
    y = alpha + beta / x[0]**t1 + gamma / x[1]**t2
    return y


def discrimination_threshold(P, x):
    P = (P + 1-P[::-1])/2
    par0 = np.array([5])
    par, mcov = curve_fit(logistic, x, P, par0)
    p75 = - np.log(1/0.75 - 1) * par[0]
    return p75, logistic(x, par)

# psychometric function
def logistic(x, beta):
    return 1. / (1 + np.exp( -x / beta ))

def upsampling_mat(ntot, upfactor = 100, sig = 1):
    xs = np.arange(0, ntot)
    ys = np.linspace(0, ntot, 1+upfactor*ntot)
    ys = ys[:-1]
    ds = np.abs(xs[np.newaxis,:] - xs[:,np.newaxis])
    ds = np.minimum(ds, ntot-ds)
    Kxx = np.exp(-ds**2 / (2*sig**2) )
    ds = np.abs(ys[np.newaxis,:] - xs[:,np.newaxis])
    ds = np.minimum(ds, ntot-ds)
    Kyx = np.exp(-ds**2 / (2*sig**2) )
    Kup = Kyx.T @ np.linalg.inv(Kxx)

    return Kup

def binned(x, y, bins):
    ''' bin x and compute y in each bin, and standard error'''
    nx, be = np.histogram(x, bins=bins)
    ny, be = np.histogram(x, bins=bins, weights=y)
    ne, be = np.histogram(x, bins=bins, weights=y**2)
    ny /= nx
    serr = (ne/nx - ny**2)**0.5
    serr /= (nx-1)**0.5
    tbins = bins[:-1] + (bins[1]-bins[0])/2
    return ny, serr, tbins

def resample_frames(data, torig, tout):
    ''' resample data at times torig at times tout '''
    ''' data is components x time '''
    fs = torig.size / tout.size # relative sampling rate
    data = gaussian_filter1d(data, np.ceil(fs/4), axis=1)
    f = interp1d(torig, data, kind='linear', axis=-1, fill_value='extrapolate')
    dout = f(tout)
    return dout

def compile_resp(dat, nskip=4, npc=0, zscore=True):
    istim = dat['istim']
    # split stims into test and train
    itest = np.zeros((istim.size,), np.bool)
    itest[::nskip] = 1
    itrain = np.ones((istim.size,), np.bool)
    itrain[itest] = 0
    itrain = itrain.nonzero()[0]
    itest = np.nonzero(itest)[0]
    if zscore:
        # subtract off spont PCs
        sresp = (dat['sresp'].copy() - dat['mean_spont'][:,np.newaxis]) / dat['std_spont'][:,np.newaxis]
        if npc > 0:
            sresp = sresp - dat['u_spont'][:,:npc] @ (dat['u_spont'][:,:npc].T @ sresp)
        sresp = sresp[:,:istim.size]
        # zscore sresp across stimuli (so each neuron has mean 0 / std 1 responses)
        ssub0 = sresp.mean(axis=1)
        sstd0 = sresp.std(axis=1) + 1e-6
        sresp = (sresp - ssub0[:,np.newaxis]) / sstd0[:,np.newaxis]
    else:
        sresp = dat['sresp'].copy()
    return sresp, istim, itrain, itest

def stripe_split(ypos, nstrips):
    ymax   = np.max(ypos)
    nby    = np.floor(ymax / nstrips)
    ytrain = np.arange(0,nstrips,2,int)[:,np.newaxis] * nby + np.arange(0,nby-100/nstrips,1,int)[np.newaxis,:]
    ytrain = ytrain.flatten()
    n1     = (ypos[:,np.newaxis] == ytrain[np.newaxis,:]).sum(axis=1).nonzero()[0]
    ytest  = np.arange(1,nstrips,2,int)[:,np.newaxis] * nby + np.arange(0,nby-100/nstrips,1,int)[np.newaxis,:]
    ytest  = ytest.flatten()
    n2     = (ypos[:,np.newaxis] == ytest[np.newaxis,:]).sum(axis=1).nonzero()[0]
    return n1, n2

def get_powerlaw(ss, trange):
    logss = np.log(np.abs(ss))
    y = logss[trange][:,np.newaxis]
    trange += 1
    nt = trange.size
    x = np.concatenate((-np.log(trange)[:,np.newaxis], np.ones((nt,1))), axis=1)
    w = 1.0 / trange.astype(np.float32)[:,np.newaxis]
    b = np.linalg.solve(x.T @ (x * w), (w * x).T @ y).flatten()

    allrange = np.arange(0, ss.size).astype(int) + 1
    x = np.concatenate((-np.log(allrange)[:,np.newaxis], np.ones((ss.size,1))), axis=1)
    ypred = np.exp((x * b).sum(axis=1))
    alpha = b[0]
    return alpha,ypred

def shuff_cvPCA(X, nshuff=10):
    ''' X is 2 x stimuli x neurons '''
    nc = min(1024, X.shape[1])
    ss=np.zeros((nshuff,nc))
    for k in range(nshuff):
        iflip = np.random.rand(X.shape[1]) > 0.5
        X0 = np.float64(X.copy())
        X0[0,iflip] = X[1,iflip]
        X0[1,iflip] = X[0,iflip]

        ss[k]=cvPCA(X0)
    return ss

def cvPCA(X):
    ''' X is 2 x stimuli x neurons '''
    pca = PCA(n_components=min(1024, X.shape[1])).fit(X[0].T)
    u = pca.components_.T
    sv = pca.singular_values_

    xproj = X[0].T @ (u / sv)
    cproj0 = X[0] @ xproj
    cproj1 = X[1] @ xproj
    ss = (cproj0 * cproj1).sum(axis=0)
    return ss

def SVCA(X):
    # compute power law
    # SVCA
    #X -= X.mean(axis=1)[:,np.newaxis]

    NN,NT = X.shape

    # split cells into test and train
    norder = np.random.permutation(NN)
    nhalf = int(norder.size/2)
    ntrain = norder[:nhalf]
    ntest = norder[nhalf:]

    # split time into test and train
    torder = np.random.permutation(NT)
    thalf = int(torder.size/2)
    ttrain = torder[:thalf]
    ttest = torder[thalf:]
    #if ntrain.size > ttrain.size:
    #    cov = X[np.ix_(ntrain, ttrain)].T @ X[np.ix_(ntest, ttrain)]
    #    u,sv,v = svdecon(cov, k=min(1024, nhalf-1))
    #    u = X[np.ix_(ntrain, ttrain)] @ u
    #    u /= (u**2).sum(axis=0)**0.5
    #    v = X[np.ix_(ntest, ttrain)] @ v
    #    v /= (v**2).sum(axis=0)**0.5
    #else:
    cov = X[np.ix_(ntrain, ttrain)] @ X[np.ix_(ntest, ttrain)].T
    u = PCA(n_components=min(1024, nhalf-1), svd_solver='randomized').fit_transform(cov)
    u /= (u**2).sum(axis=0)**0.5
    v = cov.T @ u
    v /= (v**2).sum(axis=0)**0.5

    strain = u.T @ X[np.ix_(ntrain,ttest)]
    stest = v.T @ X[np.ix_(ntest,ttest)]

    # covariance k is uk.T * F * G.T * vk / npts
    scov = (strain * stest).mean(axis=1)
    varcov = (strain**2 + stest**2).mean(axis=1) / 2

    return scov, varcov
