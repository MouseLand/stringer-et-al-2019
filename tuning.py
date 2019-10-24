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
import utils

def signal_variance(fs, npc=0):
    sigvar = np.zeros((0,), np.float32)
    for f in fs:
        dat = np.load(f, allow_pickle=True).item()
        sresp, istim, itrain, itest = utils.compile_resp(dat, npc=npc)
        NN,nstim = sresp.shape

        isort = np.argsort(istim)
        sresp = sresp[:,isort]
        A = np.transpose(np.reshape(sresp[:, :(nstim//2)*2], (NN, nstim//2, 2)), (1,0,2))

        A = (A - A.mean(axis=0)) / A.std(axis=0) + 1e-3
        sv0 =(A[:,:,0] * A[:,:,1]).mean(axis=0)
        print(sv0.mean())
        sigvar = np.append(sigvar, sv0, axis=0)
    return sigvar, A

def halfwidth_halfmax(x, y, xmax):
    ''' *x is sorted in ascending order* '''
    nt = x.size
    #halfmax = (y.max() - y.min()) / 2 + y.min()
    imax = np.argmin(np.abs(x - xmax))
    imin1 = np.argmin(np.abs(x - (xmax+np.pi)%(2*np.pi)))
    imin2 = np.argmin(np.abs(x - (xmax-np.pi)%(2*np.pi)))
    ymin = (y[imin1] + y[imin2]) / 2
    halfmax = (y[imax] - ymin) / 2 + ymin

    # right side of curve
    iinds = np.arange(imax+1, imax+10, 1, int)
    iinds[iinds>=nt] = nt - iinds[iinds>=nt]
    dplus = y[iinds]
    dind = np.argmin(np.abs(dplus - halfmax))
    angle_plus = x[iinds[dind]]

    # left side of curve
    iinds = np.arange(imax-10, imax, 1, int)
    # negative values will loop to correct values
    dminus = y[iinds]
    dind = np.argmin(np.abs(dminus - halfmax))
    angle_minus = x[iinds[dind]]
    hwhm = np.abs(angle_plus - angle_minus)
    if hwhm > np.pi:
        hwhm = 2*np.pi - hwhm
    hwhm /= 2

    return hwhm, angle_plus, angle_minus


def population_distances(sresp, istim):
    ''' compute distances between stimuli and embed stimuli with isomap'''
    isort = np.argsort(istim)
    sresp = zscore(sresp, axis=0)
    NN,NS = sresp.shape
    cc = (sresp.T @ sresp) / NN
    dtheta = istim[:,np.newaxis] - istim[np.newaxis,:]
    np.fill_diagonal(cc, np.nan)
    dtheta_aligned = (dtheta+np.pi/2)%(2*np.pi)
    cbinned = utils.binned((dtheta_aligned*180/np.pi).flatten()[~np.isnan(cc.flatten())],
                             cc.flatten()[~np.isnan(cc.flatten())], np.linspace(0,360,80))

    pca = PCA(n_components=100).fit(sresp)
    u = pca.components_.T
    sv = pca.singular_values_
    u *= sv

    print(u.shape)

    model = isomap.Isomap(n_components=3).fit(u)
    embedding = model.embedding_

    return cc, dtheta_aligned, cbinned, embedding

def population_tuning(fs, angle_pref, saveroot):

    # averaged tuning curves
    theta_pref = np.linspace(0, 2*np.pi, 17)[:-1]
    nth = theta_pref.size
    bins = np.linspace(0, 2*np.pi, 65)
    avg_tuning = np.zeros((nth, bins.size-1, len(fs)), np.float32)
    tdiff = np.abs(np.diff(theta_pref).mean())

    thetas = []

    for t,f in enumerate(fs):
        print(f)
        dat = np.load(f, allow_pickle=True).item()
        sresp, istim, itrain, itest = utils.compile_resp(dat)

        # compute averaged tuning curves
        avg_test_curves = np.zeros((nth,itest.size))
        stest = sresp[:,itest]
        for k,tf in enumerate(theta_pref):
            dists = np.abs(tf - angle_pref[t])
            dists[dists>np.pi] = 2*np.pi - dists[dists>np.pi]
            avg_test_curves[k] = stest[dists<tdiff/2].mean(axis=0)
            avg_tuning[k,:,t],_,_ = utils.binned(istim[itest], avg_test_curves[k], bins)

    tbins = bins[:-1]+(bins[1]-bins[0])/2

    return avg_tuning, tbins
