import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import string
from matplotlib import rc, rcParams
from scipy.stats import spearmanr
from scipy.ndimage import gaussian_filter1d
import mainfigs, tuning, utils, decoders

rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False

def linear_decoders(dataroot, saveroot, save_figure=False):
    rc('font', **{'size': 6})#, 'family':'sans-serif'})#,'sans-serif':['Helvetica']})

    ''' makes supplementary figure of decoded error vs stimulus angle '''
    d = np.load(os.path.join(saveroot, 'linear_decoder_all_stims.npy'), allow_pickle=True).item()
    E, errors, stims = d['E'], d['errors'], d['stims']

    IMG = mainfigs.visual_stimuli(dataroot)

    fig = plt.figure(figsize=(6.85,3),facecolor='w',frameon=True, dpi=300)
    yratio = 6.85/3

    # different datasets
    inds = [np.arange(0,6,1,int), np.arange(6,9,1,int), np.arange(9,15,1,int), np.arange(15,18,1,int),
            np.arange(18,21,1,int), np.arange(21,24,1,int), np.arange(24,27,1,int), np.arange(27,30,1,int)]

    xpos = np.linspace(0.08,.89,len(inds))
    ypos = [.11,.48]
    idx = [0,3,7,1,2,4,5,6]
    for k in range(len(inds)):
        ax = fig.add_axes([xpos[k],ypos[1], .08, .08*yratio])
        j=inds[idx[k]][0]
        #for j in range(len(inds[k])):
        ax.scatter(stims[j]*180/np.pi, errors[j]*180/np.pi, s=0.1, color=(0,.5,0))
        ax.text(0.5,1, 'median error \n= %2.2f$^\circ$'%E[j],ha='center', transform=ax.transAxes)
        if k==0:
            ax.set_xlabel(r'stimulus angle ($^\circ$)')
            ax.set_ylabel(r'decoding error ($^\circ$)')
        ax.set_ylim(-25,25)
        ax.set_xticks([0,180,360])

        pos = ax.get_position().get_points()
        ax_inset=fig.add_axes([pos[0][0],pos[1][1]+.07,.1,.1*yratio])
        ax_inset.imshow(IMG[idx[k]], cmap=plt.get_cmap('gray'),vmin=-1*(idx[k]==7),vmax=1)
        ax_inset.axis('off')
        xs,ys=IMG[k].shape
        ac = (0.5,0,.5)
        if idx[k]>3 and idx[k] < 7:
            plt.annotate('',(ys*.75,xs*.75), (ys*.25,xs*.25), arrowprops=dict(facecolor=ac,edgecolor=ac))
        elif idx[k]==3:
            plt.text(0, -.15, '100ms',fontweight='bold',size=6, transform = ax_inset.transAxes)
        elif idx[k]==7:
            plt.text(0, -.15, 'random phase',fontweight='bold',size=6, transform = ax_inset.transAxes)

        ax = fig.add_axes([xpos[k],ypos[0], .08, .08*yratio])
        bins = np.linspace(0,360,30)
        jj=0
        for j in inds[idx[k]]:
            ii = np.digitize(stims[j]*180/np.pi, bins=bins)
            ii -= 1
            ae = np.zeros((len(bins)-1,))
            for i in range(len(bins)-1):
                ae[i] = np.median(np.abs(errors[j][ii==i]*180/np.pi))
            ae_smoothed = gaussian_filter1d(np.concatenate((ae[-5:], ae, ae[:5])), 1)[5:-5]
            ax.plot(bins[:-1], ae_smoothed, color=plt.get_cmap('BuGn')(int(100+jj/len(inds[k])*150)))
            jj+=1
        ax.set_ylim(0,3)
        ax.set_xticks([0,180,360])
        ax.set_yticks([0,1,2,3])
        if k==0:
            ax.set_title('binned, all recordings')
            ax.set_ylabel('abs error ($^\circ$)')
            ax.set_xlabel('stimulus angle ($^\circ$)')
            ax.text(-.9,1.2, 'c', size=12, transform=ax.transAxes)
            ax.text(-.9,3.4, 'b', size=12, transform=ax.transAxes)
            ax.text(-.9,4.6, 'a', size=12, transform=ax.transAxes)

    if save_figure:
        if not os.path.isdir(os.path.join(saveroot, 'figs')):
            os.mkdir(os.path.join(saveroot, 'figs'))
        fig.savefig(os.path.join(saveroot, 'figs/supp_error_angles.pdf'))

    return fig

def stim_props(saveroot, save_figure=False):
    rc('font', **{'size': 6})#, 'family':'sans-serif'})#,'sans-serif':['Helvetica']})

    fig = plt.figure(figsize=(6.85,5),facecolor='w',frameon=True, dpi=300)
    yratio = 6.85/5

    xpos = np.linspace(0.08, 1.1, 5)
    ypos = [0.07, .46, .77]
    bz = .13
    iplot = 0

    ### ----- 2P vs EPHYS ------- ###
    hcol = [[.4,.4,.4], [1,0,0]]
    ttl = ['two-photon calcium imaging', 'electrophysiology (Neuropixels, Allen Brain)']
    for i in range(2):
        if i==0:
            d = np.load(os.path.join(saveroot, 'twop_sigvar.npy'), allow_pickle=True).item()
        else:
            d = np.load(os.path.join(saveroot, 'ephys_sigvar.npy'), allow_pickle=True).item()
        sigvar, twor_ex = d['sigvar'], d['twor_ex']
        sigvar = sigvar[~np.isnan(sigvar)]
        A = twor_ex.copy()
        A = (A - A.mean(axis=0)) / A.std(axis=0) + 1e-3
        sv0 =(A[:,:,0] * A[:,:,1]).mean(axis=0)
        print(sv0.max())
        if i==0:
            idx = np.argsort(sv0)[::-1][499]
        else:
            idx = np.argsort(sv0)[::-1][10]

        ax = fig.add_axes([xpos[2*i], ypos[2], bz, bz*yratio])
        if i==1:
            ax.scatter(twor_ex[:,idx,0], twor_ex[:,idx,1], marker='o', s=16, alpha=0.1,
                   color=[.5,.5,.5], edgecolors='none')
        else:
            ax.scatter(twor_ex[:,idx,0], twor_ex[:,idx,1], marker='o', s=6, alpha=0.25,
                   color=[.5,.5,.5], edgecolors='none')
        #a = np.dot(twor_ex[:,idx,1], twor_ex[:,idx,0]) / np.dot(twor_ex[:,idx,0], twor_ex[:,idx,0])
        #ypred = a * twor_ex[:,idx,0]
        #ax.plot(twor_ex[:,idx,0], ypred, color='k')
        ax.text(.75,.45, 'r=%2.2f'%sv0[idx], transform=ax.transAxes, ha='left')
        if i==0:
            ax.set_xlabel('response repeat 1 \n(z-scored)')
            ax.set_ylabel('response repeat 2 \n(z-scored)')
            ax.set_xlim([-1,12])
            ax.set_ylim([-1,12])
            ax.set_xticks([0,5,10])
            ax.set_yticks([0,5,10])
        else:
            ax.set_xlabel('response repeat 1 \n(spikes)')
            ax.set_ylabel('response repeat 2 \n(spikes)')
            ax.set_xlim([-0.5,7.5])
            ax.set_ylim([-0.5,7.5])
            ax.set_xticks([0,2,4,6])
            ax.set_yticks([0,2,4,6])
        ax.text(0.1,.95,'example neuron', transform=ax.transAxes)
        ax.text(-.5,1.07,string.ascii_lowercase[iplot],size=12,transform=ax.transAxes)
        ax.text(-.3,1.09,ttl[i],size=8,transform=ax.transAxes, fontstyle='italic')
        iplot+=1

        ax = fig.add_axes([xpos[2*i+1], ypos[2], bz, bz*yratio])
        nb=ax.hist(sigvar,100, color=hcol[0])
        yp=nb[0].max()
        ms = np.nanmean(sigvar)
        ax.scatter(ms, yp, s=20, marker='v', color='k')
        ax.text(ms*1.7, yp, 'mean=%2.2f'%ms, ha='left')
        ax.set_xlabel('repeat correlation (r)')
        ax.set_ylabel('# of neurons')
        #ax.set_xlim(-0.1,1.5)


    static = np.load(os.path.join(saveroot, 'avgneur_static.npy'), allow_pickle=True).item()
    drifting = np.load(os.path.join(saveroot, 'avgneur_drifting.npy'), allow_pickle=True).item()
    theta_pref = np.linspace(0, 2*np.pi, 17)[:-1]

    d = np.load(os.path.join(saveroot, 'popdist.npy'), allow_pickle=True).item()
    cbinned, dtheta_aligned, embedding, cc, istim = d['cbinned'], d['dtheta_aligned'], d['embedding'], d['cc'], d['istim']
    nc, cerr, tbins = cbinned
    isort = np.argsort(istim)

    ax=fig.add_axes([.7,ypos[0]-.1, .32,.32*yratio], projection='3d')
    x = embedding[isort,1]
    y = embedding[isort,2]
    ang = np.pi / 180 * (160-45) - 0.1*np.pi
    xx = x * np.cos(ang) + y * np.sin(ang)
    yy = -x * np.sin(ang) + y * np.cos(ang)
    zz = embedding[isort,0]
    ax.scatter(xx,yy,zz,
               c=istim[isort],marker='.', s=5, cmap='twilight', alpha=0.2)
    #npts = xx.size
    #tb = 100
    #pt = (npts//tb)*tb
    #bxx = np.reshape(xx[:npt], (npts//tb,tb)).mean(axis=-1)
    #byy = np.reshape(yy[:npt], (npts//tb,tb)).mean(axis=-1)
    #bzz = np.reshape(zz[:npt], (npts//tb,tb)).mean(axis=-1)
    #ax.plot(bxx,byy,bzz, color='k', lw=1.5)

    ir = 500
    ax.scatter(xx[::5], yy[::5], -ir,
               s=.4, color=(0.7,.7,.7),alpha=.5,marker='.')
    ax.scatter(-ir, yy[::5], zz[::5],
               s=.4, color=(0.7,.7,.7),alpha=.5,marker='.')
    ax.scatter(xx[::5], -ir, zz[::5],
               s=.4, color=(0.7,.7,.7),alpha=.5,marker='.')
    #ax.set_title('ISOMAP embedding of stimuli             ', fontsize=6)
    ax.set_ylim(-ir,ir)
    ax.set_xlim(-ir,ir)
    ax.set_zlim(-ir,ir)
    #ax.set_xticks([-300,0,300])
    ax.set_xticks([-ir,0,ir])
    ax.set_yticks([-ir,0,ir])
    ax.set_zticks([-ir,0,ir])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.view_init(20,45)

    for i in range(2):
        if i == 0:
            d = static.copy()
        else:
            d = drifting.copy()

        ax = fig.add_axes([xpos[2*i], ypos[1], bz, bz*yratio])
        ths = []
        cmap = plt.get_cmap('pink')
        cmap = cmap(np.linspace(0,1,7))
        for k,th in enumerate(d['thetas']):
            ths=np.concatenate((ths,th))
            nbs,bbs=np.histogram(th,16,density=True)
            ax.plot((bbs[:-1] + (bbs[1]-bbs[0])/2)*180/np.pi, nbs, lw=1, color=cmap[k])
        ax.set_ylim(0,.35)
        ax.set_xticks([0,180,360])
        if i==0:
            ax.set_xlabel('preferred angle ($^\circ$)')
            ax.set_title('static gratings        ', fontstyle='italic')
        else:
            ax.set_xlabel('preferred direction ($^\circ$)')
            ax.set_title('drifting gratings      ', fontstyle='italic')
        ax.set_ylabel('fraction')
        ax.text(-.4,1.05,string.ascii_lowercase[iplot],size=12,transform=ax.transAxes)
        iplot+=1

        xx,yy = np.meshgrid(np.arange(0,100)/8, np.arange(0,100)/8)
        for j in range(5):
            ax = fig.add_axes([xpos[2*i]-.01+ j*.031, ypos[1]-.11, 0.025, 0.025*yratio])
            gratings = np.cos(xx*np.cos(j*np.pi/2) + yy*np.sin(j*np.pi/2) + np.logical_and(j>1,j<4)*np.pi)
            gratings[gratings<0]=0
            gratings[gratings>0]=1
            ax.imshow(gratings, cmap=plt.get_cmap('gray'))
            ax.set_yticks([])
            ax.set_xticks([])
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            if i==1:
                ac = (0.8,0.2,.8)
                ang = j*np.pi/2 + 0
                ax.annotate('',(50+45*np.cos(ang+np.pi),50+45*np.sin(ang)),
                               (50+45*np.cos(ang), 50-45*np.sin(ang)),
                            arrowprops=dict(arrowstyle='<|-, head_width=0.5, head_length=0.3',
                                            facecolor=ac,edgecolor=ac,lw=1))
            ax.set_xlim(0,100)
            ax.set_ylim(0,100)

        cmap = plt.get_cmap('twilight_shifted')
        nth = d['avg_tuning'].shape[0]
        tbins = d['tbins']
        cmap = cmap(np.linspace(0,1,nth))
        hwhm = np.zeros((nth,))

        ax = fig.add_axes([xpos[2*i + 1], ypos[1], bz, bz*yratio])
        avg_tuning = d['avg_tuning'].copy()
        for k in range(nth):
            tun = avg_tuning[k].mean(axis=-1)
            semy = avg_tuning[k].std(axis=-1) / np.sqrt(avg_tuning.shape[-1]-1)
            semy /= tun.max()**0.5
            tun -= tun.min()
            tun /= tun.max()
            if k==0:
                ax.set_title('normalized', size=6)
                ax.set_ylabel('response')
            x = tbins * 180/np.pi
            ax.plot(x, tun, color=cmap[k], lw=0.5)
            ax.fill_between(x, tun-semy, tun+semy, facecolor=cmap[k], alpha=0.5)
            hwhm[k], angle_plus, angle_minus = tuning.halfwidth_halfmax(tbins, tun, theta_pref[k])
        ax.set_xticks([0,180,360])
        if i==0:
            ax.set_xlabel('stimulus angle ($^\circ$)')
        else:
            ax.set_xlabel('stimulus direction ($^\circ$)')
        ax.set_yticks([0,.5,1.0])
        ax.set_ylim(-.25,1.1)
        ax.text(-.4,1.05,string.ascii_lowercase[iplot],size=12,transform=ax.transAxes)
        iplot+=1


    xpos = [0.1, .44, .7]
    d = np.load(os.path.join(saveroot, 'popdist.npy'), allow_pickle=True).item()
    cbinned, dtheta_aligned, embedding, cc, istim = d['cbinned'], d['dtheta_aligned'], d['embedding'], d['cc'], d['istim']
    nc, cerr, tbins = cbinned
    isort = np.argsort(istim)

    ax=fig.add_axes([xpos[0],ypos[0],bz,bz*yratio])
    im=ax.imshow(cc[np.ix_(isort[::20],isort[::20])], vmin=-.1, vmax=.3)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title('Correlation between neural patterns', fontsize=6)
    ax.text(-.5, 1.05, 'g', size=12, transform=ax.transAxes)
    ax.set_xlabel('trials sorted by angle')
    ax.set_ylabel('trials sorted by angle')
    ax2=fig.add_axes([xpos[0]+bz*1.1, ypos[0]+bz*.5, .01,.1])
    cb=plt.colorbar(im, ax2, ticks=[0,.3])

    ax=fig.add_axes([xpos[1],ypos[0], bz, bz*yratio])
    #for n in np.random.randint(0, NS-1, 10):
    ax.scatter((dtheta_aligned*180/np.pi).flatten()[::5000], cc.flatten()[::5000],s=1,color=(.5,.5,.5), alpha=.05)
    ax.plot(tbins, nc, color='k',lw=0.5)
    ax.fill_between(tbins, nc-cerr, nc+cerr, color='k', alpha=0.5)
    ax.set_xticks([0,90,180,270,360])
    ax.set_xticklabels(['-90','0','90','180','270'])
    ax.set_xlabel('stimulus angle difference ($^\circ$)')
    ax.text(-.7, 1.05, 'h', size=12, transform=ax.transAxes)
    ax.text(1.8, 1.05, 'i', size=12, transform=ax.transAxes)
    ax.set_ylabel('correlation between \nneural patterns')
    ax.set_ylim(-.1,.3)
    ax.set_yticks(np.arange(-.1,.4,.1))

    if save_figure:
        if not os.path.isdir(os.path.join(saveroot, 'figs')):
            os.mkdir(os.path.join(saveroot, 'figs'))
        fig.savefig(os.path.join(saveroot, 'figs/supp_stimprops.pdf'))

    return fig


def stim_distances(saveroot, save_figure=False):
    rc('font', **{'size': 6})#, 'family':'sans-serif'})#,'sans-serif':['Helvetica']})

    fig = plt.figure(figsize=(6.85,5),facecolor='w',frameon=True, dpi=300)
    yratio = 6.85/5

    static = np.load(os.path.join(saveroot, 'avgneur_static.npy'), allow_pickle=True).item()
    drifting = np.load(os.path.join(saveroot, 'avgneur_drifting.npy'), allow_pickle=True).item()
    theta_pref = np.linspace(0, 2*np.pi, 17)[:-1]

    xpos = np.linspace(0.08, 1.1, 5)
    ypos = [0.07, .42, .77]
    bz = .13

    for i in range(2):
        if i == 0:
            d = static.copy()
        else:
            d = drifting.copy()

        ax = fig.add_axes([xpos[0], ypos[2-i], bz, bz*yratio])
        ths = []
        cmap = plt.get_cmap('pink')
        cmap = cmap(np.linspace(0,1,7))
        for k,th in enumerate(d['thetas']):
            ths=np.concatenate((ths,th))
            nbs,bbs=np.histogram(th,16,density=True)
            ax.plot((bbs[:-1] + (bbs[1]-bbs[0])/2)*180/np.pi, nbs, lw=1, color=cmap[k])
        ax.set_ylim(0,.35)
        ax.set_xticks([0,180,360])
        if i==0:
            ax.set_xlabel('preferred angle ($^\circ$)')
            ax.set_title('static gratings        ', fontstyle='italic')
        else:
            ax.set_xlabel('preferred direction ($^\circ$)')
            ax.set_title('drifting gratings      ', fontstyle='italic')
        ax.set_ylabel('fraction')
        ax.text(-.4,1.05,string.ascii_lowercase[i*4],size=12,transform=ax.transAxes)

        xx,yy = np.meshgrid(np.arange(0,100)/8, np.arange(0,100)/8)
        for j in range(5):
            ax = fig.add_axes([xpos[0]-.01+ j*.031, ypos[2-i]-.11, 0.025, 0.025*yratio])
            gratings = np.cos(xx*np.cos(j*np.pi/2) + yy*np.sin(j*np.pi/2) + np.logical_and(j>1,j<4)*np.pi)
            gratings[gratings<0]=0
            gratings[gratings>0]=1
            ax.imshow(gratings, cmap=plt.get_cmap('gray'))
            ax.set_yticks([])
            ax.set_xticks([])
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            if i==1:
                ac = (0.8,0.2,.8)
                ang = j*np.pi/2 + 0
                ax.annotate('',(50+45*np.cos(ang+np.pi),50+45*np.sin(ang)),
                               (50+45*np.cos(ang), 50-45*np.sin(ang)),
                            arrowprops=dict(arrowstyle='<|-, head_width=0.5, head_length=0.3',
                                            facecolor=ac,edgecolor=ac,lw=1))
            ax.set_xlim(0,100)
            ax.set_ylim(0,100)

        cmap = plt.get_cmap('twilight_shifted')
        nth = d['avg_tuning'].shape[0]
        tbins = d['tbins']
        cmap = cmap(np.linspace(0,1,nth))
        hwhm = np.zeros((nth,))
        for j in range(3):
            ax = fig.add_axes([xpos[j+1], ypos[2-i], bz, bz*yratio])
            avg_tuning = d['avg_tuning'].copy()
            for k in range(nth):
                if j==0:
                    tun = avg_tuning[k,:,-1]
                    semy = 0
                    if k==0:
                        ax.set_title('population tuning curves', size=6)
                        ax.set_ylabel('response')
                        if i==0:
                            ax.text(1.2, .9, 'pref angles:', ha='right', transform=ax.transAxes)
                        else:
                            ax.text(1.2, .9, 'pref directions:', ha='right', transform=ax.transAxes)
                    if k%2==0:
                        ax.text(1.2,.8-k*.05, '%d$^\circ$'%(int(theta_pref[k]*180/np.pi)), ha='right',
                                transform=ax.transAxes, color=cmap[k])
                elif j==1:
                    tun = avg_tuning[k].mean(axis=-1)
                    semy = avg_tuning[k].std(axis=-1) / np.sqrt(avg_tuning.shape[-1]-1)
                    if k==0:
                        ax.set_title('all recordings', size=6)
                        ax.set_ylabel('response')
                elif j==2:
                    tun = avg_tuning[k].mean(axis=-1)
                    semy = avg_tuning[k].std(axis=-1) / np.sqrt(avg_tuning.shape[-1]-1)
                    semy /= tun.max()**0.5
                    tun -= tun.min()
                    tun /= tun.max()
                    if k==0:
                        ax.set_title('normalized', size=6)
                        ax.set_ylabel('response')
                x = tbins * 180/np.pi
                ax.plot(x, tun, color=cmap[k], lw=0.5)
                ax.fill_between(x, tun-semy, tun+semy, facecolor=cmap[k], alpha=0.5)
                hwhm[k], angle_plus, angle_minus = tuning.halfwidth_halfmax(tbins, tun, theta_pref[k])
            ax.set_xticks([0,180,360])
            if i==0:
                ax.set_xlabel('stimulus angle ($^\circ$)')
            else:
                ax.set_xlabel('stimulus direction ($^\circ$)')
            ax.set_yticks([0,.5,1.0])
            ax.set_ylim(-.25,1.1)
            ax.text(-.4,1.05,string.ascii_lowercase[j+i*4+1],size=12,transform=ax.transAxes)

    d = np.load(os.path.join(saveroot, 'popdist.npy'), allow_pickle=True).item()
    cbinned, dtheta_aligned, embedding, cc, istim = d['cbinned'], d['dtheta_aligned'], d['embedding'], d['cc'], d['istim']
    nc, cerr, tbins = cbinned
    isort = np.argsort(istim)

    xpos = [0.1, .44, .7]

    ax=fig.add_axes([xpos[0],ypos[0],bz,bz*yratio])
    im=ax.imshow(cc[np.ix_(isort[::20],isort[::20])], vmin=-.1, vmax=.3)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title('Correlation between neural patterns', fontsize=6)
    ax.text(-.5, 1.05, 'I', size=12, transform=ax.transAxes)
    ax.set_xlabel('trials sorted by angle')
    ax.set_ylabel('trials sorted by angle')
    ax2=fig.add_axes([.29, .1+.25*yratio, .01,.1])
    cb=plt.colorbar(im, ax2, ticks=[0,.3])

    ax=fig.add_axes([xpos[1],ypos[0], bz, bz*yratio])
    #for n in np.random.randint(0, NS-1, 10):
    ax.scatter((dtheta_aligned*180/np.pi).flatten()[::5000], cc.flatten()[::5000],s=1,color=(.5,.5,.5), alpha=.05)
    ax.plot(tbins, nc, color='k',lw=0.5)
    ax.fill_between(tbins, nc-cerr, nc+cerr, color='k', alpha=0.5)
    ax.set_xticks([0,90,180,270,360])
    ax.set_xticklabels(['-90','0','90','180','270'])
    ax.set_xlabel('stimulus angle difference ($^\circ$)')
    ax.text(-.7, 1.05, 'J', size=12, transform=ax.transAxes)
    ax.text(1.8, 1.05, 'K', size=12, transform=ax.transAxes)
    ax.set_ylabel('correlation between \nneural patterns')
    ax.set_ylim(-.1,.3)
    ax.set_yticks(np.arange(-.1,.4,.1))

    ax=fig.add_axes([xpos[2],ypos[0]-.1, .3,.3*yratio], projection='3d')
    x = embedding[isort,1]
    y = embedding[isort,2]
    ang = np.pi * 0.9
    xx = x * np.cos(ang) + y * np.sin(ang)
    yy = -x * np.sin(ang) + y * np.cos(ang)
    zz = embedding[isort,0]
    ax.scatter(xx,yy,zz,
               c=istim[isort], s=1, cmap='twilight',alpha=1)#, depthshade=True)
    cmap = plt.get_cmap('twilight')
    cmap = cmap(np.linspace(0,1,int(isort.size)))

    ir = 500
    ax.scatter(xx[::3], yy[::3], -ir,
               s=.4, color=(0.5,.5,.5),alpha=.4)
    ax.scatter(-ir, yy[::3], zz[::3],
               s=1, color=(0.5,.5,.5),alpha=.4)
    ax.scatter(xx[::3], -ir, zz[::3],
               s=1, color=(0.5,.5,.5),alpha=.4)
    #ax.set_title('ISOMAP embedding of stimuli             ', fontsize=6)
    ax.set_ylim(-ir,ir)
    ax.set_xlim(-ir,ir)
    ax.set_zlim(-ir,ir)
    #ax.set_xticks([-300,0,300])
    ax.set_xticks([-ir,0,ir])
    ax.set_yticks([-ir,0,ir])
    ax.set_zticks([-ir,0,ir])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.view_init(20,45)


    if save_figure:
        if not os.path.isdir(os.path.join(saveroot, 'figs')):
            os.mkdir(os.path.join(saveroot, 'figs'))
        fig.savefig(os.path.join(saveroot, 'figs/supp_mani.pdf'))

    return fig

def pc_errors(dataroot, saveroot, save_figure=False):
    rc('font', **{'size': 6})#, 'family':'sans-serif'})#,'sans-serif':['Helvetica']})

    fig = plt.figure(figsize=(5,4.5),facecolor='w',frameon=True, dpi=300)
    yratio = 5/4.5
    bz = .16
    xpos = [.1, .4, .75]
    ypos = [.1, .43, .76]
    iplot=0
    labx = -0.62
    laby = 1.08

    d = np.load(os.path.join(saveroot, 'independent_decoder_and_gain.npy'), allow_pickle=True).item()
    E = d['E']
    dat = np.load(os.path.join(dataroot, 'gratings_static_TX39_2019_05_02_1.npy'), allow_pickle=True).item()
    sresp, istim, itrain, itest = utils.compile_resp(dat)
    apred, error, ypred, logL, SNR, theta_pref, A, B, B2 = decoders.independent_decoder(sresp, istim, itrain, itest, fitgain=True)
    print(np.median(np.abs(error)) * 180/np.pi)
    Apred = A.T @ B2

    d = np.load(os.path.join(saveroot, 'log2d_model.npy'), allow_pickle=True).item()
    merrors, errors_ex, stims_ex = d['merror'], d['errors_ex'], d['stims_ex']

    d = np.load(os.path.join(saveroot, 'pcdecode.npy'), allow_pickle=True).item()
    errors, atrues, apreds, nPC = d['errors'], d['atrues'], d['apreds'], d['nPC']

    dlin = np.load(os.path.join(saveroot, 'linear_decoder_asymp.npy'), allow_pickle=True).item()

    grn = [1,.2,0]
    ax = fig.add_axes([xpos[0],ypos[2],bz,bz*yratio])
    ax.text(labx, laby, string.ascii_lowercase[iplot], transform=ax.transAxes, size=12)
    ax.text(-.35,1., 'independent decoder\n with multiplicative gain', size=6, transform=ax.transAxes)
    ax.axis('off')
    iplot += 1
    isort=np.argsort(SNR)[::-1]
    iangle = np.arange(0,32,1,int) * 360/32
    xs = 0.1
    dx = 0.11
    dy = dx * yratio
    for k,idx in enumerate([100,3000,5090,50]):
        ax=fig.add_axes([xpos[0]-.04+(k%2)*dx, ypos[2]-.06+dy*((3-k)//2), xs,xs*yratio])
        ax.plot(iangle, Apred[isort[idx]], color='k', lw=1)
        ax.plot(iangle, Apred[isort[idx]]*2, '--', color='r', lw=1)
        ax.set_ylim([-2,5.5])
        ax.axis('off')
        ax.text(0,-.08,'neuron %d'%(k+1), transform=ax.transAxes)
        if k==0:
            ax.text(0.5,.7,'"high gain"', color='r', transform=ax.transAxes)
        elif k==2:
            ax.text(-.3,0.55,'tuning curves', transform=ax.transAxes, rotation=90)

    ax = fig.add_axes([xpos[1],ypos[2],bz,bz*yratio])
    ax.scatter(istim[itest]* 180/np.pi, apred*180/np.pi, marker='.', alpha=0.5,
                 s=2, color = grn, edgecolors='none')
    ax.set_xlabel(r'true angle ($^\circ$)')
    ax.set_ylabel(r'decoded angle ($^\circ$)')
    ax.set_xticks([0, 180, 360])
    ax.set_yticks([0, 180, 360])
    ax.text(-.3,1.1,'Test trials',size=6, transform=ax.transAxes)
    ax.text(labx, laby, string.ascii_lowercase[iplot], transform=ax.transAxes, size=12)
    iplot+=1

    ax = fig.add_axes([xpos[2],ypos[2],bz,bz*yratio])
    nb=plt.hist(error* 180/np.pi, np.linspace(0,25, 21), color = grn)
    merror = np.median(np.abs(error))*180/np.pi
    ax.scatter(merror, nb[0].max()*1.05, marker='v',color=[0,.0,0])
    ax.text(merror-2, nb[0].max()*1.13, '%2.2f$^\circ$ = median error'%merror,fontweight='bold')
    ax.set_xlabel(r'absolute angle error ($^\circ$)')
    ax.set_ylabel('trial counts')
    ax.set_xlim([0,20])
    ax.text(labx, laby, string.ascii_lowercase[iplot], transform=ax.transAxes, size=12)

    axins = fig.add_axes([xpos[2]+bz*.95, ypos[2]+bz*.5, .06,.06*yratio])
    axins.hist(E[1], 3, color=grn)
    axins.set_xlabel('median error')
    axins.set_ylabel('recordings')
    axins.set_yticks([0,3])
    axins.set_xticks([2,2.5])
    #axins.set_xlim([2,3.2])
    iplot+=1

    ax = fig.add_axes([xpos[0],ypos[1],bz,bz*yratio])
    ax.text(-.3,1.1,
            r'$\log \, p(\theta | \mathbf{r} ) = \mathbf{a} \cdot \mathbf{r} cos(\theta)$',
            transform=ax.transAxes)
    ax.text(-.0,.95,
            r'          $ + \mathbf{b} \cdot \mathbf{r} sin(\theta)$', transform=ax.transAxes)
    ax.text(-.0,.8,
            r'          $ - \log \, Z(\mathbf{a}, \mathbf{b}, \mathbf{r})$',
            transform=ax.transAxes)
    grn = [0.3,0.8,0]
    ax.text(labx, laby, string.ascii_lowercase[iplot], transform=ax.transAxes, size=12)
    ax.axis('off')

    ax = fig.add_axes([xpos[0]-.03,ypos[1]-.05,bz,bz*yratio])
    nv = 1
    nn = 10
    npc = 2
    cmap2 = plt.get_cmap('twilight_shifted')(np.linspace(0,0.9,nv))
    colors = [np.zeros((nn,3)), np.ones((npc,3))*0.5, np.array(grn)[np.newaxis,:]]
    mainfigs.draw_neural_net(ax, 0,1,0,1, [nn,npc,nv], colors)
    ax.axis('off')
    ax.text(0,1,'neurons', transform=ax.transAxes, ha='center')
    ax.text(0.5,.8,'cos', transform=ax.transAxes, ha='center')
    ax.text(0.5,.3,'sin', transform=ax.transAxes, ha='center')
    ax.text(1,.6,'logL', transform=ax.transAxes, ha='center')
    iplot += 1

    ax = fig.add_axes([xpos[1],ypos[1],bz,bz*yratio])
    ax.scatter(stims_ex* 180/np.pi, ((errors_ex+stims_ex) * 180/np.pi)%360, marker='.', alpha=0.5,
                 s=2, color = grn, edgecolors='none')
    ax.set_xlabel(r'true angle ($^\circ$)')
    ax.set_ylabel(r'decoded angle ($^\circ$)')
    ax.set_xticks([0, 180, 360])
    ax.set_yticks([0, 180, 360])
    ax.text(-.3,1.1,'Test trials',size=6, transform=ax.transAxes)
    ax.text(labx, laby, string.ascii_lowercase[iplot], transform=ax.transAxes, size=12)
    iplot+=1

    ax = fig.add_axes([xpos[2],ypos[1],bz,bz*yratio])
    nb=plt.hist(errors_ex* 180/np.pi, np.linspace(0,25, 21), color = grn)
    merror = np.median(np.abs(errors_ex))*180/np.pi
    ax.scatter(merror, nb[0].max()*1.05, marker='v',color=[0,.0,0])
    ax.text(merror-2, nb[0].max()*1.13, '%2.2f$^\circ$ = median error'%merror,fontweight='bold')
    ax.set_xlabel(r'absolute angle error ($^\circ$)')
    ax.set_ylabel('trial counts')
    ax.set_xlim([0,20])
    ax.text(labx, laby, string.ascii_lowercase[iplot], transform=ax.transAxes, size=12)

    axins = fig.add_axes([xpos[2]+bz*.95, ypos[1]+bz*.5, .06,.06*yratio])
    axins.hist(merrors, 3, color=grn)
    axins.set_xlabel('median error')
    axins.set_ylabel('recordings')
    axins.set_yticks([0,3])
    #axins.set_xlim([2,3.2])
    iplot+=1

    cmap = plt.get_cmap('viridis')
    cmap = cmap(np.linspace(0,1,len(nPC)))
    ax = fig.add_axes([xpos[0],ypos[0],bz,bz*yratio])
    ax.text(labx, laby,string.ascii_lowercase[iplot],size=12, transform=ax.transAxes)
    ax.axis('off')
    ax = fig.add_axes([xpos[0]-.03,ypos[0]-.02,bz,bz*yratio])
    nv = 7
    nn = 10
    npc = 4
    cmap2 = plt.get_cmap('twilight_shifted')(np.linspace(0,0.9,nv))
    colors = [np.zeros((nn,3)), np.ones((npc,3))*0.5, cmap2]
    mainfigs.draw_neural_net(ax, 0,1,0,1, [nn,npc,nv], colors)
    ax.axis('off')
    ax.text(0,1,'neurons', transform=ax.transAxes, ha='center')
    ax.text(0.5,1,'PCs', transform=ax.transAxes, ha='center')
    ax.text(1,1,'super-\nneurons', transform=ax.transAxes, ha='center')
    iplot+=1

    ax = fig.add_axes([xpos[1],ypos[0],bz,bz*yratio])
    for k in range(len(nPC)):
        ax.scatter(atrues[k], apreds[k], color=cmap[k], s=3, marker='.', alpha=0.5, edgecolors='none')
        ax.text(1.3,.0+k*.1, '%d'%nPC[k], transform=ax.transAxes, color=cmap[k], ha='right')
    ax.text(1.3,.0+(k+1)*.1, '# of PCs', transform=ax.transAxes, color='k', ha='right')
    ax.set_xlabel('true angle ($^\circ$)')
    ax.set_ylabel('decoded angle ($^\circ$)')
    ax.set_xticks([0, 180, 360])
    ax.set_yticks([0, 180, 360])
    ax.text(labx, laby,string.ascii_lowercase[iplot],size=12, transform=ax.transAxes)
    iplot+=1

    ax = fig.add_axes([xpos[2],ypos[0],bz,bz*yratio])
    for k in range(len(nPC)):
        ax.errorbar(nPC[k], errors[:,k].mean(axis=0),
                    errors[:,k].std(axis=0)/np.sqrt(6), markersize=20,
                    color=cmap[k], zorder=5)
        ax.scatter(nPC[k], errors[:,k].mean(axis=0), s=3,
                 color=cmap[k], zorder=10)
    ax.semilogx(nPC, errors.mean(axis=0), color=(0.5,.5,.5))
    ax.semilogx(nPC, dlin['E'][0,0,:].mean(axis=-1) * np.ones(nPC.shape),'--', color='k', lw=1)
    ax.text(0.1,0.04, 'all neurons', transform=ax.transAxes)
    ax.set_xlabel('number of PCs')
    ax.set_ylabel('decoding error ($^\circ$)')
    ax.set_xticks(nPC[::2])
    ax.set_xticklabels(['4', '16','64', '256', '1024', '4096'], size=4)
    ax.set_xlim(nPC[0]-.5,nPC[-1]+1300)
    ax.set_ylim(0,8)
    ax.text(labx, laby,string.ascii_lowercase[iplot],size=12, transform=ax.transAxes)

    if save_figure:
        if not os.path.isdir(os.path.join(saveroot, 'figs')):
            os.mkdir(os.path.join(saveroot, 'figs'))
        fig.savefig(os.path.join(saveroot, 'figs/supp_PC_error.pdf'))

    return fig

def discr_all(saveroot, save_figure=False):
    rc('font', **{'size': 6})#, 'family':'sans-serif'})#,'sans-serif':['Helvetica']})

    fig = plt.figure(figsize=(3,4),facecolor='w',frameon=True, dpi=300)
    yratio = 3 / 4

    d = np.load(os.path.join(saveroot, 'chron_discrimination.npy'), allow_pickle=True).item()
    drange = d['drange']
    P0 = d['P'].mean(axis=0)
    Pall = P0

    d = np.load(os.path.join(saveroot, 'layer_discrimination.npy'), allow_pickle=True).item()
    Pall = np.concatenate((Pall, d['P'].mean(axis=0)), axis=-1)

    nn = np.load(os.path.join(saveroot, 'nn_discrimination.npy'), allow_pickle=True).item()
    rf = np.load(os.path.join(saveroot, 'rf_discrimination.npy'), allow_pickle=True).item()
    Pall = np.concatenate((Pall, nn['P'].mean(axis=0)[:,np.newaxis]), axis=-1)
    Pall = np.concatenate((Pall, rf['P'].mean(axis=0)[:,np.newaxis]), axis=-1)

    ttl = ['Original', 'Chronological train/test', 'Layers 2/3',
           'Layer 4', 'neural network', 'random forest']

    cols=[[0,.5,0], [0,.5,0]]
    x0 = 0.23
    x1 = 0.7
    y0 = .09
    y1 = 0.45
    y2 = 0.78
    xpos = [x0, x1, x0, x1, x0, x1]
    ypos = [y0, y0, y1, y1, y2, y2]
    for k in range(6):
        p = Pall[:,k]
        p = (p + 1 - p[::-1])/2
        ax = fig.add_axes([xpos[k], ypos[5-k], .2, .2*yratio])
        p75,pf = utils.discrimination_threshold(p, drange)
        ax.plot(drange,100*pf, color = cols[0])
        ax.scatter(drange, 100*p, color = cols[0],s=4)
        ax.plot(p75*np.array([1,1]), [-1,75], '--', color='k')
        ax.plot([-25,p75], [75,75], '--', color='k')
        ax.text(p75+5, 25, '%2.2f$^\circ$'%p75, fontweight='bold')
        ax.set_xlim(-25,25)
        ax.set_xticks([-20,20])
        ax.set_ylim(-4,103)
        ax.set_xlabel('stimulus angle ($^\circ$)')
        ax.set_ylabel('% "choose right"')
        ax.set_title(ttl[k], size=6)
        ax.text(-1,1.2, string.ascii_lowercase[k], size=12, transform=ax.transAxes)

    if save_figure:
        if not os.path.isdir(os.path.join(saveroot, 'figs')):
            os.mkdir(os.path.join(saveroot, 'figs'))
        fig.savefig(os.path.join(saveroot, 'figs/supp_discr_all.pdf'))
    return fig

def spont_sub(saveroot, save_figure=False):
    rc('font', **{'size': 6})#, 'family':'sans-serif'})#,'sans-serif':['Helvetica']})

    fig = plt.figure(figsize=(3,2),facecolor='w',frameon=True, dpi=300)
    yratio = 3/2

    d = np.load(os.path.join(saveroot, 'linear_decoder_all_stims.npy'), allow_pickle=True).item()
    E = d['E']

    d = np.load(os.path.join(saveroot, 'linear_decoder_without_spont.npy'), allow_pickle=True).item()
    Es = d['E']

    # different datasets
    inds = [np.arange(0,6,1,int), np.arange(6,9,1,int), np.arange(9,15,1,int), np.arange(15,18,1,int),
            np.arange(18,21,1,int), np.arange(21,24,1,int), np.arange(24,27,1,int), np.arange(27,30,1,int)]

    sstr = ['static','localized', 'complex', 'short', 'drifting', 'low-contrast', 'noisy','sine rand phase']
    cmap = plt.get_cmap('Dark2')
    cmap=cmap(np.linspace(0,1,len(inds)))

    ax = fig.add_axes([.2,.2,.5,.5*yratio])
    for k in range(len(inds)):
        ax.scatter(E[inds[k]], Es[inds[k]], s=3, color=cmap[k],zorder=10)
        ax.text(1,.8-k*.1,sstr[k],transform=ax.transAxes,color=cmap[k])
    ax.plot([-5,5],[-5,5],color='k',zorder=0)
    ax.set_xlabel('decoding error ($^\circ$)')
    ax.set_ylabel('decoding error spont subtracted ($^\circ$)')
    ax.set_xlim([0.7,2])
    ax.set_ylim([0.7,2])

    if save_figure:
        if not os.path.isdir(os.path.join(saveroot, 'figs')):
            os.mkdir(os.path.join(saveroot, 'figs'))
        fig.savefig(os.path.join(saveroot, 'figs/supp_spont.pdf'))
    return fig
