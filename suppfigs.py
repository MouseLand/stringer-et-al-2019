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
            np.arange(18,21,1,int), np.arange(21,24,1,int), np.arange(24,27,1,int)]

    xpos = np.linspace(0.08,.89,7)
    ypos = [.11,.48]

    for k in range(7):    
        ax = fig.add_axes([xpos[k],ypos[1], .1, .1*yratio])
        j=inds[k][0]
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
        ax_inset.imshow(IMG[k], cmap=plt.get_cmap('gray'),vmin=0,vmax=1)    
        ax_inset.axis('off')
        xs,ys=IMG[k].shape    
        ac = (0.5,0,.5)
        if k>3:        
            plt.annotate('',(ys*.75,xs*.75), (ys*.25,xs*.25), arrowprops=dict(facecolor=ac,edgecolor=ac))    
        if k==3:
            plt.text(0, -.15, '100ms',fontweight='bold',size=6, transform = ax_inset.transAxes)

        ax = fig.add_axes([xpos[k],ypos[0], .08, .08*yratio])
        bins = np.linspace(0,360,30)
        jj=0
        for j in inds[k]:
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
            ax.text(-.9,1.2, 'C', size=12, transform=ax.transAxes)
            ax.text(-.9,3.4, 'B', size=12, transform=ax.transAxes)        
            ax.text(-.9,4.6, 'A', size=12, transform=ax.transAxes)        
    
    if save_figure:
        if not os.path.isdir(os.path.join(saveroot, 'figs')):
            os.mkdir(os.path.join(saveroot, 'figs'))
        fig.savefig(os.path.join(saveroot, 'figs/supp_error_angles.pdf'))
    
    return fig

def asymptotics(saveroot, save_figure=False):
    rc('font', **{'size': 6})#, 'family':'sans-serif'})#,'sans-serif':['Helvetica']})

    fig = plt.figure(figsize=(5,2),facecolor='w',frameon=True, dpi=300)
    yratio = 5/2

    # show neurons and stims for linear
    d = np.load(os.path.join(saveroot, 'linear_decoder_asymp.npy'), allow_pickle=True).item()
    Elin = d['E']
    npoplin = d['npop']
    Estim = d['E2']
    nstim = d['nstim']
    
    # show neurons for independent
    d = np.load(os.path.join(saveroot, 'independent_decoder_asymp.npy'), allow_pickle=True).item()
    E = d['E']
    npop = d['npop']
        
    berry = [.7,.2,.5]
    grn = [0,.5,0]

    cols = [berry, grn, grn]
    for k in range(3):
        ax = fig.add_axes([0.12+k*.3, 0.21, .18, .18*yratio])
        if k==0:
            mux = npop.mean(axis=-1)
            muy = E[:,0,:].mean(axis=-1)
            semy = E[:,0,:].std(axis=-1)/(npop.shape[-1]-1)**.5
            ax.text(-.5,1.25,'Scaling with # neurons',size=8, fontstyle='italic', transform=ax.transAxes)
            ax.set_title('Independent decoder', color=cols[k])
        elif k==1:
            mux = npoplin.mean(axis=-1)
            muy = Elin[:,0,:].mean(axis=-1)
            semy = Elin[:,0,:].std(axis=-1)/(npop.shape[-1]-1)**.5
            ax.set_title('Linear decoder', color=cols[k])
        else:
            mux = nstim.mean(axis=-1)
            muy = Estim.mean(axis=-1)
            semy = Estim.std(axis=-1)/(npop.shape[-1]-1)**.5
            ax.text(-.3,1.25,'Scaling with # stimuli',size=8, fontstyle='italic', transform=ax.transAxes)
            ax.set_title('Linear decoder', color=cols[k])

        ax.semilogx(mux, muy, color=cols[k], linewidth=0.5)
        alpha,beta,r2 = utils.fit_asymptote(mux[::-1][-12:], muy[::-1][-12:])
        ax.semilogx(mux, alpha + beta / np.sqrt(mux), '--', lw=0.5, color='k')
        ax.text(.5,.6, r'$\alpha + \frac{\beta}{\sqrt{N}}$', transform=ax.transAxes,size=10)
        ax.text(.5,.4,r'$\alpha$=%2.2f$^{\circ}$'%alpha, transform=ax.transAxes)
        ax.text(.5,.3, r'$\beta$=%2.0f$^{\circ}$'%beta, transform=ax.transAxes)
        ax.set_ylim([0, 15])
        ax.set_ylabel(r'decoding error ($^\circ$)')
        if k<2:
            ax.set_xlabel('# of neurons')
        else:
            ax.set_xlabel('# of stimuli')
        ax.tick_params(axis='y')
        ax.fill_between(mux, muy-semy, muy+semy, facecolor=cols[k], alpha=0.5)
        ax.text(-0.5, 1.08, string.ascii_uppercase[k], transform=ax.transAxes, size=12);
        
    if save_figure:
        if not os.path.isdir(os.path.join(saveroot, 'figs')):
            os.mkdir(os.path.join(saveroot, 'figs'))
        fig.savefig(os.path.join(saveroot, 'figs/supp_asymp.pdf'))
    
    return fig

def stim_distances(saveroot, save_figure=False):
    rc('font', **{'size': 6})#, 'family':'sans-serif'})#,'sans-serif':['Helvetica']})

    fig = plt.figure(figsize=(6.85,2.5),facecolor='w',frameon=True, dpi=300)
    yratio = 6.85/2.5

    d = np.load(os.path.join(saveroot, 'popdist.npy'), allow_pickle=True).item()
    cbinned, dtheta_aligned, embedding, cc, istim = d['cbinned'], d['dtheta_aligned'], d['embedding'], d['cc'], d['istim']
    nc, cerr, tbins = cbinned

    isort = np.argsort(istim)
    ax=fig.add_axes([0.65,-.07,.38,.38*yratio], projection='3d')
    x = embedding[isort,0]
    y = embedding[isort,1]
    ang = np.pi*.2
    xx = x * np.cos(ang) + y * np.sin(ang)
    yy = -x * np.sin(ang) + y * np.cos(ang)
    ax.scatter(xx,yy, embedding[isort,2], 
               c=istim[isort], s=1, cmap=plt.get_cmap('twilight_shifted'),alpha=1, depthshade=True)
    cmap = plt.get_cmap('twilight_shifted')
    cmap = cmap(np.linspace(0,1,int(isort.size/2)))

    ir = 400
    #for i in range(int(isort.size/2)):
    #    ax.plot([model.embedding_[isort[i*2],0], model.embedding_[isort[i*2+1],0]],
    #            [model.embedding_[isort[i*2],1], model.embedding_[isort[i*2+1],1]],
    #            [model.embedding_[isort[i*2],2], model.embedding_[isort[i*2+1],2]], color=cmap[i], lw=0.5)
               #c=istim[isort], s=0.4, cmap=,alpha=.5, depthshade=True)
    ax.scatter(embedding[::2,0], embedding[::2,1], -ir, 
               s=.4, color=(0.5,.5,.5),alpha=.4)
    ax.scatter(-ir, embedding[::2,1], embedding[::2,2], 
               s=1, color=(0.5,.5,.5),alpha=.4)
    ax.scatter(embedding[::2,0], -ir, embedding[::2,2], 
               s=1, color=(0.5,.5,.5),alpha=.4)
    ax.set_title('ISOMAP embedding of stimuli                   ')

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


    ax=fig.add_axes([0.03,.2,.25,.25*yratio])
    im=ax.imshow(cc[np.ix_(isort,isort)], vmin=-.1, vmax=.3)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title('    Correlation between neural patterns')
    ax.text(-.1, 1.05, 'A', size=12, transform=ax.transAxes)
    ax.text(1.3, 1.05, 'B', size=12, transform=ax.transAxes)
    ax.text(2.53, 1.05, 'C', size=12, transform=ax.transAxes)
    ax.set_xlabel('trials sorted by angle')
    ax.set_ylabel('trials sorted by angle')
    ax2=fig.add_axes([.29, .1+.25*yratio, .01,.1])
    cb=plt.colorbar(im, ax2, ticks=[0,.3])

    ax=fig.add_axes([0.4,.2,.25,.25*yratio])
    #for n in np.random.randint(0, NS-1, 10):
    ax.scatter((dtheta_aligned*180/np.pi).flatten()[::500], cc.flatten()[::500],s=1,color=(.5,.5,.5), alpha=.05)

    ax.plot(tbins, nc, color='k',lw=0.5)
    ax.fill_between(tbins, nc-cerr, nc+cerr, color='k', alpha=0.5)
    ax.set_xticks([0,90,180,270,360])
    ax.set_xticklabels(['-90','0','90','180','270'])
    ax.set_xlabel('stimulus angle difference ($^\circ$)')
    ax.set_ylabel('correlation between neural patterns')
    ax.set_ylim(-.1,.3)
    ax.set_yticks(np.arange(-.1,.4,.1))
    
    if save_figure:
        if not os.path.isdir(os.path.join(saveroot, 'figs')):
            os.mkdir(os.path.join(saveroot, 'figs'))
        fig.savefig(os.path.join(saveroot, 'figs/supp_mani.pdf'))
    
    return fig

def pc_errors(saveroot, save_figure=False):
    rc('font', **{'size': 6})#, 'family':'sans-serif'})#,'sans-serif':['Helvetica']})

    fig = plt.figure(figsize=(5,2),facecolor='w',frameon=True, dpi=300)
    yratio = 5/2
    bz = .3

    d = np.load(os.path.join(saveroot, 'pcdecode.npy'), allow_pickle=True).item()
    errors, atrues, apreds, nPC = d['errors'], d['atrues'], d['apreds'], d['nPC']

    dlin = np.load(os.path.join(saveroot, 'linear_decoder_asymp.npy'), allow_pickle=True).item()

    cmap = plt.get_cmap('viridis')
    cmap = cmap(np.linspace(0,1,len(nPC)))

    ax = fig.add_axes([.1,.2,bz,bz*yratio])
    for k in range(len(nPC)):
        ax.scatter(atrues[k], apreds[k], color=cmap[k], s=0.1, alpha=0.5)
        ax.text(1.25,.02+k*.08, '%d'%nPC[k], transform=ax.transAxes, color=cmap[k], ha='right')
    ax.text(1.25,.02+(k+1)*.08, '# of PCs', transform=ax.transAxes, color='k', ha='right')
    ax.set_xlabel('true angle ($^\circ$)')
    ax.set_ylabel('decoded angle ($^\circ$)')
    ax.text(-.3,.95,'A',size=12, transform=ax.transAxes)

    ax = fig.add_axes([.6,.2,bz,bz*yratio])
    for k in range(len(nPC)):
        ax.errorbar(nPC[k], errors[:,k].mean(axis=0), 
                    errors[:,k].std(axis=0)/np.sqrt(6), markersize=20,
                    color=cmap[k], zorder=5)
        ax.scatter(nPC[k], errors[:,k].mean(axis=0), s=3,
                 color=cmap[k], zorder=10)
    ax.semilogx(nPC, errors.mean(axis=0), color=(0.5,.5,.5))
    ax.semilogx(nPC, dlin['E'][0,0,:].mean(axis=-1) * np.ones(nPC.shape),'--', color=(0.5,.5,.5))

    ax.set_xlabel('number of PCs')
    ax.set_ylabel('decoding error ($^\circ$)')
    ax.set_xticks(nPC[::2])
    ax.set_xticklabels(['4', '16','64', '256', '1024', '4096'])
    ax.set_xlim(nPC[0]-.5,nPC[-1]+1300)
    ax.set_ylim(0,8)
    ax.text(-.3,.95,'B',size=12, transform=ax.transAxes)

    fig.savefig(os.path.join(saveroot, 'figs/supp_PC_error.pdf'))
    
    if save_figure:
        if not os.path.isdir(os.path.join(saveroot, 'figs')):
            os.mkdir(os.path.join(saveroot, 'figs'))
        fig.savefig(os.path.join(saveroot, 'figs/supp_mani.pdf'))
    
    return fig

def population_curves(saveroot, save_figure=False):
    rc('font', **{'size': 6})#, 'family':'sans-serif'})#,'sans-serif':['Helvetica']})

    fig = plt.figure(figsize=(6.85,3.43),facecolor='w',frameon=True, dpi=300)
    yratio = 2
    bz = .13

    static = np.load(os.path.join(saveroot, 'avgneur_static.npy')).item()
    drifting = np.load(os.path.join(saveroot, 'avgneur_drifting.npy')).item()
    theta_pref = np.linspace(0, 2*np.pi, 17)[:-1]

    xpos = np.linspace(0.1, 1, 5)
    ypos = [.16, .68]

    for i in range(2):
        if i == 0:
            d = static.copy()
        else:
            d = drifting.copy()

        ax = fig.add_axes([xpos[0], ypos[1-i], bz, bz*yratio])
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
        ax.text(-.4,1.05,string.ascii_uppercase[i*4],size=12,transform=ax.transAxes)

        xx,yy = np.meshgrid(np.arange(0,100)/8, np.arange(0,100)/8)
        for j in range(5):
            ax = fig.add_axes([xpos[0]-.01+ j*.031, ypos[1-i]-.16, 0.025, 0.025*yratio])
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
            ax = fig.add_axes([xpos[j+1], ypos[1-i], bz, bz*yratio])
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
            ax.text(-.4,1.05,string.ascii_uppercase[j+i*4+1],size=12,transform=ax.transAxes)

    if save_figure:
        if not os.path.isdir(os.path.join(saveroot, 'figs')):
            os.mkdir(os.path.join(saveroot, 'figs'))
        fig.savefig(os.path.join(saveroot, 'figs/supp_popcurves.pdf'))
    
    return fig

def discr_nn_rf(saveroot, save_figure=False):
    rc('font', **{'size': 6})#, 'family':'sans-serif'})#,'sans-serif':['Helvetica']})

    fig = plt.figure(figsize=(5,2),facecolor='w',frameon=True, dpi=300)
    yratio = 5/2

    nn = np.load(os.path.join(saveroot, 'nn_discrimination.npy'), allow_pickle=True).item()
    rf = np.load(os.path.join(saveroot, 'rf_discrimination.npy'), allow_pickle=True).item()
    drange = rf['drange']

    ttl = ['Neural network', 'Random forest']
    cols=[[0,.5,0], [0,.5,0]]
    for k in range(2):
        if k==0:
            P0 = nn['P']
        else:
            P0 = rf['P']
        p = P0.mean(axis=0)
        p = (p + 1 - p[::-1])/2
        ax = fig.add_axes([.12+k*.5, .18, .25, .25*yratio])
        p75 = utils.discrimination_threshold(p, drange)[0]
        ax.plot(drange,100*p, color = cols[k])
        ax.scatter(drange, 100*p, color = cols[k],s=10)
        ax.plot(p75*np.array([1,1]), [-1,75], '--', color='k')
        ax.plot([-25,p75], [75,75], '--', color='k')
        ax.text(p75+5, 25, '%2.2f$^\circ$'%p75, fontweight='bold')    
        ax.set_xlim(-25,25)
        ax.set_xticks([-20,20])    
        ax.set_ylim(-4,103)  
        ax.set_xlabel('stimulus angle ($^\circ$)')
        ax.set_ylabel('% "choose right"')
        ax.set_title(ttl[k])
        ax.text(-.4,1, string.ascii_uppercase[k], size=12, transform=ax.transAxes)

    if save_figure:
        if not os.path.isdir(os.path.join(saveroot, 'figs')):
            os.mkdir(os.path.join(saveroot, 'figs'))
        fig.savefig(os.path.join(saveroot, 'figs/supp_nn.pdf'))    
    return fig

def discr_layers(saveroot, save_figure=False):
    rc('font', **{'size': 6})#, 'family':'sans-serif'})#,'sans-serif':['Helvetica']})

    fig = plt.figure(figsize=(5,2),facecolor='w',frameon=True, dpi=300)
    yratio = 5/2

    d = np.load(os.path.join(saveroot, 'layer_discrimination.npy'), allow_pickle=True).item()
    drange = d['drange']
    P0 = d['P']

    ttl = ['Layers 2/3', 'Layer 4']
    cols=[[0,.5,0], [0,.5,0]]
    for k in range(2):
        p = P0.mean(axis=0)[:,k]
        p = (p + 1 - p[::-1])/2
        ax = fig.add_axes([.12+k*.5, .18, .25, .25*yratio])
        p75 = utils.discrimination_threshold(p, drange)[0]
        ax.plot(drange,100*p, color = cols[k])
        ax.scatter(drange, 100*p, color = cols[k],s=10)
        ax.plot(p75*np.array([1,1]), [-1,75], '--', color='k')
        ax.plot([-25,p75], [75,75], '--', color='k')
        ax.text(p75+5, 25, '%2.2f$^\circ$'%p75, fontweight='bold')    
        ax.set_xlim(-25,25)
        ax.set_xticks([-20,20])    
        ax.set_ylim(-4,103)  
        ax.set_xlabel('stimulus angle ($^\circ$)')
        ax.set_ylabel('% "choose right"')
        ax.set_title(ttl[k])
        ax.text(-.4,1, string.ascii_uppercase[k], size=12, transform=ax.transAxes)

    if save_figure:
        if not os.path.isdir(os.path.join(saveroot, 'figs')):
            os.mkdir(os.path.join(saveroot, 'figs'))
        fig.savefig(os.path.join(saveroot, 'figs/supp_layers.pdf'))    
    return fig

def discr_chron(saveroot, save_figure=False):
    rc('font', **{'size': 6})#, 'family':'sans-serif'})#,'sans-serif':['Helvetica']})

    fig = plt.figure(figsize=(5,2),facecolor='w',frameon=True, dpi=300)
    yratio = 5/2

    d = np.load(os.path.join(saveroot, 'chron_discrimination.npy'), allow_pickle=True).item()
    drange = d['drange']
    P0 = d['P']

    ttl = ['Original', 'Chronological train/test']
    cols=[[0,.5,0], [0,.5,0]]
    for k in range(2):
        p = P0.mean(axis=0)[:,k]
        p = (p + 1 - p[::-1])/2
        ax = fig.add_axes([.12+k*.5, .18, .25, .25*yratio])
        p75 = utils.discrimination_threshold(p, drange)[0]
        ax.plot(drange,100*p, color = cols[k])
        ax.scatter(drange, 100*p, color = cols[k],s=10)
        ax.plot(p75*np.array([1,1]), [-1,75], '--', color='k')
        ax.plot([-25,p75], [75,75], '--', color='k')
        ax.text(p75+5, 25, '%2.2f$^\circ$'%p75, fontweight='bold')    
        ax.set_xlim(-25,25)
        ax.set_xticks([-20,20])    
        ax.set_ylim(-4,103)  
        ax.set_xlabel('stimulus angle ($^\circ$)')
        ax.set_ylabel('% "choose right"')
        ax.set_title(ttl[k])
        ax.text(-.4,1, string.ascii_uppercase[k], size=12, transform=ax.transAxes)

    if save_figure:
        if not os.path.isdir(os.path.join(saveroot, 'figs')):
            os.mkdir(os.path.join(saveroot, 'figs'))
        fig.savefig(os.path.join(saveroot, 'figs/supp_layers.pdf'))    
    return fig

def spont_sub(saveroot, save_figure=False):
    rc('font', **{'size': 6})#, 'family':'sans-serif'})#,'sans-serif':['Helvetica']})

    fig = plt.figure(figsize=(3,2),facecolor='w',frameon=True, dpi=300)
    yratio = 3/2

    d = np.load(os.path.join(saveroot, 'linear_decoder_all_stims.npy'), allow_pickle=True).item()
    E = d['E']

    d = np.load(os.path.join(saveroot, 'linear_decoder_spont_sub.npy'), allow_pickle=True).item()
    Es = d['E']

    # different datasets
    inds = [np.arange(0,6,1,int), np.arange(6,9,1,int), np.arange(9,15,1,int), np.arange(15,18,1,int),
            np.arange(18,21,1,int), np.arange(21,24,1,int), np.arange(24,27,1,int)]

    sstr = ['static','localized', 'complex', 'short', 'drifting', 'low-contrast', 'noisy']
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

    

    

