import numpy as np
from matplotlib import pyplot as plt
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import string
from matplotlib import rc, rcParams
from matplotlib.colors import hsv_to_rgb
from scipy.stats import spearmanr,zscore
from scipy.ndimage import gaussian_filter1d
import mainfigs, decoders, utils, tuning
from PIL import Image

rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False


def visual_stimuli(dataroot, nsc=5):
    IMG = []
    for k in range(7):    
        if k==1:
            xx,yy = np.meshgrid(np.arange(0,640*nsc)/(60*nsc), np.arange(0,480*nsc)/(60*nsc))
            gratings = np.cos(xx*np.cos(np.pi/4+.1) + yy*np.sin(np.pi/4+.1))
            gratings[gratings<0]=0
            gratings[gratings>0]=1
            xcent = gratings.shape[1]*.75
            ycent = gratings.shape[0]/2
            xxc,yyc = np.meshgrid(np.arange(0,gratings.shape[1]), np.arange(0,gratings.shape[0]))
            icirc = ((xxc-xcent)**2 + (yyc-ycent)**2)**0.5 < 640/3/2*nsc
            gratings[~icirc] = 0.5
            IMG.append(gratings)
            #img = plt.imread(os.path.join(figroot, imgs[k]))[:,:,0]
        elif k==2:
            minnie=plt.imread(os.path.join(dataroot, 'minnie.png'))[:,:,0]
            minnie
            img = np.ones((480,640)) * 0.5
            xcent = img.shape[1]*.75
            ycent = img.shape[0]/2
            ix = int(ycent-minnie.shape[0]/2) + np.arange(0,minnie.shape[0],1,int)
            iy = int(xcent-minnie.shape[1]/2) + np.arange(0,minnie.shape[1],1,int)
            img[np.ix_(ix,iy)] = minnie
            IMG.append(img)
        elif k==0 or k==3 or k==4 :
            xx,yy = np.meshgrid(np.arange(0,640*nsc)/(28*nsc), np.arange(0,480*nsc)/(28*nsc))
            gratings = np.cos(xx*np.cos(np.pi/4) + yy*np.sin(np.pi/4))
            gratings[gratings<0]=0
            gratings[gratings>0]=1
            img = gratings
            IMG.append(img)
        elif k==5:
            xx,yy = np.meshgrid(np.arange(0,640*nsc)/(28*nsc), np.arange(0,480*nsc)/(28*nsc))
            gratings = np.cos(xx*np.cos(np.pi/4) + yy*np.sin(np.pi/4))
            gratings[gratings>0]=.52
            gratings[gratings<0]=.48
            img = gratings
            IMG.append(img)
        elif k==6:
            xx,yy = np.meshgrid(np.arange(0,640*nsc)/(28*nsc), np.arange(0,480*nsc)/(28*nsc))
            gratings = np.cos(xx*np.cos(np.pi/4) + yy*np.sin(np.pi/4))
            gratings[gratings>0]=.52
            gratings[gratings<0]=.48
            gratings += .25*np.random.randn(img.shape[0],img.shape[1])
            img = gratings      
            IMG.append(img)
            
    return IMG

def draw_neural_net(ax, left, right, bottom, top, layer_sizes, colors):
    ''' modified from @craffel (thanks!) (https://gist.github.com/craffel/2d727968c3aaebd10359)
    Draw a neural network cartoon using matplotilb.
    
    :usage:
        >>> fig = plt.figure(figsize=(12, 12))
        >>> draw_neural_net(fig.gca(), .1, .9, .1, .9, [4, 7, 2])
    
    :parameters:
        - ax : matplotlib.axes.AxesSubplot
            The axes on which to plot the cartoon (get e.g. by plt.gca())
        - left : float
            The center of the leftmost node(s) will be placed here
        - right : float
            The center of the rightmost node(s) will be placed here
        - bottom : float
            The center of the bottommost node(s) will be placed here
        - top : float
            The center of the topmost node(s) will be placed here
        - layer_sizes : list of int
            List of layer sizes, including input and output dimensionality
    '''
    n_layers = len(layer_sizes)
    v_spacing = []
    for l in range(n_layers):
        v_spacing.append((top - bottom)/float(layer_sizes[l]))
    v_spacing = np.array(v_spacing)
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing[n]*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing[n]), v_spacing.max()/6.,
                                color=colors[n][m], ec=colors[n][m], zorder=4)
            ax.add_artist(circle)
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing[n]*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing[n+1]*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing[n], layer_top_b - o*v_spacing[n+1]], 
                                  c=(0.5,0.5,0.5), linewidth=0.5)
                ax.add_artist(line)
                
def fig1(dataroot, saveroot, save_figure=False):
    rc('font', **{'size': 6})#, 'family':'sans-serif'})#,'sans-serif':['Helvetica']})

    # load spks
    dat=np.load(os.path.join(dataroot, 'spks_gratings_static_TX40_2019_05_02_1.npy')).item()
    stimtimes,stat,ops = dat['stimtimes'],dat['stat'],dat['ops_plane6']
    sresp, istim, itrain, itest = utils.compile_resp(dat)
    
    nbase = 10
    A, B, D, rez         = decoders.fit_indep_model(sresp[:, itrain], istim[itrain], nbase)
    apred1, logL, B2, Kup = decoders.test_indep_model(sresp[:, itest], A, nbase)
    Apred = A.T @ B2
    SNR = np.var(Apred, axis=1) / np.var(rez, axis=1)
    btheta = np.argmax(Apred @ Kup.T, axis=1) / Kup.shape[0] * 2 * np.pi

    dtheta = np.pi/180 * 10
    theta0 = np.pi/4 + np.pi
    #ix = (np.logical_and(btheta>theta0, btheta<theta0 + dtheta)).nonzero()[0].astype('int')
    #isort = np.argsort(SNR[ix])[::-1]

    # subtract off spont PCs from full data
    trange = [1430, 1850]
    spks_norm = (dat['spks'] - dat['mean_spont'][:,np.newaxis]) / dat['std_spont'][:,np.newaxis]
    #sspont = spks.T @ dat['u_spont']
    #spks_norm = spks_norm - dat['u_spont'] @ (dat['u_spont'].T @ spks_norm)
    stimtimes = dat['stimtimes']

    stimtrace = np.zeros((dat['spks'].shape[1],), np.bool)
    stimtrace[stimtimes] = True
    stimtrace = stimtrace[trange[0]:trange[-1]]

    isort = np.argsort(btheta)
    
    dsmooth =zscore(gaussian_filter1d((spks_norm[isort,trange[0]:trange[-1]]),50,axis=0), axis=1) # 1430:1850
    
    fig = plt.figure(figsize=(6.85,4.5),facecolor='w',frameon=True, dpi=300)
    yratio = 6.85/4.5

    mimg = np.zeros((ops['Ly'], ops['Lx']))
    mimg[ops['yrange'][0]:ops['yrange'][-1], ops['xrange'][0]:ops['xrange'][-1]] = ops['max_proj']
    #mimg = ops['meanImg']
    mimg = mimg[80:130, 10:-10]

    NN = len(stat)
    masks = np.zeros((ops['Ly'], ops['Lx'], 3))
    LamAll = np.zeros((ops['Ly'], ops['Lx']))
    iplane=np.zeros((NN,),np.int32)
    ipl = 6
    Lx = ops['Lx']
    Ly = ops['Ly']
    nX = np.ceil(np.sqrt(ops['Ly'] * ops['Lx'] * ops['nplanes'])/ops['Lx'])
    nX = int(nX)
    nY = int(np.ceil(ops['nplanes']/nX))
    dx = (ipl%nX) * Lx
    dy = int(ipl/nX) * Ly

    for n in range(NN):
        iplane[n] = stat[n]['iplane']
        if iplane[n]==ipl:
            ypix,xpix,lam = stat[n]['ypix']-dy,stat[n]['xpix']-dx,stat[n]['lam']
            lam /= lam.sum()
            LamAll[ypix,xpix] = lam

    nnp = (iplane==ipl).sum()
    cols = np.random.rand(nnp)
    LamMean = LamAll[LamAll>1e-10].mean()
    for k,n in enumerate((iplane==ipl).nonzero()[0]):
        ypix,xpix,lam = stat[n]['ypix']-dy,stat[n]['xpix']-dx,stat[n]['lam']
        lam /= lam.sum()
        V = np.maximum(0, np.minimum(1.0, 0.75*lam/LamMean))
        masks[ypix,xpix,0] = cols[k]
        masks[ypix,xpix,1] = 1.0
        masks[ypix,xpix,2] = V

    masks = hsv_to_rgb(masks)
    masks = masks[80:130, 10:-10]

    fig.tight_layout()
    plt.subplots_adjust(left=.05, bottom=.05, right=0.95, top=0.95, wspace=None, hspace=None)

    img=Image.open(os.path.join(dataroot, 'planes_meso.png'))
    ax=fig.add_axes([.02,.63,.42,.42])
    imgplot=ax.imshow(img)
    imgplot.set_interpolation('bicubic')
    ax.axis('off')
    ax.text(-0.03, 1.0, string.ascii_uppercase[0], transform=ax.transAxes, size=12)
    ax.text(0.45, 1.0, string.ascii_uppercase[1], transform=ax.transAxes, size=12)

    ax=fig.add_axes([.5,.77,.5,.31])
    ax.imshow(mimg, cmap=plt.get_cmap('gray'),vmin=1000,vmax=6000, aspect=1.5)
    #ax.set_title('mean image', fontsize=12)
    ax.text(0.01, 0.73, 'Maximum fluorescence image', color='k', transform=ax.transAxes)
    ax.axis('off')
    ax.text(-0.05, .73
            , string.ascii_uppercase[2], transform=ax.transAxes, size=12)
    plt.plot([masks.shape[1],masks.shape[1]-75],[-3,-3],color='k')
    ax.set_xlim(0,masks.shape[1])
    ax.set_ylim(-4,75)
    ax.text(424,-12,r'100 $\mu$m')

    ax = fig.add_axes([.5,.61,.5,.31])
    ax.imshow(masks,aspect=1.5)
    ax.text(0.01, .68, 'Masks from suite2p', color='k', transform=ax.transAxes)
            #fontweight='bold')
    ax.axis('off')
    ax.set_xlim(0,masks.shape[1])
    ax.set_ylim(0,78)


    ax = fig.add_axes([0.04,.31,.45,.3])
    nt = dsmooth.shape[1]
    ax.imshow(dsmooth[:,:], cmap=plt.get_cmap('gray'),vmin=-.3, vmax=6, aspect='auto')
    ax.text(-.05,.5, 'neurons sorted by pref angle', verticalalignment='center', transform=ax.transAxes,rotation=90)
    ax.text(1.01,0.4, '5,000 neurons',transform=ax.transAxes,rotation=270)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.plot((nt-1+7)*np.array([1,1]), [NN-1,NN-5000], color='k', linewidth=4)
    ax.set_xlim(0,nt+8)
    ax.axis('off')
    ax.text(-0.08, 1.01, string.ascii_uppercase[3], transform=ax.transAxes, size=12)

    ax = fig.add_axes([.04,.25,.45,.05])
    scol = (0.7,0.6,.7)
    ax.bar(x=np.arange(nt),height=stimtrace, width=3, color=scol)
    ax.plot([0,nt-1],[0,0],color=scol)
    ax.plot([0,10/0.3],np.array([1,1])*-0.1,color='k')
    ax.text(0,-.5,'10 sec')
    ax.text(nt-140,-0.5,'stimulus times',color=scol)
    ax.axis('off')
    ax.set_xlim(0,nt+8)

    dtheta = np.pi/180 * 10
    theta0 = np.pi/4 + np.pi
    ix = (np.logical_and(btheta>theta0, btheta<theta0 + dtheta)).nonzero()[0].astype('int')
    ixsort = np.argsort(SNR[ix])[::-1]
    iex = ix[ixsort[5]]#, ix[ixsort[150]]]
    istimtest = istim[:-2][itest[:-2]]
    issort = np.argsort(istimtest)
    thpref=btheta[iex]
    rneur = spks_norm[iex]
    istimtimes = stimtimes[:-2] + np.arange(-4,10,1,int)[:,np.newaxis]
    rresp = rneur[istimtimes]
    rresp = rresp[:,itest[:-2]]
    idist = np.abs(istimtest-thpref)
    iss = (idist < .2).nonzero()[0]
    idsort = np.argsort(idist[iss])
    rresp = rresp[:,iss[idsort]]
    istimrange=istimtest[iss[idsort]]*180/np.pi

    ax=fig.add_axes([.62,.31,.15,.3])
    im=ax.imshow(rresp.T, aspect='auto', extent=(-4*.33, 10*.33, istimrange[0],istimrange[-1]),
              cmap=plt.get_cmap('gray'), vmin=-.3,vmax=6)
    ax.text(0,1.05,r'example neuron #%d'%iex, transform=ax.transAxes)#, (thpref-.2)*180/np.pi, (thpref+.2)*180/np.pi), fontsize=8)
    ax.set_xlabel('time from stim (s)')
    ax.set_ylabel('stimulus angles')
    ax.text(-0.4, 1.01, string.ascii_uppercase[4], transform=ax.transAxes, size=12)

    axi = fig.add_axes([.49,.61-.1,.01,.1])
    plt.colorbar(im,axi)
    axi.set_ylabel('  z-score', rotation=270)

    ax = fig.add_axes([.88,.52,.09,.11])
    ax.scatter(istim[itest]/(np.pi)*180, sresp[iex,itest],color=(0.5,.5,.5), s=0.5, alpha=0.1)
    ypred = Apred[iex]  @ Kup.T
    iori = np.linspace(0, 360, ypred.size)
    ax.plot(iori, ypred, color='k', linewidth=0.5)
    ax.text(200,7.6,'neuron #%d\nSNR = %2.2f'%(iex, SNR[iex]), horizontalalignment='center')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(-3,13-3)
    ax.set_xticks([0, 180, 360])  
    ax.set_xlabel('stimulus angle ($^\circ$)')
    ax.set_ylabel('response\n(z-scored)')
    ax.text(-1, .88, string.ascii_uppercase[5], transform=ax.transAxes, size=12)
    #ax.axis('square')

    ax = fig.add_axes([.88,.31,.09,.11])
    nb=plt.hist(SNR,100, color=(0.5,.5,.5))
    merror = np.mean(SNR)
    ax.scatter(merror, nb[0].max()*.9, marker='v',color='k')
    plt.text(merror+.1, nb[0].max()*1.05, '%2.2f'%merror, 
             horizontalalignment='center',fontsize=6,fontweight='bold')
    ax.set_xlim([0,1.])
    ax.set_xlabel('SNR')
    ax.set_ylabel('counts')
    #ax.set_yticklabels(['0','2','4','6'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(.12,3000,r'SNR = $\frac{var(signal)}{var(noise)}$')
    ax.text(-1, .95, string.ascii_uppercase[6], transform=ax.transAxes, size=12)


    img=Image.open(os.path.join(dataroot,'hypotheses.png'))
    ax=fig.add_axes([0.04,0.01,.92,.92*np.asarray(img).shape[0]/np.asarray(img).shape[1] * yratio])
    imgplot=ax.imshow(img)
    imgplot.set_interpolation('bicubic')
    ax.axis('off')
    ax.text(-0.04, 1.08, string.ascii_uppercase[7], transform=ax.transAxes, size=12)
    ax.text(-0.01, 1.08, 'Coordination of decoding errors between neurons (hypotheses)', transform=ax.transAxes, size=8)
    
    
    if save_figure:
        if not os.path.isdir(os.path.join(saveroot, 'figs')):
            os.mkdir(os.path.join(saveroot, 'figs'))
        fig.savefig(os.path.join(saveroot, 'figs/fig1.pdf'))
    
    
    return fig

                
def fig2(dataroot, saveroot, save_figure=False):
    rc('font', **{'size': 6})#, 'family':'sans-serif'})#,'sans-serif':['Helvetica']})

    #### EXAMPLE DATASET + POOLED DATA
    dat = np.load(os.path.join(dataroot, 'gratings_drifting_GT1_2019_04_12_1.npy'), allow_pickle=True).item()
    sresp, istim, itrain, itest = utils.compile_resp(dat, npc=32)

    d = np.load(os.path.join(saveroot, 'linear_decoder_asymp.npy'), allow_pickle=True).item()
    Elin = d['E']
    npoplin = d['npop']
    Estim = d['E2']
    nstim = d['nstim']
    d = np.load(os.path.join(saveroot, 'independent_decoder_asymp.npy'), allow_pickle=True).item()
    E = d['E']
    ccE = d['ccE']

    ypos = np.array([dat['stat'][j]['med'][0] for j in range(len(dat['stat']))])
    # split neurons for decoder into strips (no Z overlap between two sets)
    nstrips = 8
    ypos = np.array([dat['stat'][j]['med'][0] for j in range(len(dat['stat']))])
    n1, n2 = utils.stripe_split(ypos, 8)
    nangle = 2*np.pi

    ssi = itrain
    apred1, err1, ypred1, logL1, SNR, theta_pref = decoders.independent_decoder(sresp[n1,:], istim, itrain, itest)
    apred2, err2, ypred2, logL2, SNR, theta_pref = decoders.independent_decoder(sresp[n2,:], istim, itrain, itest)
    #A, B, D, rez        =  decoders.fit_indep_model(sresp[np.ix_(n1, ssi)], istim[ssi], nbase)
    #apred1, logL1, B2, Kup = decoders.test_indep_model(sresp[np.ix_(n1,itest)], A, nbase)

    nbase=10
    Apred, error, ypred, logL, SNR, theta_pref = decoders.independent_decoder(sresp, istim, itrain, itest)
    A, B, D, rez = decoders.fit_indep_model(sresp[:, itrain], istim[itrain], nbase)
    apred, logL, B2, Kup = decoders.test_indep_model(sresp[:, itest], A, nbase)
    print(np.median(np.abs(error)) * 180/np.pi)
    Apred = A.T @ B2
    RS = spearmanr(err1, err2)
    btheta = theta_pref #np.argmax(Apred @ Kup.T, axis=1) / Kup.shape[0] * 2 * np.pi

    # example neurons
    dtheta = np.pi/180 * 10
    theta0 = np.pi/4 + np.pi
    ix = (np.logical_and(btheta>theta0, btheta<theta0 + dtheta)).nonzero()[0].astype('int')
    isort = np.argsort(SNR[ix])[::-1]
    iN = np.zeros(5, 'int32')
    iN[0] = ix[isort][1]
    iN[1] = ix[isort][int(isort.size/2)+5]
    iN[2] = ix[isort][97]#97
    theta1 = -np.pi/4 + np.pi
    ix = (np.logical_and(btheta>theta1, btheta<theta1 + dtheta)).nonzero()[0]
    isort = np.argsort(SNR[ix])[::-1]
    iN[3] = ix[isort][22]
    iN[4] = ix[isort][50]

    itest_trial = np.argsort(np.abs(istim[itest]- theta0))[37] # 53
    itest = np.arange(0, istim.size, 4, int)
    ind_trial = itest[itest_trial]
    ind_trial = itest[itest_trial]
    ind_trial = 888
    itest_trial = (itest==ind_trial).nonzero()[0][0]
    print(itest_trial)
    
    nrez = -(sresp[:, ind_trial][:, np.newaxis] - Apred)**2 
    print(nrez.shape)
    nodes = 32
    Kup = utils.upsampling_mat(nodes, int(3200/nodes), nodes/32)
    logup = nrez @ Kup.T

    NN = sresp.shape[0]

    apredLin, errorLin, ypredLin, _ = decoders.vonmises_decoder(sresp, istim, itrain, itest)

    fig = plt.figure(figsize=(6.85,5),facecolor='w',frameon=True, dpi=300)
    yratio = 6.85/5

    isort = np.argsort(istim[itrain])
    NN = sresp.shape[0]
    ncol = 6
    dy = .9 / 6
    bzx = 0.1
    bzy = 0.08
    lrange = [-8,1]
    larange= [-0.8, -.3]

    berry = [.7,.2,.5]
    grn = [0,.5,0]

    col0 = [.3, 0, .4]
    col1 =  [.2, 0, .3]
    col2 = [.5, 0.3, .6]

    theta0=istim[ind_trial]

    for k in range(3):
        ax = fig.add_axes([.08, .11+(5-k)*dy, bzx,bzy])
        istimtest = istim[itest]
        istimsort = np.argsort(istim[itest])
        ax.scatter(istim*180/np.pi, sresp[iN[k], :], s=.5, color=(0.5,0.5,0.5), alpha=0.1)
        ypredNeur = A[:, iN[k]]  @ B
        ax.plot(istim[itrain[isort]]*180/np.pi, ypredNeur[isort], color='k', lw=0.5)    
        ax.scatter(istim[ind_trial]*180/np.pi, sresp[iN[k], ind_trial], marker='x', s=20, color=col0)#facecolors='none',
        ax.set_ylim(-3,13-3)
        ax.set_xticks([0, 180, 360])    
        if k==2:
            ax.set_xlabel('stimulus angle ($^\circ$)')
        plt.text(10, 10, 'SNR = %2.2f'%(SNR[iN[k]]),size=6)
        if k==0:
            ax.set_ylabel('response\n(z-score)')    
            ax.text(-.3,1.35,'Independent decoder',size=8, transform=ax.transAxes, color=berry)
            ax.text(-0.6, 1.35, string.ascii_uppercase[0], transform=ax.transAxes, size=12)

        plt.annotate('',[360,4],[540,4], arrowprops=dict(arrowstyle= '<|-',facecolor='black',lw=1))
        ax = fig.add_axes([.28, .11+(5-k)*dy, bzx, bzy])
        plt.plot(np.linspace(0, 360, logup.shape[1]+1)[:-1], logup[iN[k], :], color = col0)
        ax.set_ylim(lrange[0],lrange[-1])
        if k==0:
            ax.set_ylabel('log-likelihood')
            ax.set_title('trial #%d (test)'%ind_trial,size=6)
            ax.text(0.8,0.6,'true \n= %2.0f$^\circ$'%(istim[ind_trial]*180/np.pi), transform=ax.transAxes)

        ax.set_xticks([0, 180, 360])
        ax.plot(np.array([1,1]) * 180/np.pi * istim[ind_trial], np.array([-8, 1]), '--', color=(0.5,.5,.5), linewidth=1)

    ax = fig.add_axes([.28, .11+2*dy, bzx, bzy])
    plt.plot(np.linspace(0, 360, logup.shape[1]+1)[:-1], logup.mean(axis=0), color = col0, linewidth = 2)
    ax.set_ylim(larange[0],larange[1])
    ax.set_xticks([0, 180, 360])
    ax.set_yticks([-.8,-.4])
    ax.set_ylabel('average logL')
    ax.set_xlabel('angle ($^\circ$)')
    ax.plot(np.array([1,1]) * 180/np.pi * istim[ind_trial], larange, '--', color=(0.5,.5,.5), linewidth=1)
    logmax = np.argmax(logup.mean(axis=0))
    ax.scatter(np.linspace(0, 360, logup.shape[1]+1)[:-1][logmax],
               logup.mean(axis=0)[logmax], marker='*',color=berry,s=40,zorder=10)
    ax.text(0.7,0.6,'decoded \n= %2.0f$^\circ$'%(apred[itest_trial]*180/np.pi),color=berry, size=6, transform=ax.transAxes)
    ax.text(0.5,1.1, '=', size=14, transform=ax.transAxes, ha='center',fontweight='bold')
    ax.text(.5,2.9, '+',size=14,transform=ax.transAxes,fontweight='bold', ha='center')
    ax.text(.5,4.8, '+',size=14,transform=ax.transAxes,fontweight='bold', ha='center')
    #ax.text(1.2,4.2, '+',size=10,transform=ax.transAxes)
    #ax.text(1.2,6.1, '+',size=10,transform=ax.transAxes)
    ##### SUBPLOT SETTINGS
    xpos = np.linspace(0.08, .78, 4)
    ypos = np.linspace(.08, .75, 3)[::-1]
    bz = .15
    xpos[2]-=0.04

    ax = fig.add_axes([xpos[2], ypos[0], bz, bz*yratio])
    ax.scatter(istim[itest]* 180/np.pi, apred * 180/np.pi, marker='.', alpha=0.5, 
             s=.1, color = berry)#, alpha=0.1)
    #ax.plot(istim[itest]* 180/np.pi, apredLin * 180/np.pi, marker=',', lw=0, color=berry)
    ax.set_xlabel(r'true angle ($^\circ$)')
    ax.set_ylabel(r'decoded angle ($^\circ$)')
    ax.set_xticks([0, 180, 360])
    ax.set_yticks([0, 180, 360])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_aspect(aspect=1)
    ax.set_title('Test trials                               ')
    ax.text(-.4, 1.08, string.ascii_uppercase[1], transform=ax.transAxes, size=12)

    ax = fig.add_axes([xpos[3], ypos[0], bz, bz*yratio])
    nb=plt.hist(error* 180/np.pi, np.linspace(0,25, 21), color = berry)
    merror = np.median(np.abs(error))*180/np.pi
    ax.scatter(merror, nb[0].max()*1.05, marker='v',color=[0,.0,0])
    ax.text(merror-1, nb[0].max()*1.13, '%2.2f$^\circ$ = median error'%merror,fontweight='bold')
    ax.set_xlabel(r'absolute angle error ($^\circ$)')
    ax.set_ylabel('trial counts')
    ax.set_xlim([0,20])
    ax.text(-.4, 1.08, string.ascii_uppercase[2], transform=ax.transAxes, size=12)

    axins = fig.add_axes([xpos[3]+bz*1, ypos[0]+bz*.45, .06,.06*yratio])
    axins.hist(E[0,0,:], 3, color=berry)
    axins.set_xlabel('median error')
    axins.set_ylabel('recordings')

    ax = fig.add_axes([xpos[2]+bz+.01, ypos[1]+bz*.82, .03, bz*.55])
    plt.scatter([0, 0, 0, 0], [1, 2, 3, 4], color=col2, s = 10)
    plt.xlim([-1, 3])
    ax.set_ylim([0.5,4.5])
    plt.scatter(np.array([0, 0, 0, 0])+2, np.array([1, 2, 3, 4]), color=col1, s = 10)
    plt.text(-0.5, 0, 'population 2', rotation=270, color = col2,size=6)
    plt.text(1.5, 0, 'population 1', rotation=270, color = col1,size=6)
    ax.axis('off')


    ax = fig.add_axes([xpos[2], ypos[1]+bz*.82, bz, bz*.55])
    logup1 = logL1[itest_trial, :] @ Kup.T
    logup2 = logL2[itest_trial, :] @ Kup.T
    plt.plot(np.linspace(0, 360, logup.shape[1]+1)[:-1], logup1, color = col1)
    plt.plot(np.linspace(0, 360, logup.shape[1]+1)[:-1], logup2, color = col2)
    ax.set_xticks([0, 180, 360])
    ax.set_yticks([-.4,-.8])
    ax.set_ylabel('avg logL')
    ax.plot(np.array([1,1]) * 180/np.pi * istim[ind_trial], larange, '--', color='k',linewidth=1)
    ax.set_ylim(larange)
    ax.set_title('Decoder probabilities             ')
    xx = (theta0+np.array([-.5, .2])) * 180/np.pi
    yy = larange
    ax.fill([xx[0], xx[1], xx[1], xx[0]], [yy[0], yy[0], yy[1], yy[1]], color=[.7, .7, .7], alpha=0.3)
    ax.text(-.4, 1.2, string.ascii_uppercase[3], transform=ax.transAxes, size=12);

    ax = fig.add_axes([xpos[2]+bz*.185, ypos[1]-.01, bz*.6, bz*.6])
    ax.plot(np.linspace(0, 360, logup.shape[1]+1)[:-1], logup1, color = col1)
    ax.plot(np.linspace(0, 360, logup.shape[1]+1)[:-1], logup2, color = col2)
    ax.set_ylabel('avg logL')
    ax.plot(np.array([1,1]) * 180/np.pi * istim[ind_trial], larange, '--', color='k',linewidth=1)
    ax.set_xlim((theta0+np.array([-.5, .2])) * 180/np.pi)
    ax.set_ylim(-.6,-.3)
    xx = (theta0+np.array([-.5, .2])) * 180/np.pi
    yy = [-2,2]
    ax.fill([xx[0], xx[1], xx[1], xx[0]], [yy[0], yy[0], yy[1], yy[1]], color=[.7, .7, .7], alpha=0.3)

    ax = fig.add_axes([xpos[3], ypos[1], bz, bz*yratio])
    ax.scatter(180/np.pi * (istim[itest] - apred1), 180/np.pi * (istim[itest] -apred2), s=1, color = berry)
    ax.set_ylim([-25, 25])
    ax.set_xlim([-25, 25])
    ax.set_title('Decoding errors ($^\circ$)                 ')
    ax.set_ylabel('population 2', color = col2)
    ax.set_xlabel('population 1', color = col1)
    ax.tick_params(axis='y', labelcolor=col2)
    ax.tick_params(axis='x', labelcolor=col1)
    ax.text(-23, 17, '$R_{S}$=%2.2f'%RS[0], color ='k')
    ax.text(-0.4, 1.08, string.ascii_uppercase[4], transform=ax.transAxes, size=12);
    #ax.scatter(E[0,0,:nstatic], ccE[0,1,:nstatic], marker='+', s=10, color = 'k', lw=0.7)

    axins = fig.add_axes([xpos[3]+bz*1, ypos[1]+bz*.38, .06,.06*yratio])
    axins.hist(ccE[0,1,:], 3, color=berry)
    axins.set_yticks([0,3])
    axins.set_xticks([.5,1])
    axins.set_xlabel(r'$R_s$')
    axins.set_ylabel('recordings')

    ax = fig.add_axes([xpos[0]-.02, ypos[2], bz, bz*yratio])
    ax.axis('off')
    nb=7
    nv = 5
    #cmap=plt.get_cmap('hsv')
    #cmap=cmap(np.linspace(0,0.6,nv))
    cmap=plt.get_cmap('twilight_shifted')
    cmap=cmap(np.linspace(0,.9,nv))
    colors= [[],[]]
    for n in range(nb):
        colors[0].append((0,0,0))
    colors[1] = cmap
    draw_neural_net(ax, 0.05, .7, 0.0, .96, [nb,nv], colors)
    #ax.set_aspect('equal')
    #ax.set_xlim(0.07, 1)
    plt.text(-.16,1.12,'Linear decoder',verticalalignment='center', size=8, color=(0,.5,0))
    plt.text(-.05,1.02,'neurons',verticalalignment='center', size=6)#rotation=90, 
    plt.text(.37,1.02, 'super-neurons',verticalalignment='center', size=6)#rotation=90, 
    plt.text(.2,.5,'   linear\nregression',verticalalignment='center', size=8, 
              fontweight='bold',rotation=270)
    ax.text(-.3, 1.08, string.ascii_uppercase[5], transform=ax.transAxes, size=12)

    ax= fig.add_axes([0.18,0.07,0.23,.22])
    #cmap=plt.get_cmap('hsv')
    #cmap=cmap(np.linspace(0,0.9,nv))
    nv=5
    theta_pref = np.linspace(0.5,2*np.pi-.5,nv)[:nv]
    v0 = theta_pref
    theta = np.linspace(0,2*np.pi,181)[:180]
    vm = np.exp((np.cos(theta[np.newaxis,:]-theta_pref[:,np.newaxis])-1)/0.1)
    vey = np.zeros(nv)
    vex = np.zeros(nv)
    istimtest = istim[itest]
    ix = itest_trial
    thstim = istimtest[ix]
    for n in range(nv):
        y = vm[n,:]-n*1.3
        ax.plot(theta, y, color=cmap[n], linewidth=1)
        vc = np.argmin(np.abs(v0[n] - np.linspace(0,2*np.pi,ypredLin.shape[1]+1)[:-1]))
        y = ypredLin[:,vc] 
        y -= y.min()
        y /= y.max()
        y -= n*1.3
        x = istim[itest]+2*np.pi*1.3
        ax.plot(x, y, ',', color=cmap[n])
        ax.scatter(x[ix], y[ix], color='k', s=10)
        vex[n] = x[ix]
        vey[n] = y[ix]
    ax.plot([x[ix],x[ix]],[y[ix]+.4,1], '--', color='k',lw=1)
    ymax = 1.4*n
    nv = 48
    cmap=plt.get_cmap('twilight_shifted')
    cmap=cmap(np.linspace(0,.9,nv))
    theta_pref = np.linspace(0,2*np.pi,nv+1)[:nv]
    theta = np.linspace(0,2*np.pi,181)[:180]
    vm = np.exp((np.cos(theta[np.newaxis,:]-theta_pref[:,np.newaxis])-1)/0.1)
    vy = np.zeros(nv)
    vx = np.zeros(nv)
    vshift = vm*1.2*np.pi+2*2.8*np.pi
    for n in range(nv):
        vx[n] = vshift[n,int(thstim/2*180/np.pi)]
        vy[n] = (1-theta_pref[n]/2*np.pi)*.68+.3
        ax.plot(vx[n], vy[n], 'o',color=cmap[n], ms=1)
    theta = np.linspace(0,2*np.pi,181)[:180]
    ax.scatter(vshift.max(),(1-thstim/2*np.pi)*.68+.3, marker='*',color=[0,.5,0],s=40,zorder=10)
    ax.text(vshift.min()-1.5,-2.,'super-neurons', rotation=270,verticalalignment='center',size=6)
    ax.text(5*np.pi,1.7,'response to\ntrial #%d'%ind_trial,verticalalignment='center',size=6, color='k')
    ax.text(6.3*np.pi,-2.5,'decoded \n= %2.0f$^\circ$'%(istim[ind_trial]*180/np.pi), color=[0,.5,0], size=6)
    ax.text(np.pi*1.1,-6.4,'train\ntargets', horizontalalignment='center',size=6)
    ax.text(np.pi*3.7,-6.4,'test\noutputs', horizontalalignment='center',size=6)
    ax.set_xlabel('stimulus orientation')
    ax.axis('off')

    ax = fig.add_axes([xpos[2], ypos[2], bz, bz*yratio])
    ax.scatter(istim[itest]* 180/np.pi, apredLin * 180/np.pi, marker='.', alpha=0.5, 
             s=.1, color = grn)
    #ax.plot(istim[itest]* 180/np.pi, apredLin * 180/np.pi, marker=',', lw=0, color=grn)#, alpha=0.2, 
             #s=1, color = grn)
    plt.xlabel(r'true angle ($^\circ$)')
    plt.ylabel(r'decoded angle ($^\circ$)')
    ax.set_xticks([0, 180, 360])
    ax.set_yticks([0, 180, 360])
    ax.set_title('Test trials                               ')
    ax.text(-.4, 1.08, string.ascii_uppercase[6], transform=ax.transAxes, size=12)

    ax = fig.add_axes([xpos[3], ypos[2], bz, bz*yratio])
    nb=plt.hist(errorLin* 180/np.pi, np.linspace(0,25, 21), color = grn)
    merror = np.median(np.abs(errorLin))*180/np.pi
    ax.scatter(merror, nb[0].max()*1.05, marker='v',color=[0,.0,0])
    ax.text(merror-2, nb[0].max()*1.13, '%2.2f$^\circ$ = median error'%merror,fontweight='bold')
    ax.set_xlabel(r'absolute angle error ($^\circ$)')
    ax.set_ylabel('trial counts')
    ax.set_xlim([0,20])
    ax.text(-0.4, 1.08, string.ascii_uppercase[7], transform=ax.transAxes, size=12);

    axins = fig.add_axes([xpos[3]+bz*1, ypos[2]+bz*.45, .06,.06*yratio])
    axins.hist(Elin[0,0,:], 3, color=grn)
    axins.set_xlabel('median error')
    axins.set_ylabel('recordings')
    
    if save_figure:
        if not os.path.isdir(os.path.join(saveroot, 'figs')):
            os.mkdir(os.path.join(saveroot, 'figs'))
        fig.savefig(os.path.join(saveroot, 'figs/fig2.pdf'))
    
    return fig

def fig3(dataroot, saveroot, save_figure=False):
    rc('font', **{'size': 12})#, 'family':'sans-serif'})#,'sans-serif':['Helvetica']})

    d = np.load(os.path.join(saveroot, 'linear_discrimination.npy'), allow_pickle=True).item()
    P = d['P']
    drange = d['drange']

    # different datasets
    inds = [np.arange(0,6,1,int), np.arange(6,9,1,int), np.arange(9,15,1,int), np.arange(15,18,1,int),
            np.arange(18,21,1,int), np.arange(21,24,1,int), np.arange(24,27,1,int)]

    Pavg = np.zeros((len(inds), len(drange)), np.float32)
    for k in range(len(inds)):
        Pavg[k,:] = P[inds[k], :].mean(axis=0)
        Pavg = (Pavg + (1-Pavg[:,::-1]))/2

    Psd = P[inds[0],:].std(axis=0)/5**.5
    Psd = (Psd + Psd[::-1])/2

    d = np.load(os.path.join(saveroot, 'runspeed_discrimination.npy'), allow_pickle=True).item()
    Prun=d['P'].mean(axis=0)
    Prun= (Prun + 1-Prun[::-1, :])/2

    d = np.load(os.path.join(saveroot, 'dense_discrimination.npy'), allow_pickle=True).item()
    drange2 = d['drange']
    Pneur = d['Pneur'].mean(axis=-1)
    Pneur = (Pneur + 1-Pneur[:, ::-1])/2
    Pnsd = d['Pneur'].std(axis=-1)/d['Pneur'].shape[-1]**.5
    Pnsd = (Pnsd + Pnsd[:,::-1])/2
    Pstim = d['Pstim'].mean(axis=-1)
    Pstim = (Pstim + 1-Pstim[:, ::-1])/2
    Pssd = d['Pstim'].std(axis=-1)/d['Pstim'].shape[-1]**.5
    Pssd = (Pssd + Pssd[:,::-1])/2

    # discrimination thresholds
    pn75 = [utils.discrimination_threshold(d['Pneur'][i,:,j], drange2)[0]
            for i in range(21) for j in range(5)]
    pn75 = np.reshape(np.array(pn75), (21, 5))
    
    ps75 = [utils.discrimination_threshold(d['Pstim'][i,:,j], drange2)[0]
            for i in range(21) for j in range(5)]
    ps75 = np.reshape(np.array(ps75), (21, 5))

    IMG = visual_stimuli(dataroot)

    my_green = [0, .5 , 0]
    fig = plt.figure(figsize=(15,10),facecolor='w', dpi = 300)

    iplot=0

    from skimage.transform import rotate
    ax = fig.add_axes([.03, 1/3 + .7 * 2/3, .1, .25 * 2/3])
    xx,yy = np.meshgrid(np.arange(0,2000)/60, np.arange(0,2000)/60)
    gratings = np.cos(xx*np.cos(np.pi/4 - np.pi/180) + yy*np.sin(np.pi/4 - np.pi/180))
    ax.imshow(np.sign(gratings), cmap=plt.get_cmap('gray'))
    ax.axis('off')
    ax.text(-.3, .5, '44$^\circ$',fontsize=12, transform=ax.transAxes)
    ax.text(0.15,1.2,'Angle > 45$^\circ$?', transform=ax.transAxes, fontsize=14)
    ax.text(-.45, 1.27, string.ascii_uppercase[iplot], transform=ax.transAxes, 
                    size=24)
    iplot+=1
    #ax.set_position(ax.get_position().bounds + np.array([-.05, -.05,0.1,0.1]))

    ax = fig.add_axes([.03, .6, .1, .25* 2/3])
    gratings = np.cos(xx*np.cos(1*np.pi/180 + np.pi/4) + yy*np.sin(1*np.pi/180 + np.pi/4))
    ax.imshow(np.sign(gratings), cmap=plt.get_cmap('gray'))
    ax.axis('off')
    ax.text(-.3, .5, '46$^\circ$',fontsize=12, transform=ax.transAxes)

    ax = fig.add_axes([.03, 1/3 + .1* 2/3, .1, .25* 2/3])
    gratings = np.cos(xx*np.cos(5*np.pi/180 + np.pi/4) + yy*np.sin(5*np.pi/180 + np.pi/4))
    #ratings[~icirc]=1
    ax.imshow(np.sign(gratings), cmap=plt.get_cmap('gray'))
    ax.axis('off')
    ax.text(-.3, .5, '50$^\circ$',fontsize=12, transform=ax.transAxes)



    ax = fig.add_axes([.2325, .76, .24 * 2/3, .24])
    pn=Pavg[0,:]
    semy=Psd
    p75 = np.interp(0.75,pn,drange)
    ax.plot(drange,100*pn, color = my_green)
    ax.scatter(drange, 100*pn, color = my_green,s=10)
    ax.fill_between(drange, 100*(pn-semy), 100*(pn+semy), facecolor=my_green, alpha=0.5)

    ax.plot(p75*np.array([1,1]), [-1,75], '--', color='k')
    ax.plot([-25,p75], [75,75], '--', color='k')
    ax.text(p75+1, 2, '%2.2f$^\circ$=\ndiscrimination\nthreshold'%p75, fontweight='bold')
    #ax.text(p75+5, 25, '%2.2f$^\circ$='%p75, fontweight='bold')
    pbeh = np.array([22,32,44,56,44*.5+50,68*.5+50,80*.5+50])
    brange = np.array([-45,-30,-15,0,15,30,45])
    pbeh = (pbeh + 100-pbeh[::-1])/2
    p75 = np.interp(75,pbeh,brange)
    print(p75)
    ax.scatter(brange,pbeh, color='r',s=10)
    ax.plot(brange,pbeh, color='r')
    #ax.scatter(40, 89, color=(1,.7,.7), marker='*', s=200)
    plt.text(.25, .85, 'neurometric\n10 trials/deg', transform=(plt.gca()).transAxes, color = my_green,fontsize=13,
            horizontalalignment='center')
    plt.text(.82, .475, 'psychometric', transform=(plt.gca()).transAxes, color = 'r',
             horizontalalignment='center',fontsize=13)
    plt.text(.82, .325, '(Abdolrahmani \n et al, 2019)', transform=(plt.gca()).transAxes, color = 'r',
             horizontalalignment='center',fontsize=11)
    #plt.text(.5, .2, '(Poort*, Khan*\net al, 2014)', transform=(plt.gca()).transAxes, color = (1,.7,.7))
    ax.set_ylim([-1, 101])
    ax.set_yticks([0,25,50,75,100])
    ax.set_xlim([-25, 25])
    #ax.set_aspect((25+25)/101)
    ax.set_ylabel('% "choose right"',fontsize=14)
    ax.set_xlabel('angle difference  ($^\circ$)',fontsize=14)
    #ax.set_position(ax.get_position().bounds - np.array([.13, -.2, 0.04, 0.04]))
    ax.text(-.35, 1.0, string.ascii_uppercase[iplot], transform=ax.transAxes, 
                    size=24)
    #ax.set_title('Orientation discrimination',fontsize=16)

    ax = fig.add_axes([.2325, 1/3 + .1, .24*2/3, .24])
    ax.plot(drange2,100*Pneur[0], color = [.5, .3, .1])
    ax.scatter(drange2, 100*Pneur[0], color = [.5, .3, .1],s=10)
    ax.fill_between(drange2, 100*(Pneur[0]-Pnsd[0]), 100*(Pneur[0]+Pnsd[0]), facecolor=[.5, .3, .1], alpha=0.5)

    ax.set_ylabel('% "choose right"',fontsize=14)
    plt.text(.3, .85, 'neurometric \n 1,000 trials/deg', transform=(plt.gca()).transAxes, color = [.5, .3, .1],fontsize=13,
            horizontalalignment='center')
    p75 = np.interp(0.75,Pneur[0],drange2)
    ax.plot(p75*np.array([1,1]), [-1,75], '--', color='k')
    ax.plot([-2,p75], [75,75], '--', color='k')
    ax.text(p75+.1, 10, '%2.2f$^\circ$'%p75, fontweight='bold')
    ax.set_xlabel('angle difference  ($^\circ$)',fontsize=14)
    ax.set_ylim([-1, 101])
    ax.set_yticks([0,25,50,75,100])
    ax.set_xlim([-2, 2])
    iplot+=1
    ax.text(-.35, 1.05, string.ascii_uppercase[iplot], transform=ax.transAxes,  size=24)


    ax = fig.add_axes([.49, .805, .1, .18])
    iplot+=1
    mux = ps75.mean(axis=1)
    sdx = ps75.std(axis=1)/5**.5

    nstimx = d['nstim'].mean(axis=-1)/4
    ax.semilogx(nstimx[:13], mux[:13], color = [.5, .3, .1], linewidth=2)
    ax.scatter(nstimx[:13], mux[:13], color = [.5, .3, .1])
    ax.fill_between(nstimx[:13], mux[:13]-sdx[:13], mux[:13]+sdx[:13], facecolor=[.5, .3, .1], alpha=0.5)

    ax.set_ylabel('discrimination\n threshold ($^{\circ}$)')
    ax.set_xlabel('trials / deg\n (training set)')
    ax.set_ylim([0, 2])
    ax.set_xlim([10, 1000])
    ax.text(-.5, 1.075, string.ascii_uppercase[iplot], transform=ax.transAxes,  size=24)
    ax.text(-.25, 1.075, 'Asymptotics', transform=ax.transAxes,  size=14)

    ax = fig.add_axes([.605, .805, .1, .18])
    mux = pn75.mean(axis=1)
    sdx = pn75.std(axis=1)/5**.5
    nnx = d['npop'].mean(axis=-1)
    ax.semilogx(nnx[:16], mux[:16], color = [.5, .3, .1], linewidth=2)
    ax.scatter(nnx[:16], mux[:16], color = [.5, .3, .1])
    ax.fill_between(nnx[:16], mux[:16]-sdx[:16], mux[:16]+sdx[:16], facecolor=[.5, .3, .1], alpha=0.5)
    ax.set_yticks([])
    ax.set_xlabel('neurons')
    ax.set_ylim([0, 2])
    #ax.set_xlim([10, 1000])

    ax = fig.add_axes([.49, 1/3 + .1, .1, 2/3 * .25])
    iplot+=1
    pn=Prun[:, 0]
    p75 = np.interp(0.75,pn,drange)
    ax.plot(drange,100*pn, color = my_green)
    ax.scatter(drange, 100*pn, color = my_green,s=10)
    ax.plot(p75*np.array([1,1]), [-1,75], '--', color='k')
    ax.plot([-25,p75], [75,75], '--', color='k')
    ax.text(p75+5, 25, '%2.2f$^\circ$'%p75, fontweight='bold')    
    ax.set_xlim(-25,25)
    ax.set_ylim(-1,100)   
    ax.set_ylabel('% "choose right"',fontsize=14)
    ax.set_xlabel('angle difference  ($^\circ$)',fontsize=14)
    ax.text(-.5, 1.525, string.ascii_uppercase[iplot], transform=ax.transAxes,  size=24)
    ax.text(-.25, 1.525, 'Effect of running (10 trials/deg)', transform=ax.transAxes,  size=14)
    pos = ax.get_position().get_points()
    ax_inset=fig.add_axes([pos[0][0],pos[1][1]-.01,.1,.1])
    runfig=plt.imread(os.path.join(dataroot, 'sitmouse.png'))[:,:,0]
    ax_inset.imshow(runfig, cmap=plt.get_cmap('gray'),vmin=0,vmax=1)
    ax_inset.axis('off')

    ax = fig.add_axes([.605, 1/3 + .1, .1, 2/3 * .25])
    pn=Prun[:, 1]
    p75 = np.interp(0.75,pn,drange)
    ax.plot(drange,100*pn, color = my_green)
    ax.scatter(drange, 100*pn, color = my_green,s=10)
    ax.plot(p75*np.array([1,1]), [-1,75], '--', color='k')
    ax.plot([-25,p75], [75,75], '--', color='k')
    ax.text(p75+5, 25, '%2.2f$^\circ$'%p75, fontweight='bold')    
    ax.set_xlim(-25,25)
    ax.set_ylim(-1,100)   
    ax.set_yticks([])
    pos = ax.get_position().get_points()
    ax_inset=fig.add_axes([pos[0][0],pos[1][1] - .01,.1,.1])
    runfig=plt.imread(os.path.join(dataroot, 'runmouse.png'))[:,:,0]
    ax_inset.imshow(runfig, cmap=plt.get_cmap('gray'),vmin=0,vmax=1)
    ax_inset.axis('off')


    x0 = .03
    ipermute = [0, 2, 3, 4, 5, 1]
    for t in range(6):    
        k = ipermute[t]
        ax = fig.add_axes([x0, .05, .1, .15])    
        if t==0:
            iplot+=1
            ax.set_ylabel('% "choose right"',fontsize=14)
            ax.set_xlabel('angle difference ($^\circ$)',fontsize=14)
            ax.text(-.5, 1.9, string.ascii_uppercase[iplot], transform=ax.transAxes,  size=24)
            ax.text(-.25, 1.9, 'Other stimuli (10 trials/deg)', transform=ax.transAxes,  size=14)
            ax.set_yticks([0,25,50,75,100])
        else:
            ax.set_yticks([])        

        pn=Pavg[k+1,:]
        p75 = np.interp(0.75,pn,drange)
        ax.plot(drange,100*pn, color = my_green)
        ax.scatter(drange, 100*pn, color = my_green,s=10)
        ax.plot(p75*np.array([1,1]), [-1,75], '--', color='k')
        ax.plot([-25,p75], [75,75], '--', color='k')
        ax.text(p75+5, 25, '%2.2f$^\circ$'%p75, fontweight='bold')    
        ax.set_xlim(-25,25)
        ax.set_xticks([-20,20])    
        ax.set_ylim(-1,100)   
        x0 += .1 + .015

        pos = ax.get_position().get_points()
        ax_inset=fig.add_axes([pos[0][0],pos[1][1]+.02,.1,.1])
        ax_inset.imshow(IMG[k], cmap=plt.get_cmap('gray'),vmin=0,vmax=1)    
        ax_inset.axis('off')

        xs,ys=IMG[k].shape    
        ac = (0.5,0,.5)
        if k>2:        
            plt.annotate('',(ys*.75,xs*.75), (ys*.25,xs*.25), arrowprops=dict(facecolor=ac,edgecolor=ac))    
        if k==2:
            plt.text(0, -.15, '100ms',fontweight='bold',size=12, transform = ax_inset.transAxes)

    if save_figure:
        if not os.path.isdir(os.path.join(saveroot, 'figs')):
            os.mkdir(os.path.join(saveroot, 'figs'))
        fig.savefig(os.path.join(saveroot, 'figs/fig3.pdf'), bbox_inches='tight')
    
    return fig

def fig4(dataroot, saveroot, save_figure=False):
    rc('font', **{'size': 12})#, 'family':'sans-serif'})

    fig = plt.figure(figsize=(15,10),facecolor='w', dpi = 300)

    d = np.load(os.path.join(saveroot, 'strong_learn.npy'), allow_pickle=True).item()
    nstim32, perf32, perf, nstim = d['nstim32'], d['perf32'], d['perf'], d['nstim']
    perf = np.squeeze(perf)
    nstim = np.squeeze(nstim)
    
    d = np.load(os.path.join(saveroot, 'weak_learn.npy'), allow_pickle=True).item()
    P, drange, ccN = d['P'], d['drange'], d['ccN'][0]
    
    iplot=0
    ax = fig.add_axes([.03, .6, .25, .25 * 3/2 * 60/75])

    cols = [[0, .5, 0], [255/255, 174/255, 0],[200/255, 0, 0], [0, 0, 125/255], [.7,.2,.5]]
    col2 = [[.25, .25, .5], [.5, .5, 0], [1, .5, .75]]

    learn_fig = plt.imread(os.path.join(dataroot, 'learning.png'))
    ax.imshow(learn_fig)
    ax.axis('off')
    ax.text(-.1, 1.1, string.ascii_uppercase[iplot], transform=ax.transAxes, 
                    size=24)
    ax.text(.0, 1.1, 'Perceptron learners', transform=ax.transAxes, fontsize=14)

    ax.text(.1, -.1, 'not trial-by-trial', transform=ax.transAxes, fontsize=14)
    ax.text(.325, -.2, 'linear decoder', color = cols[0], transform=ax.transAxes) 

    iplot += 1
    ax.text(1.0, 1.1, string.ascii_uppercase[iplot], transform=ax.transAxes, size=24)
    ax.text(1.1, 1.1, 'Easy task', transform=ax.transAxes, fontsize=14)
    iplot += 1
    ax.text(2.0, 1.1, string.ascii_uppercase[iplot], transform=ax.transAxes, size=24)
    ax.text(2.1, 1.1, 'Hard task', transform=ax.transAxes, fontsize=14)
    iplot += 1
    ax.text(-.1, -.425, string.ascii_uppercase[iplot], transform=ax.transAxes, size=24)
    ax.text(.0, -.425, 'Weak learners', transform=ax.transAxes, size=14)
    iplot += 1
    ax.text(2.0, -.425, string.ascii_uppercase[iplot], transform=ax.transAxes, size=24)
    ax.text(2.1, -.425, 'Easy task', transform=ax.transAxes, fontsize=14)

    ax.text(.4,   -.645, '"best neuron"\nlearner', transform=ax.transAxes, size=14, ha = 'center', color = col2[0])
    ax.text(.95,  -.645, '"one-shot"\nlearner', transform=ax.transAxes, size=14, ha = 'center', color = col2[1])
    ax.text(1.55, -.645, '"random\n projection"\nlearner', transform=ax.transAxes, size=14, ha = 'center', color = col2[2])

    ax = fig.add_axes([.35, .545, .15, .15 * 3/2])
    for j in range(4):
        ax.plot(nstim32.mean(axis=-2).mean(axis=0), 100 * perf32[j].mean(axis=-2).mean(axis=0), color=cols[j])
        #print(perf32[j, :, :, -1].mean())

    ax.set_ylim([0, 100])
    ax.plot([0, 500], [50, 50], '--', color='black')
    ax.set_xlabel('trials')
    ax.set_ylabel('percent correct')
    ax.set_yticks([0, 25, 50, 75, 100])

    ax = fig.add_axes([.6, .545, .15, .15 * 3/2])
    for j in range(4):
        ax.plot(nstim.mean(axis=0), 100 * perf[j].mean(axis=0), color=cols[j])
        #print(perf[j, :, -1].mean())
    ax.set_ylim([0, 100])
    ax.plot([0, 3500], [50, 50], '--', color='black')
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel('trials')
    ax.set_ylabel('percent correct')

    ax = fig.add_axes([.35, .845, .15, .075])
    ax.set_xlabel('stimulus angle ($^\circ$)')
    ax.set_xlim([-30, 30])
    ax.set_xticks([-20, -5, 5, 20])
    ax.spines['left'].set_visible(False)
    ax.set_yticks([])
    ax.fill_between([-30, -5], [0, 0], [1, 1], facecolor=[1, .75, .75])
    ax.set_ylim([0, 1])
    ax.fill_between([5, 30], [0, 0], [1, 1], facecolor=[.75, .75, 1])
    ax.text(-25, .25, 'left\nchoice')
    ax.text(10, .25, 'right\nchoice')

    ax = fig.add_axes([.6, .845, .15, .075])
    ax.set_xlabel('stimulus angle ($^\circ$)')
    ax.set_xlim([-2, 2])
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.spines['left'].set_visible(False)
    ax.set_yticks([])
    ax.fill_between([-2, 0], [0, 0], [1, 1], facecolor=[1, .75, .75])
    ax.set_ylim([0, 1])
    ax.fill_between([0, 2], [0, 0], [1, 1], facecolor=[.75, .75, 1])
    ax.text(-1.75, .25, 'left\nchoice')
    ax.text(.5,    .25, 'right\nchoice')


    ax = fig.add_axes([.6,  .2, .15, .15 * 3/2])
    for j in range(P.shape[1]):
        ax.plot(drange, 100*P[:,j], 'o', color = col2[j])
    ax.set_xlabel('angle difference ($^\circ$)')
    ax.set_ylabel('% "choose right"')
    ax.set_ylim([0, 1])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xticks([-20, -5, 5,20])

    ax = fig.add_axes([.075,  .24, .1, .1*3/2])
    ysort = np.sort(np.abs(ccN))[::-1]
    plt.semilogx(np.arange(ccN.size)+1, ysort, color = [.0, .0, 0])
    plt.plot(1, ysort[0], 'o', markersize=12, color=col2[0])
    plt.ylim([0, .8])
    ax.set_xticks([1, 100, 10000])
    ax.set_ylabel('correlation\n with correct choice')
    ax.set_xlabel('sorted neurons')

    ax = fig.add_axes([.225,  .24, .1, .1*3/2])
    np.random.seed(101)
    x1 = .75 * np.random.randn(100,2) + np.array([1,1])
    x2 = .75 * np.random.randn(100,2) + np.array([-1,-1])
    plt.scatter(x1[:,0], x1[:,1], color = [.5, 0, 0], s = 2)
    plt.scatter(x2[:,0], x2[:,1], color = [0, 0, .5], s = 2)
    k1 = 3
    k2 =24
    plt.plot(x1[k1, 0], x1[k1, 1], '*', color=[.5, .0, 0], markersize = 20)
    plt.plot(x2[k2, 0], x2[k2, 1], '*', color=[.0, .0, .5], markersize = 20)
    plt.plot([x1[k1, 0], x2[k2, 0]], [x1[k1, 1], x2[k2, 1]], '--', color='black', linewidth=2)
    rat = (x1[k1, 1] - x2[k2, 1]) / (x1[k1, 0] - x2[k2, 0]) 
    plt.plot([-3, 3], [3/rat, -3 /rat], '-', color='black')
    plt.xlim([-3,3])
    plt.ylim([-3,3])
    ax.set_ylabel('neural feature 1')
    ax.set_xlabel('neural feature 2')
    ax.set_yticks([])
    ax.set_xticks([])

    ax = fig.add_axes([.375,  .24, .1, .1*3/2])
    np.random.seed(101)
    x1 = .75 * np.random.randn(100,2) + np.array([1,1])
    x2 = .75 * np.random.randn(100,2) + np.array([-1,-1])
    plt.scatter(x1[:,0], x1[:,1], color = [.5, 0, 0], s = 2)
    plt.scatter(x2[:,0], x2[:,1], color = [0, 0, .5], s = 2)
    xs = np.random.randn(5,2)
    xs = xs / np.sum(xs**2,axis=1)[:, np.newaxis]
    xs = 100 * xs * np.sign(xs[:,:1])
    offs = np.random.randn(5,)
    for j in range(xs.shape[0]):
        plt.plot([-xs[j,0], xs[j, 0]], [-xs[j,1]+offs[j], xs[j,1]+offs[j]], color = 'black')
    plt.xlim([-3,3])
    plt.ylim([-3,3])
    ax.set_ylabel('neural feature 1')
    ax.set_xlabel('neural feature 2')
    ax.set_yticks([])
    ax.set_xticks([])
    
    if save_figure:
        if not os.path.isdir(os.path.join(saveroot, 'figs')):
            os.mkdir(os.path.join(saveroot, 'figs'))
        fig.savefig(os.path.join(saveroot, 'figs/fig4.pdf'), bbox_inches='tight')
    
    return fig