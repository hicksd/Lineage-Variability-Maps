import numpy as np
#import scipy.io  #required to read Matlab *.mat file
from scipy import linalg
import pandas as pd
import networkx as nx
#import pickle
import itertools
from sklearn.covariance import GraphLassoCV, ledoit_wolf, graph_lasso
from statsmodels.stats.correlation_tools import cov_nearest
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
from matplotlib.ticker import MaxNLocator
import json
import glob




def MLEchordal(S,Morder=1):
    """Calculate Kmle from the marginals of cliques and separators in S.
    S is the sample covariance. Morder is the order of the Markov chain."""
    
    p = S.shape[0] #dimensionality of one side of S
    idx = range(p) #list of variables
    K_mle = np.zeros((p,p)) # initialize K
    
    """Markov order cannot be greater than dimensionality-1"""
    if Morder > p-1:
        Morder = p-1
        
    """Identify cliques and separators in a Markov chain"""
    Cq = []; Sp = []
    for i in range(0,p-Morder):
        Cq.append(list(idx[i:i+1+Morder]))
        if i > 0:
            Sp.append(sorted(set(Cq[i-1]).intersection(set(Cq[i]))))

    """ Build K_mle from cliques and separators"""
    # Add cliques
    for c in Cq:
        Scq = S[np.ix_(c,c)]
        Kcq = np.linalg.pinv(Scq)
        K_mle[np.ix_(c,c)] += Kcq
    
    # Subtract separators
    for s in Sp:
        Ssp = S[np.ix_(s,s)]
        Ksp = np.linalg.pinv(Ssp)
        K_mle[np.ix_(s,s)] -= Ksp
        
    cov_mle = np.linalg.pinv(K_mle)
    #cov_mle = cov_mle.round(10) #remove bit noise to make symmetric
    cov_mle = (cov_mle + cov_mle.T)/2 #symmetrize 
    
    return cov_mle, K_mle
    
    
""" Modified Cholesky decomposition"""
def LDL_decomp(A):
    if not (A.T == A).all():
        print('Matrix must be symmetric!')
        print(A-A.T) #show to make sure asymmetry is just numerical error
        A = (A + A.T)/2 #make it symmetric
        
    C  = np.linalg.cholesky(A)
    d    = np.diag(C)
    D    = np.diag(d**2)
    dinv = np.diag(1/d)
    L    = C.dot(dinv)
    return L, D
        
def Estep(X,params):
    """ E-Step of the EM algorithm"""
    
    n, p = X.shape
    mu = params['mu']
    cov = params['cov']
    
    """Sufficient statistics"""
    x_cum = np.zeros(p)      # cumulative sum of completed rows 
    xx_cum = np.zeros([p,p]) # cumulative sum of completed outer products
    #ent_cum = 0              # cumulative entropy
    #num_m_cum = 0            # cumulative number of missing elements
    
    for i in range(n):
        """Partition cov into observed and missing quadrants"""
        j_o = np.array(np.where(~np.isnan(X[i,:]))).flatten()
        j_m = np.array(np.where(np.isnan(X[i,:]))).flatten()
        num_m = len(j_m)
        mu_o = mu[j_o]
        mu_m = mu[j_m]
        cov_oo = cov[np.ix_(j_o,j_o)] #np.ix_(a,b) = outer product <a,b>
        cov_mm = cov[np.ix_(j_m,j_m)]
        cov_om = cov[np.ix_(j_o,j_m)]
        cov_mo = cov[np.ix_(j_m,j_o)]
        xo = X[i,j_o] #observed values
    
        if num_m == 0: #Simple when no missing data
            x_cum += xo
            xx_cum += np.outer(xo,xo)
        else: 
            """Expected x,xx conditioned on xo, P(xm|xo;params)"""
            """Expected vals of x and xx"""
            xm_E = mu_m +cov_mo.dot(np.linalg.pinv(cov_oo)).dot(xo-mu_o) 
            x_all = np.empty(p)
            x_all[j_o] = xo
            x_all[j_m] = xm_E # use E(xm|xo;params) for missing elements
            xx_all = np.outer(x_all,x_all) #need to correct xmxm block
            """Add residual covariance E(xm*xm|xo) to xmxm:
            p.648 Eq (17.44) Hastie et al.
            p.225, Eq (11.5) Little & Rubin"""
            xmxm_E = cov_mm - cov_mo.dot(np.linalg.pinv(cov_oo)).dot(cov_om)
            xx_all[np.ix_(j_m,j_m)] += xmxm_E 
            """Cumulative sum over previous x and xx"""
            x_cum += x_all
            xx_cum += xx_all
        
            #Non-constant terms of entropy of P(xm|xo;params) for LL
            #ent_cum += 0.5*np.log(np.linalg.det(xx_mm_E))  
        #Increment cumulative number of missing elements
        #num_m_cum += num_m
    # Constant entropy term P(z|xo;params)
    #ent_const = 0.5*num_m_cum*(1+np.log(2*np.pi))

    """Expected complete data log-likelihood"""
    S = xx_cum - 2*np.outer(mu,x_cum)  + n*np.outer(mu,mu) #Sample cov
    ll = -0.5*(n*np.log(np.linalg.det(cov))+
        np.trace(S.dot(np.linalg.pinv(cov))))
    
    """Compute log-likelihood"""
    #ell = -0.5*n*np.log(np.linalg.det(2*np.pi*cov)) - \
    #    0.5*np.trace(np.linalg.pinv(cov).dot(S))
    #ll = ell #+ ent_cum + ent_const
    
    """Store sufficient statistics in dict"""
    ss = {'xTot':x_cum, 'xxOuter':xx_cum, 'nExamples':n}
    
    return ss,ll
            
class MVN:
    """ Fits a Multivariate Gaussian Distribution from a dataset which may
    have missing values by using the Expectation-Maximisation Algorithm
    Code originally sourced from:
        https://github.com/charlienash/mvn-missing-data/blob/master/mvn.py"""
    
    def __init__(self, em_tol=1e-2, em_maxIter=1000, em_verbose=True):
        self.em_tol = em_tol
        self.em_maxIter = em_maxIter
        self.em_isFitted = False
        self.em_verbose = em_verbose
        
        self.pattern = None #covariance pattern
        self.label_map = None # mapping from covariance pattern to indices
        
        #plt.rc('text', usetex=False)
        #plt.rc('font', **{'family':'serif', 'serif':['Computer Modern Roman'], 
        #                        'monospace': ['Computer Modern Typewriter']})
        #import matplotlib as mpl
        #mpl.rcParams.update(mpl.rcParamsDefault)                      


    def create_F(self, gmax,plabels):
        """ Creates the change-of-basis matrix. Each row is a basis vector. 
        Returned as a dataframe with indices and columns properly labeled. """
        
        """Create gmax Haar transform matrices and store in list"""
        Nvars = 2**gmax-1
        Hlist = [np.ones(1)[:,np.newaxis]] # Haar list, start with [[1]].
        modes = [(1,1,1)] # List of mode names
        dim_func = lambda i: 1 if i==1 else 2**(i-2) #dimensionality of irrep
        for g in range(1,gmax):
            Ng = Hlist[g-1].shape[0]
            a  = np.kron(Hlist[g-1], np.array([1.,1.]))
            b  = np.kron(np.eye(Ng),np.array([1.,-1.]))
            Hlist.append(np.concatenate([a,b],axis=0))
            for irrep in range(1,g+2): 
                for d in range(1,dim_func(irrep)+1):
                    modes.append((g+1,irrep,d)) #Hard-coded index ordering
        
        """Label modes: (Multiplicity, Irreducible, Dimension) """
        modes = pd.MultiIndex.from_tuples(modes, names=['mul','irp','dim'])

        """Direct sum of H matrices for each gen.""" 
        F = pd.DataFrame(linalg.block_diag(*Hlist),index=modes,columns=plabels)      
        """Normalize each symmetry adapted basis vector (=row)"""
        F = F.apply(lambda x: x/np.sqrt(x.dot(x)), axis=1) 
        """ Reorder indices in order of increasing index frequency """
        F = F.reorder_levels(['irp','dim','mul'],axis=0)
        """Sort symmetry-adapted basis vectors by irp then by dim"""
        F = F.sort_index(axis=0,level=['irp','dim']) #Sort by irp,dim 
        
        """Set the \ell and \tau indices to start at 0 instead of 1."""
        F.index = F.index.set_levels(F.index.levels[0]-1, level=0) #'irp' = \ell
        F.index = F.index.set_levels(F.index.levels[1]-1, level=1) #'dim' = \tau
        
        """Aggregate the degenerate eigenspaces"""
        #Q = F.groupby(axis=0,level=['irp','mul']).sum() #sum the degeneracies
        Q = F.groupby(axis=0,level=['irp','mul']).mean() #avrge the degeneracies
        
        return F,Q

    def avPattern(self, dat, pmap):
        """ Average together elements that have the same label.
        Args
        ----
        dat: array of data values
        pmap: mapping from unique array elements to pooling indices
        
        Returns
        ----
        dat: array of pooled data (changes dat in place)
        """
        for k in pmap:
            if len(dat.shape) == 1:
                q = pmap[k][0]
            elif len(dat.shape) == 2:
                q = pmap[k]
            dat[q] = dat[q].mean()

        return dat

    def find_MLE(self, cov_emp):
        """ Finds the penalised MLE given an empirical cov"""
        
        """ No decomposition: find MLE of full cov matrix all at once """
        if self.MLEalgo == 'glasso': 
            """Convert cov_emp to correlation matrix """
            sig_emp = np.sqrt(np.diag(cov_emp))
            corr_emp = cov_emp / np.outer(sig_emp,sig_emp)
            
            """ gLasso of patterned correlation matrix """
            corr_MLE = graph_lasso(corr_emp, 
                alpha = self.alpha,
                verbose = self.gl_verbose, 
                max_iter = self.gl_maxIter)[0] #Ignore prec_reg output, [1]
            
            """Convert back to covariance"""
            cov_MLE = corr_MLE * np.outer(sig_emp,sig_emp)

        """ Decomposition: Find MLE of orthogonal components then recombine"""
        if self.MLEalgo == 'ortho_glasso' or self.MLEalgo == 'ortho_select':  
            """Convert to symmetry-adapted basis"""
            scov_emp = self.F.dot(cov_emp.dot(self.F.T)) #DataFrame type
            scov_emp.columns = self.F.index #since df has no column index (hack)
            
            """Initialise penalised spec_cov"""
            scov_p = pd.DataFrame(0,columns=scov_emp.columns,
                          index=scov_emp.index)

            """Extract list of unique irreps"""
            irp = scov_emp.index.get_level_values('irp').unique().tolist()

            for i in irp:
                """Extract entire irreducible subspace"""
                scov_iD=scov_emp.xs(key=i,level='irp',axis=0,drop_level=False)\
                      .xs(key=i,level='irp',axis=1,drop_level=False)
                
                """Get multiplicity of subspace"""
                Nm_i = scov_iD.index.get_level_values('mul').unique().shape[0]
                
                if Nm_i == 1: # No glasso if multiplicity is 1
                    scov_p.loc[scov_iD.index,scov_iD.index] = scov_iD
                elif Nm_i > 1: # Use glasso
                    """Get dimensionality of subspace"""
                    Nd_i=scov_iD.index.get_level_values('dim').unique().shape[0]
                    if Nd_i == 1: #Single block only
                        cv = scov_iD.values
                        sg = np.sqrt(np.diag(cv))
                        cr = cv / np.outer(sg,sg) #elementwise division
                        
                        if self.MLEalgo == 'ortho_glasso':
                            """ Perform glasso optimization """
                            scov_p.loc[scov_iD.index,scov_iD.index] = \
                                graph_lasso(cr, alpha=self.alpha, 
                                verbose=self.gl_verbose, 
                                max_iter=self.gl_maxIter)[0] * np.outer(sg,sg)
                        if self.MLEalgo == 'ortho_select':
                            """Perform covariance selection; explicit MLE"""
                            scov_p.loc[scov_iD.index,scov_iD.index] = \
                                MLEchordal(cv,Morder=self.Morder)[0]
                            
                    elif Nd_i > 1: #Direct sum over repeated blocks
                        """Extract 1st dimensions of irreps (unique params)"""
                        scov_i1 = scov_iD\
                          .xs(key=1,level='dim',axis=0,drop_level=False)\
                          .xs(key=1,level='dim',axis=1,drop_level=False)
                        
                        """glasso over 1st dimensions"""
                        cv = scov_i1.values
                        sg = np.sqrt(np.diag(cv))
                        cr = cv / np.outer(sg,sg)

                        if self.MLEalgo == 'ortho_glasso':
                            """ Perform glasso optimization """
                            scovp_tmp = graph_lasso(cr, alpha=self.alpha, 
                                verbose=self.gl_verbose, 
                                max_iter=self.gl_maxIter)[0] * np.outer(sg,sg)
                        
                        if self.MLEalgo == 'ortho_select':
                            """Perform covariance selection; explicit MLE"""
                            scovp_tmp = MLEchordal(cv,Morder=self.Morder)[0]
                                 
                        """ direct sum over dimensionality of subspace. Nd_i
                        is the number of repeated blocks in the irrep """
                        scov_p.loc[scov_iD.index,scov_iD.index] = \
                            linalg.block_diag(*[scovp_tmp]*Nd_i)
                
            cov_MLE = self.F.T.dot(scov_p.dot(self.F)).values
                    
        return cov_MLE
            

    def _mStep(self, ss):
        """ M-Step of the EM-algorithm.
        The M-step takes the expected sufficient statistics computed in the 
        E-step, and maximizes the expected complete data log-likelihood with 
        respect to the parameters.
        
        Args
        ----
        ss : dict
        pattern: matrix of labels to identify unique covariance matrix elements
        
        Returns
        -------
        params : dict
        """
        mu_emp = 1/ss['nExamples'] * ss['xTot']

        """ Pool mu elements according to pattern matrix """
        if self.pattern == True:
            mu_emp = self.avPattern(mu_emp, self.mu_map)

        cov_emp = 1/ss['nExamples'] * (ss['xxOuter']) - np.outer(mu_emp,mu_emp)

        """ Pool cov elements according to pattern matrix """
        if self.pattern == True:
            cov_emp = self.avPattern(cov_emp, self.cov_map)

        """Check that the covariance matrix is positive definite. If it is not
        this usually spells doom for the iteration."""
        if ~np.all(np.linalg.eigvals(cov_emp) > 0):
            print('MStep: Matrix is NOT positive definite')
            print('Replace cov with nearest positive definite matrix')
            cov_emp = cov_nearest(cov_emp)
            #cov_emp = (cov_emp + cov_emp.T)/2.
        
        """ Find MLE for covariance """
        cov_MLE = self.find_MLE(cov_emp)
            
        """Make sure that shared elements are equal (the glasso nearly preserves 
        their equality, just not exactly, so need to pool elements again)"""
        #if self.pattern == True:
        #    self.avPattern(corr_p, self.cov_map) 

        #mu = mu_emp.copy()
        #cov = cov_p.copy()
        
        # Store params in dictionary
        params = {
            'mu'   : mu_emp,
            'cov'  : cov_MLE
             }

        return params
    
    def fit(self, dfX, pattern=None, paramsInit=None, 
            MLEalgo='ortho_select', alpha=0.0, Morder=1, 
            gl_maxIter=500, gl_verbose=False):
        """ Fit the model using EM with data X.
        Args
        ----
        dfX : DataFrame, [nExamples, dataDim]
            df of training data, where nExamples is the number of
            examples and dataDim is the number of dimensions.
        MLEalgo: algorithm for finding MLE
            'glasso' - graphical lasso of observable covariance
            'ortho_glasso' - glasso of orthogonal components
            'ortho_select' - explicit MLE from covariance selection; this
                    requires setting the Markov order, Morder 
        alpha: regularisation parameter for glasso (0=no regularisation)
        Morder: order of the Markov chain to use in covariance selection
        """
        nExamples, dataDim = np.shape(dfX)
        X = dfX.values.astype(float)
        plabels = dfX.columns #name of each data variable
        
        self.alpha = alpha
        self.MLEalgo = MLEalgo
        self.Morder = Morder
        
        self.gl_maxIter = gl_maxIter
        self.gl_verbose = gl_verbose

        """ Load positions of shared parameters in mean and cov"""
        if pattern is not None:
            self.pattern = True
            self.cov_pattern = pattern.values
            self.mu_pattern = np.diag(pattern.values) #derived from cov_pattern
            
            # Create dicts mapping unique cov element identifier to index: 
            # key=unique identifier; value=list of indices
            pattern_to_map = lambda p: {i:np.where(p==i) for i in np.unique(p)}
            self.cov_map = pattern_to_map(self.cov_pattern)
            self.mu_map  = pattern_to_map(self.mu_pattern)          
        
        """ Initial guess of mu and cov; required for missing data """
        if paramsInit is None:
            #params = {'mu' : np.zeros(dataDim),#np.random.normal(size=dataDim),
            #          'cov' : np.eye(dataDim)}
            
            mu00 = dfX.mean().mean() #global mean
            mu0 = dfX.mean().values
            mu0 = self.avPattern(mu0, self.mu_map) #pooled mean values
            mu0[np.isnan(mu0)] = mu00 #remaining nan's set to global mean
            
            var00 = dfX.var().mean() #global variance
            var0 = dfX.var().values
            var0 = self.avPattern(var0, self.mu_map) #pooled variance
            var0[np.isnan(var0)] = var00 #remaining nan's set to global mean
            cov0 = np.diag(var0)
            
            params = {'mu':mu0, 'cov':cov0}
            
            #mu0 = dfX.fillna(dfX.mean()).mean().values
            #cov0 = dfX.fillna(dfX.mean()).cov().values
            #params = {'mu': mu0, 'cov':cov0}
        else:
            params = paramsInit

        """ Create change-of-basis matrix """
        gens = [len(id) for id in plabels] #recover generation values
        gmin, gmax = min(gens), max(gens)
        self.F, self.Q = self.create_F(gmax,plabels) # Change-of-basis matrix
        
        """ Initialise log-likelihood """
        oldL = -np.inf

        """ Iterate over E and M steps """
        for i in range(self.em_maxIter):
            """E-Step"""
            #ss.keys() = {'xTot', 'xxOuter', 'nExamples'}
            """ WHAT IS THE DIFFERENCE BETWEEN Estep and _eStep??? """
            ss, ll = Estep(X,params) 
            #ss, ll = self._eStep(X, params)
            
            """M-step"""
            params = self._mStep(ss)

            """Evaluate likelihood"""
            if self.em_verbose:
                print("Iter {:d}   NLL: {:.3f}   Change: {:.3f}".format(i,
                      -ll, -(ll-oldL)), flush=True)

            """Break if change in likelihood is small"""
            if np.abs(ll - oldL) < self.em_tol:
                break
            oldL = ll

        else:
            if self.em_verbose:
                print("MVN did not converge within the specified tolerance." +
                      " You might want to increase the number of iterations.")
        
        """Mean, Covariance, Sigma """
        mu  = params['mu']
        cov = params['cov']
        sig = np.sqrt(np.diag(cov))
        self.mu  = pd.Series(mu, index=plabels)
        self.cov = pd.DataFrame(cov, index=plabels, columns=plabels)
        self.sig = pd.Series(sig, index=plabels)
        
        """Correlation """
        corr = self.cov/np.outer(self.sig,self.sig) 
        self.corr = pd.DataFrame(corr, index=plabels, columns=plabels)
        
        """ Partial correlation """
        prec = np.linalg.pinv(cov)
        psig = 1/np.sqrt(np.diag(prec))
        pcorr = -prec * np.outer(psig,psig)
        np.fill_diagonal(pcorr,1.0)
        
        self.prec  = pd.DataFrame(prec, index=plabels, columns=plabels)
        self.psig  = pd.Series(psig, index=plabels)
        self.pcorr = pd.DataFrame(pcorr, index=plabels, columns=plabels)
               
        """ Transform to spectral covariance & precision matrices """
        self.spec_cov = self.F.dot(self.cov.dot(self.F.T))
        self.spec_prec = self.F.dot(self.prec.dot(self.F.T))
        
        spec_sig = np.sqrt(np.diag(self.spec_cov.values))
        spec_psig = 1/np.sqrt(np.diag(self.spec_prec.values))
        self.spec_corr = self.spec_cov/np.outer(spec_sig,spec_sig)
        self.spec_pcorr = -self.spec_prec * np.outer(spec_psig,spec_psig)
        np.fill_diagonal(self.spec_pcorr.values,1.0)
        
        #spec_cov = F @ self.cov @ F.T
        #spec_prec = F @ self.prec @ F.T
        #self.spec_cov = pd.DataFrame(spec_cov, index=modes, columns=modes)
        #self.spec_prec = pd.DataFrame(spec_prec, index=modes, columns=modes)
        #self.F = pd.DataFrame(F, index=modes, columns=plabels)
        #self.T = pd.DataFrame(F.T, index=plabels, columns=modes)
        
        """ Bayesian network """
        self.Q_cov = self.Q.dot(self.cov.dot(self.Q.T))
        
        modes = self.Q.index
        nq = modes.shape[0]
        
        self.Q_L = pd.DataFrame(np.zeros((nq,nq)),index=modes,columns=modes)
        self.Q_Linv = pd.DataFrame(np.zeros((nq,nq)),index=modes,columns=modes)
        self.Q_D = pd.DataFrame(np.zeros((nq,nq)),index=modes,columns=modes)
        
        """Extract list of unique irreps"""
        irp = modes.get_level_values('irp').unique().tolist()
        
        for i in irp:
            """Extract entire irreducible subspace"""
            iQ_cov = self.Q_cov.xs(key=i,level='irp',axis=0,drop_level=False)\
                      .xs(key=i,level='irp',axis=1,drop_level=False)
                
            """Get multiplicity of subspace"""
            #Nm_i = modes.get_level_values('mul').unique().shape[0]    
            
            L,D = LDL_decomp(iQ_cov.values.round(10))
            Linv = linalg.inv(L)
            
            self.Q_L.loc[iQ_cov.index,iQ_cov.index] = L
            self.Q_Linv.loc[iQ_cov.index,iQ_cov.index] = Linv
            self.Q_D.loc[iQ_cov.index,iQ_cov.index] = D
        
        """Update Object attributes"""
        self.trainNll = ll
        self.em_isFitted = True
        self.dataDim = dataDim  
        
        
    def imageplot(self, data='pcorr', output=None):
        """ Shows heatmap of data matrix
        Data = name of the dataframe to be displayed """
        
        nrows, ncols = 1, 1
        fig = plt.figure(figsize=(5,4))
        gs = plt.GridSpec(nrows,ncols)

        if data == 'cov':
            im = self.cov
            #cmap = plt.get_cmap('Oranges')
            cmap = plt.get_cmap('PuOr_r')
            vmin, vmax = im.min().min(), im.max().max()
            #pnorm=plt.Normalize(vmin=vmin, vmax=vmax)
            pnorm = MidpointNormalize(midpoint=0,vmin=vmin, vmax=vmax)
        if data == 'prec':
            im = self.prec
            cmap = plt.get_cmap('PuOr_r')
            vmin, vmax = im.min().min(), im.max().max()
            pnorm = MidpointNormalize(midpoint=0,vmin=vmin, vmax=vmax)
        if data == 'corr' or data == 'spec_corr':
            im = self.corr if data == 'corr' else self.spec_corr
            #cmap = plt.get_cmap('Oranges')
            #vmin, vmax = 0, 1 
            #pnorm=plt.Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.get_cmap('PuOr_r')
            vmin, vmax = im.min().min(), im.max().max()
            vlim = np.abs((vmin,vmax)).max()
            vmin, vmax = -vlim, vlim
            pnorm = MidpointNormalize(midpoint=0,vmin=vmin, vmax=vmax)
        if data == 'pcorr' or data=='spec_pcorr':
            im = self.pcorr if data == 'pcorr' else self.spec_pcorr
            #cmap = plt.get_cmap('gist_heat_r')
            #vmin, vmax = 0, 1; 
            #pnorm=plt.Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.get_cmap('PuOr_r')
            vmin, vmax = im.min().min(), im.max().max()
            vlim = np.abs((vmin,vmax)).max()
            vmin, vmax = -vlim, vlim
            pnorm = MidpointNormalize(midpoint=0,vmin=vmin, vmax=vmax)
        if data == 'spec_cov':
            im = self.spec_cov
            cmap = plt.get_cmap('gist_heat_r')
            vmin, vmax = im.min().min(), im.max().max()
            pnorm=plt.Normalize(vmin=vmin, vmax=vmax)
        if data == 'spec_prec':
            im = self.spec_prec
            cmap = plt.get_cmap('PuOr_r')
            vmin, vmax = im.min().min(), im.max().max()
            vlim = np.abs((vmin,vmax)).max()
            vmin, vmax = -vlim, vlim
            pnorm = MidpointNormalize(midpoint=0,vmin=vmin, vmax=vmax)
        if data == 'Q_L':
            im = self.Q_L
            cmap = plt.get_cmap('PuOr_r')
            vmin, vmax = im.min().min(), im.max().max()
            vlim = np.abs((vmin,vmax)).max()
            vmin, vmax = -vlim, vlim
            pnorm = MidpointNormalize(midpoint=0,vmin=vmin, vmax=vmax)
        if data == 'Q_Linv':
            im = self.Q_Linv
            cmap = plt.get_cmap('PuOr_r')
            vmin, vmax = im.min().min(), im.max().max()
            vlim = np.abs((vmin,vmax)).max()
            vmin, vmax = -vlim, vlim
            pnorm = MidpointNormalize(midpoint=0,vmin=vmin, vmax=vmax)
        if data == 'Q_D':
            im = self.Q_D
            cmap = plt.get_cmap('PuOr_r')
            vmin, vmax = im.min().min(), im.max().max()
            vlim = np.abs((vmin,vmax)).max()
            vmin, vmax = -vlim, vlim
            pnorm = MidpointNormalize(midpoint=0,vmin=vmin, vmax=vmax)
            
            
            
        ax = fig.add_subplot(gs[0,0])
        ax.imshow(im.values, interpolation='nearest', cmap=cmap,
            clim=(vmin,vmax), norm=pnorm)
        
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=pnorm)
        sm._A = []
        cbar = fig.colorbar(sm,ax=ax)

        #print(im.columns)
        
        plabels = im.columns
        dummy = range(len(plabels)) # dummy axis
        
        plt.xticks(dummy,plabels,size=8,rotation='vertical')
        plt.yticks(dummy,plabels,size=8)
        ax.xaxis.tick_top()
        
        """Show boundaries for the irrep blocks and the blocks within irreps"""
        if data == 'spec_cov' or data == 'spec_prec' or data == 'spec_corr' \
            or data == 'spec_pcorr':
            
            """ List of irrep and dimension indices, indexed by group vars"""
            irreps = plabels.get_level_values('irp').tolist()
            idims = plabels.get_level_values('dim').tolist()

            """ Create dataframe of multiindex to enable groupby queries"""
            indx = pd.DataFrame([(a,b,c) for a,b,c in plabels],
                columns=plabels.names) #df version of multiindex
            """ Dim of irrep, indexed by each group variable """
            dim = indx.groupby('irp')['dim'].transform(lambda x:len(x.unique()))
            """ Multiplicity of irrep, indexed by each group variable"""
            mul = indx.groupby('irp')['mul'].transform(lambda x:len(x.unique()))
 
            """ Find position of irrep boundaries """
            w_irrep = np.where(np.diff(irreps) == 1)[0]
            pos_irrep = w_irrep + 0.5         
            """ Size of each irrep given by multiplicity*dimensionality """
            md = mul*dim
            
            """ Plot irrep boundaries """
            for i in range(len(w_irrep)):
                b = w_irrep[i]
                c,lo,hi = pos_irrep[i], pos_irrep[i]-md[b], pos_irrep[i]+md[b+1]
                ax.plot([lo,hi],[c,c], linestyle='dashed', color='k')
                ax.plot([c,c],[lo,hi], linestyle='dashed', color='k')

            """ Find position of subspace boundaries between repeated subspaces
             in an irrep"""
            w = np.where(np.diff(idims) == 1)[0]
            dpos =  w + 0.5 
            
            """ Plot subspace boundaries """
            for i in range(len(w)):
                b = w[i]
                c,lo,hi = dpos[i], dpos[i]-mul[b], dpos[i]+mul[b]
                ax.plot([lo,hi],[c,c], linestyle='dotted', color='k')
                ax.plot([c,c],[lo,hi], linestyle='dotted', color='k')
        
        if data == 'Q_L' or data == 'Q_Linv' or data == 'Q_D':
            
            """ List of irrep indices, indexed by group vars"""
            irreps = plabels.get_level_values('irp').tolist()

            """ Create dataframe of multiindex to enable groupby queries"""
            indx = pd.DataFrame([(a,b) for a,b in plabels],
                columns=plabels.names) #df version of multiindex

            """ Multiplicity of irrep, indexed by each group variable"""
            mul = indx.groupby('irp')['mul'].transform(lambda x:len(x.unique()))
 
            """ Find position of irrep boundaries """
            w_irrep = np.where(np.diff(irreps) == 1)[0]
            pos_irrep = w_irrep + 0.5         
            
            """ Plot irrep boundaries """
            md = mul
            for i in range(len(w_irrep)):
                b = w_irrep[i]
                c,lo,hi = pos_irrep[i], pos_irrep[i]-md[b], pos_irrep[i]+md[b+1]
                ax.plot([lo,hi],[c,c], linestyle='dashed', color='k')
                ax.plot([c,c],[lo,hi], linestyle='dashed', color='k')
        
        
        
        plt.tight_layout()
        if output=='pdf':
            plt.savefig(data+'.pdf',format='pdf')     
        if output=='eps':
            plt.savefig(data+'.eps',format='eps')     
             
    def graphplot(self, data='pcorr', threshold=0.3, output=None):
        """ Shows matrix im as an undirected graph in observable space.
        im = matrix """

        if data == 'cov':
            im = self.cov
            #cmap = plt.get_cmap('Oranges')
            cmap = plt.get_cmap('PuOr_r')
            vmin, vmax = im.min().min(), im.max().max()
            pnorm = MidpointNormalize(midpoint=0,vmin=vmin, vmax=vmax)
            thk = 3
        if data == 'corr':
            im = self.corr
            #cmap = plt.get_cmap('Oranges')
            cmap = plt.get_cmap('PuOr_r')
            #vmin, vmax = im.min().min(),im.max().max()
            #vlim = np.abs((vmin,vmax)).max()
            vmin, vmax = -1,1 # -vlim, vlim
            #pnorm = plt.Normalize(vmin=vmin,vmax=vmax)
            pnorm = MidpointNormalize(midpoint=0,vmin=vmin, vmax=vmax)
            thk = 3
        if data == 'pcorr':
            im = self.pcorr
            #cmap = plt.get_cmap('gist_heat_r')
            cmap = plt.get_cmap('PuOr_r')
            #vlim = np.abs((im.min().min(),im.max().max())).max()
            vmin, vmax = -1,1
            pnorm = MidpointNormalize(midpoint=0,vmin=vmin, vmax=vmax)
            thk = 4

        plabels = im.columns
        gen_max = max([len(i) for i in plabels])
        #gen_max = int(np.ceil(np.log2(im.shape[0])))
        
        """Create tree with integer labels"""
        G = nx.Graph()
        for i in range(1,2**(gen_max)):
            G.add_node(i)
        
        """Create mapping to convert from integer to binary convention"""
        mapping = {i:str(bin(i))[2:] for i in G.nodes()}
        
        """ Assign node positions for graph drawing"""
        dr = 1 # radial step between generations
        theta0 = np.pi/3 # angular size of sector
        
        pos = {}
        for i in G.nodes():
            x = int(np.log2(i))
            r = x*dr
            dtheta = 0 if x==0 else theta0/(2**x-1)
            theta = (i - 2**x - (2**x-1)/2)*dtheta
            x = r*np.cos(theta)
            y = r*np.sin(theta)
            pos[mapping[i]] = (x,y)

        """Relabel nodes using binary convention"""
        G = nx.relabel_nodes(G,mapping)
        
        """Assign weights to nodes"""
        for i in G.nodes():
            for j in G.nodes():
                if i < j:
                    w_ij = im.loc[i,j]
                    if np.abs(w_ij) > threshold:
                        G.add_edge(i,j,weight=w_ij)
        
        """Display tree"""
        nrows, ncols = 1, 1
        fig = plt.figure(figsize=(5,4))
        gs = plt.GridSpec(nrows,ncols)
        
        ax = fig.add_subplot(gs[0,0],aspect='equal') #ax.set_aspect(1)
        plt.axis('off')
       
        edges = G.edges()
        weights = [np.abs(G[u][v]['weight'])*thk for u,v in edges]
        labels = {i:i for i in G.nodes()}
        edge_colors = [G[u][v]['weight'] for u,v in edges]
        
        #vmin, vmax = min(edge_colors), max(edge_colors)

        nx.draw_networkx_nodes(G,pos,ax=ax, node_size=200, node_color='w',
            alpha=0.5, linewidths=0.5).set_edgecolor('k')
        nx.draw_networkx_edges(G,pos,ax=ax,edges=edges, width=weights, 
                       edge_cmap=cmap, edge_color=pnorm(edge_colors), 
                       edge_vmin=pnorm(vmin), edge_vmax=pnorm(vmax))
        nx.draw_networkx_labels(G,pos,labels,ax=ax,font_size=5)
        
        sm = plt.cm.ScalarMappable(norm=pnorm,cmap=cmap)
        sm._A = []
        cbar = fig.colorbar(sm,ax=ax)

        gs.tight_layout(fig)
        #plt.tight_layout()
        if output=='pdf':
            plt.savefig(data+'-graph'+'.pdf',format='pdf')
        if output=='eps':
            plt.savefig(data+'-graph'+'.eps',format='eps')

    def specgraph(self, data='spec_pcorr', output=None):
        """Plots a graph of the spectral covariance or spectral precision"""
        # Edge thicknesses come from the off-diagonal elements of 'im'
        # Node sizes come from 'diag'
        """
        # spec_corr and spec_pcorr DON'T REALLY MAKE SENSE FOR A DIRECTED GRAPH
        if data == 'spec_corr':
            im = self.spec_corr
            diag = np.diag(self.cov)
            thk = 3
            cmap = plt.get_cmap('PuOr_r')
            #vlim = np.abs((im.min().min(),im.max().max())).max()
            vmin, vmax = -1, 1 #-vlim, vlim
            pnorm = MidpointNormalize(midpoint=0,vmin=vmin, vmax=vmax)
        if data == 'spec_pcorr':
            im = self.spec_pcorr
            diag = np.diag(1/self.spec_prec)
            thk = 6
            cmap = plt.get_cmap('PuOr_r')
            #vlim = np.abs((im.min().min(),im.max().max())).max()
            vmin, vmax = -1, 1 #-vlim, vlim
            pnorm = MidpointNormalize(midpoint=0,vmin=vmin, vmax=vmax)
        """
        if data == 'Q_Linv':
            #need to transpose since the edges found from upper triangle
            # -ve is necessary to get convert from Linv to regression param
            im = -self.Q_Linv.T 
            diag = np.diag(self.Q_D) #np.sqrt(np.diag(self.Q_D))
            thk = 2
            cmap = plt.get_cmap('PuOr_r')
            #cmap = plt.get_cmap('Spectral')
            vlim = np.abs((im.min().min(),im.max().max())).max()
            vmin, vmax = -vlim, vlim
            pnorm = MidpointNormalize(midpoint=0,vmin=vmin, vmax=vmax)
        
        """ Assign indices to the array of node strengths """
        diag = pd.Series(diag,index=im.index)
        diag = diag.groupby(axis=0,level=['irp','mul']).sum()           
        dmin, dmax = min(diag), max(diag)
        node_size = (diag - dmin)/(dmax - dmin)*(400-10) + 10
        
        """DO THE SAME FOR EDGE THICKNESS"""
        
        """ MAYBE SHOULD COLLAPSE DEGENERACY DIMENSION FIRST BEFORE
        PROCEEDING WITH THE PLOT. 
        CANNOT DO THIS WITH PCORR AND CORR. NEED TO USE COV AND PREC?"""
        #im = im.groupby(axis=0,level=['irp','mul']).sum()
        #im = im.groupby(axis=1,level=['irp','mul']).sum()
        
        """ List of unique irreps """
        irps = im.index.get_level_values('irp').unique().tolist()
        """ List of unique generations """
        all_gens = im.index.get_level_values('mul').unique().tolist()
        
        
        G_dict = {} #dict of graphs for each irrep
        pos_dict = {} #dict of node positions
        lbl_dict = {} #dict of labels
        nodesize_dict = {} #dict of node sizes
        
        for irp in irps:
            im_iD = im.xs(key=irp,level='irp',axis=0,drop_level=False)\
                      .xs(key=irp,level='irp',axis=1,drop_level=False)
            if im_iD.index.nlevels == 3: #if the 'dim' level exists
                im_i1 = im_iD.xs(key=1,level='dim',axis=0,drop_level=True)\
                         .xs(key=1,level='dim',axis=1,drop_level=True)
            else: # if the 'dim' level does not exist
                im_i1 = im_iD
            
            """List unique generations available in this irrep"""
            gens = sorted(im_i1.index.get_level_values('mul').unique().tolist())
            
            G_dict[irp] = nx.DiGraph() #Directed Graph
            pos_dict[irp] = {} #node positions
            lbl_dict[irp] = {} #node labels
            nodesize_dict[irp] = {} #node sizes
            
            """ Add node attributes """
            for g in gens:
                G_dict[irp].add_node((irp,g))
                pos_dict[irp][(irp,g)] = (g,irp)
                lbl_dict[irp][(irp,g)] = '%s,%s'%(irp,g)
                nodesize_dict[irp][(irp,g)] = node_size[(irp,g)]
            
            """ Add edge attributes """
            for ja in range(len(gens)):
                for jb in range(ja+1,len(gens)):
                    a, b = (irp,gens[ja]), (irp,gens[jb])
                    w = im_i1.loc[a,b]
                    #G_dict[irp].add_edges(a,b,weight=im_i1.loc[a,b])
                    G_dict[irp].add_weighted_edges_from([(a,b,w)])
        
        """Custom function to draw an arc between nodes"""
        def draw_curved_edge(ax,p0,p1,rad=-0.3, width=3, clr='C0'):
            """p0, p1: (x,y) tuples specifying the edge start and end points. 
                  rad: angle in degrees cw """
            
            dist = np.sqrt((p1[0]-p0[0])**2 + (p1[1]-p0[1])**2)
            rad_ = rad if dist > 1 else 0
            patch = patches.FancyArrowPatch(p0,p1,arrowstyle='simple',
                connectionstyle='arc3, rad=%s'%rad_, 
                mutation_scale=10,linewidth=width,color=clr,
                shrinkA=10.0, shrinkB=10.0)
            ax.add_patch(patch)
    
            return patch
            
        nrows, ncols = 1, 1
        fig = plt.figure(figsize=(5,4))
        gs = plt.GridSpec(nrows,ncols)
        
        """Set bounds for colormap: for input into cmap(norm(value))"""
        #norm = plt.Normalize(vmin=vmin,vmax=vmax)
        
        ax = fig.add_subplot(gs[0,0])#,aspect='equal')
        #plt.axis('off')
        buffer = 0.2
        ax.set_xlim(min(all_gens)-buffer,max(all_gens)+buffer)
        ax.set_ylim(min(irps)-buffer,max(irps)+buffer)
        ax.set_xlabel('Generation-g')
        ax.set_ylabel('Subtree-$\ell$')
        #[int(g) for g in all_gens]
        #ax.set_xticklabels(
        #    [item.get_text()[0] for item in ax.get_xticklabels()])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        
        """ Draw each connected component """
        for k in G_dict:
            G = G_dict[k]
            pos = pos_dict[k]
            lbl = lbl_dict[k]
            nodesize = nodesize_dict[k]
            
            """Create a list of nodes along with the correponding node size.
            This is needed as an input to draw_networkx_nodes to ensure that 
            the node sizes are assigned to the correct nodes """
            node_list = G.nodes()
            nodesize_list = [nodesize[n] for n in node_list]
            
            nx.draw_networkx_nodes(G,pos,ax=ax, node_color='w', alpha=0.5, 
                linewidths=2, node_shape='o', nodelist = node_list,
                node_size = nodesize_list).set_edgecolor('k')
            #nx.draw_networkx_labels(G,pos,lbl,ax=ax,font_size=5)
            """Draw each edge """
            for a,b in G.edges():
                ya, xa = a
                yb, xb = b 
                wght = np.abs(G[a][b]['weight'])
                w = thk*wght
                clr_ = cmap(pnorm(G[a][b]['weight'])) #(r,g,b,a) tuple
                if wght > 0.2:
                    draw_curved_edge(ax,(xa,ya),(xb,yb),width=w,clr=clr_)

        sm = plt.cm.ScalarMappable(norm=pnorm,cmap=cmap)
        sm._A = []
        cbar = fig.colorbar(sm,ax=ax)
        gs.tight_layout(fig)

        if output=='pdf':
            plt.savefig(data+'-graph'+'.pdf',format='pdf')
        if output=='eps':
            plt.savefig(data+'-graph'+'.eps',format='eps')
        
        """ Assign node positions for graph drawing"""
        #dr = 1 # radial step between generations
        #theta0 = np.pi/4 # angular size of sector
        
        #pos = {}
        #for i in G.nodes():
        #    x = int(np.log2(i))
        #    r = x*dr
        #    dtheta = 0 if x==0 else theta0/(2**x-1)
        #    theta = (i - 2**x - (2**x-1)/2)*dtheta
        #    x = r*np.cos(theta)
        #    y = r*np.sin(theta)
        #    pos[mapping[i]] = (x,y)

        """Relabel nodes using binary convention"""
        #G = nx.relabel_nodes(G,mapping)
        
        """Assign weights to nodes"""
        #for i in G.nodes():
        #    for j in G.nodes():
        #        if i != j:
        #            G.add_edge(i,j,weight=im.loc[i,j])
        
        """Display tree"""
        #nrows, ncols = 1, 1
        #fig = plt.figure(figsize=(5,4))
        #gs = plt.GridSpec(nrows,ncols)
        
        #ax = fig.add_subplot(gs[0,0],aspect='equal') #ax.set_aspect(1)
        #plt.axis('off')
       
        #edges = G.edges()
        #weights = [G[u][v]['weight']*thk for u,v in edges]
        #labels = {i:i for i in G.nodes()}
        #edge_colors = [G[u][v]['weight'] for u,v in edges]
        
        #vmin, vmax = 0,1 #min(edge_colors), max(edge_colors)

        #nx.draw_networkx_nodes(G,pos,ax=ax, node_size=200, node_color='w',
        #    alpha=0.5, linewidths=0.5).set_edgecolor('k')
        #nx.draw_networkx_edges(G,pos,ax=ax,edges=edges, width=weights, 
        #               edge_cmap=cmap, edge_color=edge_colors, 
        #               edge_vmin=vmin, edge_vmax=vmax)
        #nx.draw_networkx_labels(G,pos,labels,ax=ax,font_size=5)
        
        
        #plt.tight_layout()
        #if output=='pdf':
        #    plt.savefig(data+'-network'+'.pdf',format='pdf')
            
 
        
    def diagplot(self, output=None):
        """Plots the diagonal quantities: 
        mean, variance, and residual variance """
        
        gens = [len(c) for c in self.mu.index]
        
        #plt.rc('text', usetex=True)
        #plt.rc('font', family='serif')

        nrows, ncols = 1, 2
        fig = plt.figure(figsize=(5,4))
        gs = plt.GridSpec(nrows,ncols)
        
        ax = fig.add_subplot(gs[0,0])
        plt.plot(gens, self.mu,'ko-')
        plt.xticks(gens)
        ax.set_ylim(ymin=0)
        ax.set_xlabel('Generation')
        ax.set_ylabel(r'$\mu$')
        
        ax = fig.add_subplot(gs[0,1])
        vtot, = plt.plot(gens, self.sig**2,'o-', label="Total Variance")
        vres, = plt.plot(gens, self.psig**2,'o-', label="Residual variance")
        plt.xticks(gens)
        ax.set_ylim(ymin=0)
        ax.set_xlabel('Generation')
        ax.set_ylabel(r'$\sigma^2$')
        plt.legend(handles=[vtot,vres],loc='lower right',fontsize=9)
        
        gs.tight_layout(fig)
        
        if output=='pdf':
            plt.savefig('diag.pdf',format='pdf')     

    def get_vc(self,spec_cov):
        """ Creates a dict of variance components from the diagonal of the 
        spectral covariance """
        
        """ Get sorted list of generations """
        ###gens = sorted(list(set([g for g,irrep,d in spec_cov.index])))
        ###gmin, gmax = min(gens), max(gens)

        #gens = sorted(list(set([g for l,t,g in spec_cov.index])))
        irps = spec_cov.index.levels[0]
        gens = spec_cov.index.levels[2]
        
        vc = dict()
        dims = [1]+[2**(g-2) for g in gens[1:]] # degeneracy of irreps
        
        for g in gens:
            vc_temp = [] #vc's ordered by mrca/irrep with degenerates summed
            #ddeg = [] # list of eigenvalue degeneracies ordered by irp
            #for irrep in range(1,g+1):
            for irrep in range(0,g):
                eigs = [] #create list of degenerate eigenvalues
                #dim = 1 if irrep==1 else 2**(irrep-2) #dims of irrep 
                #ddeg.append(dim)      
                #for d in range(1,dims[irrep-1]+1): # list of degen eigenvals
                for d in range(0,dims[irrep]):
                    x = (irrep,d,g) #HARD CODES ORDERING OF MULTIINDEX
                    eigs.append(spec_cov[x][x])
                    #self.spec_cov.loc[[x],[x]].values[0][0]
                vc_temp.append(sum(eigs)) # sum degenerate eigenvalues
                
            vc[g] = np.array(vc_temp)/sum(dims[0:g]) 
            #vc_cumul = vc.cumsum()
           
        return vc #returns a dict of vc's; each key corresponds to a gen
    
    def get_R2(self,cov,a,bb):
        """ Extracts R2 for a given each b and given all bb together """
        var_aa = (cov.loc[a,a].values).flatten() #Total variance of a
        R2       = np.zeros(len(bb))
        R2_cumul = np.zeros(len(bb)) #cumulative var over ancestors    
        for i in range(len(bb)):
            b = bb[i] #single ancestors
            R2[i] = cov.loc[a,b]*cov.loc[b,a]/cov.loc[b,b]/var_aa
            c = bb[0:i+1] #cumulative ancestors
            vcum = cov.loc[a,c].values @ linalg.pinv(cov.loc[c,c]) @ \
                   cov.loc[c,a].values
            R2_cumul[i] = vcum.flatten()/var_aa

        return R2, R2_cumul 
                
    def explvariance(self, a=None, bb=None, output=None, 
        cov_true=None):
        """ Plots the variance of a explained by b
        a = single variable name for response variable
        b = multiple variable names for predictor variables """
        
        """ If a or b are not defined (the usual case) then a is taken to 
        be in the last generation and b is all its direct ancestors """
        if a == None:
            a = [self.cov.columns[-1]]
        
        if bb == None:
            bb = [a[0][:i] for i,s in enumerate(a[0])][1:]
        
        #vc_all = np.diag(self.spec_cov)
        #self.spec_cov.loc[(4,4,1),(4,4,2)]
        #print(self.spec_cov)
        
        """ Generation of response variable """
        g = len(a[0])

        """ Extract variance components """
        vc = self.get_vc(self.spec_cov)[g]
        ell = self.spec_cov.index.levels[0][:g] #indices for subtrees
        #gen = self.spec_cov.index.levels[2][:g] # indices for generations
        
        eta2 = vc/vc.sum()
        eta2_cumul = eta2.cumsum()
        #print(self.get_vc(self.spec_cov))
        #print(vc)
        if cov_true is not(None):
            spec_cov_true = self.F.dot(cov_true.dot(self.F.T))
            vc_true = self.get_vc(spec_cov_true)[g]
            eta2_true = vc_true/vc_true.sum()
            eta2_true_cumul = eta2_true.cumsum()
            
        cov = self.cov.copy()

        ###Max explained variance###
        #pcov = pd.DataFrame(linalg.pinv(cov.values), cov.columns, cov.index)
        #var_a_expl = var_a_tot - (1/pcov.loc[a,a].values).flatten()

        """ Variance of a explained by bb (=total - residual)"""
        R2, R2_cumul = self.get_R2(cov, a, bb)
        if cov_true is not(None):
            R2_true, R2_true_cumul = self.get_R2(cov_true, a, bb)
        
        anc_gens = [len(i) for i in bb] #ancestral generations
        
        """ Start plotting """       
        nrows, ncols = 2, 1
        fig = plt.figure(figsize=(5,8))
        gs = plt.GridSpec(nrows,ncols)
        
        """ Variance explained by each ancestor """
        ax = fig.add_subplot(gs[0,0])
        
        #ax.axhline(var_aa, linestyle='dashed',color='grey',label='Total')
        ax.axhline(1, linestyle='solid', color='white')
        ax.axhline(0, linestyle='solid', color='white')
        #print(eta2)

        ax.plot(ell, eta2, linestyle='solid',marker='o',color='C0',
            label='$\eta^2(\ell | G={})$'.format(g))
        ax.plot(anc_gens, R2,linestyle='solid',marker='o',color='C1',
            label='$R^2(g | G={})$'.format(g))   
        
        if cov_true is not(None):
            ax.plot(ell,eta2_true,linestyle='dotted',marker='x',color='C0',
                label='$\eta^2(\ell | G={})$ - exact'.format(g))
            ax.plot(anc_gens,R2_true,linestyle='dotted',marker='x',color='C1',
                label='$R^2(g | G={})$ - exact'.format(g))
            
        #plt.xticks(range(g),[str(i) for i in range(1,g+1)])
        plt.xticks(ell)
        
        ax.set_ylabel('Proportion of Variance Explained')
        ax.set_xlabel('Subtree-$\ell$ or Generation-$g$')
        ax.legend(loc='upper left')
        ymin, ymax = ax.get_ylim()

        #ax2 = ax.twinx()
        #new_ticks = [0,0.2,0.4,0.6,0.8,1]
        #ax2.yaxis.set_ticks((new_ticks*var_a_tot-ymin)/(ymax-ymin))
        #ax2.yaxis.set_ticklabels(new_ticks)
        #ax2.set_ylabel('Proportion of Variance Explained by Ancestor')
        
        """ Cumulative proportion of variance explained """
        ax = fig.add_subplot(gs[1,0])

        #ax.axhline(var_a_tot, linestyle='dashed',color='grey',label='Total')
        ax.axhline(0, linestyle='solid', color='white')
        ax.axhline(1, linestyle='solid', color='white')
        #plt.axhline(var_a_expl,linestyle='dashed',color='C2')

        ax.plot(ell,eta2_cumul, linestyle='solid', marker='o', color='C0', 
            label='$\eta^2_{cml}$'+'($\ell$|G={})'.format(g))
        ax.plot(anc_gens,R2_cumul,linestyle='solid', marker='o',color='C1',label
            ='$R^2_{cml}$'+'($g$|G={})'.format(g))   

        if cov_true is not(None):
            ax.plot(ell,eta2_true_cumul,linestyle='dotted', marker='x',
              color='C0',label='$\eta^2_{cml}'+'(\ell|G={})$ - exact'.format(g))
            ax.plot(anc_gens,R2_true_cumul,linestyle='dotted',marker='x',
              color='C1',label='$R^2_{cml}'+'(g|G={})$ - exact'.format(g))
        
        #plt.xticks(range(g),[str(i) for i in range(1,g+1)])
        plt.xticks(ell)
        
        #ax.set_ylim(ymin=0)
        ax.set_ylabel('Cumulative Proportion of Variance Explained')
        ax.set_xlabel('Subtree-$\ell$ or Generation-$g$')
        #ax.set_title('Cell: '+a[0])
        #tick_max = ax.yaxis.get_ticklocs()[-1]
        ymin, ymax = ax.get_ylim()
        
        ax.legend(loc='best')
        #ax.text(0,var_a_tot,'Total variance')
        #ax.text(0,var_a_expl,'Max explained variance')
        
        #ax2 = ax.twinx()
        #new_ticks = [0,0.2,0.4,0.6,0.8,1]
        #ax2.yaxis.set_ticks((new_ticks*var_a_tot-ymin)/(ymax-ymin))
        #ax2.yaxis.set_ticklabels(new_ticks)
        #ax2.set_ylabel('Proportion of Variance Explained by Ancestors')
        
        plt.tight_layout()
        if output=='pdf':
            plt.savefig('explvariance'+'.pdf',format='pdf')
        if output=='eps':
            plt.savefig('explvariance'+'.eps',format='eps')
        
        
class simData(object):
    """Simulate data that has the symmetry and sparsity structure of a binary
    tree """
    
    def makeX(self, n_gens=4, n_samples=19, prob=0.8, missing=0.2, seed=0):
        """Generate synthetic pedigree data with imposed covariance structure
        that respects the exchangeability of certain cells (i.e. variables) 
        
        n_gens    = number of generations
        n_samples = number of pedigrees
        prob      = probability of transferance for 1 relationship step
        missing   = fraction of missing data
        """
        
        """Vector of integer cell names (=variable names, or features)"""
        cint = np.arange(1,2**n_gens)      
        """Convert cell names to binary form to capture ancestral history"""   
        cbin = [str(bin(i))[2:] for i in cint] 
        """Vector giving the generation of each cell_id"""
        gens = np.log2(cint).astype(int)+1     

        """Build vector of means for each cell variable"""
        mu_true = np.zeros(np.shape(gens))#gens/gens #1/gens**0.5

        """Get covariance pattern and create patterned covariance matrix """
        cov_pattern, cov = make_cov_pattern(cbin, h=prob, sim=True)
        
        cov_true = cov.astype(float)
        prec_true = linalg.pinv(cov_true)

        """Check that the covariance matrix is positive definite"""
        if np.all(np.linalg.eigvals(cov_true) > 0):
            pass
            #print('Matrix is positive definite')
        else:
            print('simData: Matrix is NOT positive definite')
        
        #print('Cov pattern');      print(cov_pattern)
        #print('Distance matrix');  print(dist)
        #print('Kinship matrix');   print(kin)
        
        sig_true  = np.sqrt(np.diag(cov_true))
        corr_true = cov_true/np.outer(sig_true,sig_true)
        
        psig_true = 1/np.sqrt(np.diag(prec_true))
        pcorr_true = -prec_true * np.outer(psig_true,psig_true)
        np.fill_diagonal(pcorr_true,1.0)
        
        """Generate random dataset"""
        prng = np.random.RandomState(seed)
        data = prng.multivariate_normal(mu_true, cov_true, size=n_samples)
        
        """ Introduce missing data """
        mcarID = prng.rand(n_samples,len(cbin)) > 1-missing
        data[mcarID] = np.nan
        
        plabels = cbin
        self.X = pd.DataFrame(data, columns=plabels)
        
        self.cov_pattern = cov_pattern
        
        self.cov_true = pd.DataFrame(cov_true, index=cbin, columns=cbin)


class wormData(object):
    """ Container for incorporating worm data into our pedigree format """
    
    def __init__(self):
        self.direc = './Worm-WT/'
        self.map_file = './Worm-WT/WormWeb-LineageMap.json'
        self.id_to_binary_file = "./Worm-WT/map_id_to_binary.json"  
        self.name_to_binary_file = "./Worm-WT/map_name_to_binary.json"
    
    def create_worm_map(self):
        """Creates mapping files that relate the WormWeb.org cell id
        or name, to a de-identified binary name (e.g. '1100') """
        
        jdat = json.load(open(self.map_file))
    
        G_json = nx.readwrite.json_graph.tree_graph(jdat)
        # In the json file the key 'name' refers to the common name of
        # the cell (I think). In pydot, 'name' refers to the name of the node.
        # This causes problems with plotting when the 'name' identifier in
        # the json file is not unique. Generate a new graph to clean this up
        # Non-unique names appear at gen9 and above.
        G_id = nx.DiGraph()
        for i,j in G_json.edges():
            G_id.add_edge(i,j)
        # common_name is an attribute.
        for i in G_id.nodes():
            G_id.node[i]['common_name'] = G_json.node[i]['name']
            
        print("Graph is a tree: ");print(nx.is_tree(G_id))
        #print(Gworm.node['abarpaapp']['common_name']) #abarpaapp
        #print(Gworm.node['abarpaapaa']['common_name'])
    
        """Label each node with its generation, found from shortest 
        distance to root"""
        dist_dict = nx.shortest_path_length(G_id,'p0')
        for name in dist_dict:
            G_id.node[name]['gen'] = dist_dict[name] + 1
        
        """Prune nodes that are too distant from the root node"""
        #for name in G_id.nodes():
        # Should not be lower than >8 to avoid duplicate names
        #    if G_id.node[name]['gen'] > 8: 
        #        G_id.remove_node(name)
    
        """Check to see when duplicate common names start to appear (gen9)"""
        #id_to_name = {i:G_id.node[i]['common_name'] for i in G_id.nodes()}
        #print(len(set(id_to_name.keys())),len(set(id_to_name.values())))
        
        """ Recursively add binary labels"""
        def add_binary_labels(G,parent):
            children = G.successors(parent)
            #if len(children) > 0: #remove for networkx 2.0
            for i,child in enumerate(children):
                G.node[child]['binary_label'] = \
                    G.node[parent]['binary_label']+str(i)
                add_binary_labels(G,child)
        
        G_id.node['p0']['binary_label'] = '1'
        add_binary_labels(G_id,'p0')
        
        """Create dict to map from id to binary and from name to binary"""
        id_to_binary = {i: G_id.node[i]['binary_label'] for i in G_id.nodes()}
        name_to_binary = {G_id.node[i]['common_name']: 
                      G_id.node[i]['binary_label'] for i in G_id.nodes()}
        
        print(len(name_to_binary.values()),len(set(name_to_binary.values())))
        print(name_to_binary['P1'],id_to_binary['p1'])
    
        """Dump a map connecting worm names to binary labels"""
        json.dump(id_to_binary,open(self.id_to_binary_file,'w'))
        json.dump(name_to_binary,open(self.name_to_binary_file,'w'))
        
        print('Mapping files have been written')
        
        """Create temporary network to visualize & check the label mapping"""
        #label_map_temp = {i: i+'\n'+G_id.node[i]['binary_label'] 
        #    for i in G_id.nodes()}
        #nx.relabel_nodes(G_id,label_map_temp,copy=False)
        #p=nx.drawing.nx_pydot.to_pydot(G_id)
        #p.write_pdf('junk.pdf')
        
        #pos = nx.drawing.nx_pydot.graphviz_layout(Gworm, prog='twopi', args='')
        #nrows, ncols = 1, 1
        #fig = plt.figure(figsize=(8,8))
        #gs = plt.GridSpec(nrows,ncols)
        #ax = fig.add_subplot(gs[0,0])
        #plt.axis('off')
        #nx.draw_networkx(Gworm,pos,node_size=50,alpha=0.2,with_labels=True) 
    
    def load_worm_data(self,imarker=0):
        """ Loads worm data into a df"""

        # imarker is an index 0, 1 or 2 that corresponds to a particular marker
        # 0: RW10348 nhr-25 (hypodermal)
        # 1: RW10425 pha-4 (pharynx)
        # 2: RW10434 cnd-1 (neuronal)

        marker_map = {0:'10348',1:'10425',2:'10434'}
        marker = marker_map[imarker]
        
        #Load file names with that marker
        files = glob.glob(self.direc+'info_ZD_RW'+marker+'*.txt')
        print("Number of files loaded = ", str(len(files)))
        #0th file: info_ZD_RW10425_WT_20110428_3_s1_emb3_edited.txt
        
        """ Load the mapping dictionaries from cell_name and cell_id to binary.
        The worm data uses a mixture of both so we'll need some logic to do 
        the mapping correctly."""
        name_to_binary = json.load(open("./Worm-WT/map_name_to_binary.json"))
        id_to_binary = json.load(open("./Worm-WT/map_id_to_binary.json"))
        
        """If files is a single file, as a string, convert it to a list"""
        if type(files) == str:
            files = [files]
        
        """Build a mapping from binary_label back to cellname"""
        binary_to_cellname = {'1':'P0'} #start with P0 to make sure it appears 
            
        """ Loop through each file"""
        for j,file in enumerate(files):
            """Read the file: (cell_name,average_expression)"""
            dfj=pd.read_csv(file, delimiter='\t', usecols=[0,1])
            
            """Change the 'average_expression' column to a pedigree number"""
            dfj.rename(columns={'average_expression':'ped'+str(j)},inplace=True)
            
            """Now loop through each cell, convert its label to binary, and
            record the data"""
            lost = [] #record the cells that do cannot be found in the map
            for i in dfj.index:
                try: #first try mapping with the cell id (avoids duplicates)
                    binary_label = id_to_binary[dfj.loc[i,'cell_name'].lower()]
                    dfj.loc[i,'binary_label'] = binary_label
                    #dfj.loc[i,'binary_label_int'] = int(binary_label)
                    #dfj.loc[i,'gen'] = len(binary_label)
                    binary_to_cellname[binary_label] = \
                        dfj.loc[i,'cell_name']
                    #'P0' comes from here
                except:
                    try: #next try mapping with the cell name
                        binary_label = name_to_binary[dfj.loc[i,'cell_name']]
                        dfj.loc[i,'binary_label'] = binary_label
                        #dfj.loc[i,'binary_label_int'] = int(binary_label)
                        #dfj.loc[i,'gen'] = len(binary_label)
                        binary_to_cellname[binary_label]=dfj.loc[i,'cell_name']
                    except:
                        lost.append(dfj.loc[i,'cell_name'])

            if len(lost) > 0:
                print("These cells were not assigned a binary variable:")
                print(lost)
            
            dfj.dropna(axis=0,inplace=True)
            #dfj.sort_values('binary_label_int',axis=0,inplace=True)
            #del dfj['binary_label_int']
            
            """Identify any duplicate binary labels"""
            vc = dfj['binary_label'].value_counts()
            duplicates = vc[vc > 1].index.tolist()
            for d in duplicates:
                print(dfj[dfj['binary_label'] == d])
            
            """ Merge/join dfj with the cumulative df"""
            if j == 0:
                df_cuml = dfj.loc[:,['binary_label','ped'+str(j)]]
            else:
                dfj = dfj.loc[:,['binary_label','ped'+str(j)]]
                #Use full outer join - union of binary labels from both df's
                df_cuml = pd.merge(df_cuml,dfj,on='binary_label',how='outer')

        """Sort binary labels now that all cell positions have been recorded"""
        df_cuml['binary_label_int'] = df_cuml['binary_label'].apply(int)
        df_cuml.sort_values('binary_label_int', axis=0, inplace=True)
        del df_cuml['binary_label_int']   
              
        print('Cell positions with nans: ',
            pd.isnull(df_cuml).any(1).nonzero()[0])
        
        """Organize as tidy data. This will be unbalanced since the 
        worm lineage is unbalanced beyond gen6."""
        self.df_cuml = df_cuml.set_index('binary_label').T
        
        """ Binary to cellname map"""
        self.binary_to_cellname = binary_to_cellname
        self.id_to_binary = id_to_binary
        self.name_to_binary = name_to_binary
        
        """Standardize data. This is needed since the data values are large
        and seem to cause errors in the covariance estimation procedure."""
        #mu = self.df_cuml.mean().mean()
        #std = self.df_cuml.std().std()
        #self.df_cuml = (self.df_cuml-mu)/std
        #self.df_cuml = self.df_cuml/std
        
        """HACK to deal with imarker=2 pathology of only one founder point"""
        if imarker == 2:
              self.df_cuml.loc['ped3','1'] = np.nan


    def balance_pedigree(self,gmax=5,gmin=1):
        """Creates a df with balanced cell positions up to gmax """
        
        """ Create list of all cell id's within given generations"""
        # Integer id's, e.g. [1,2,3,4,5,6,7]
        cint = range(int(2**(gmin-1)),int(2**gmax)) 
        
        """Convert to binary id's: ['1','10','11','100','101','110','111']"""
        cbin  = [str(bin(i))[2:] for i in cint] 
        cells = cbin
    
        """Populate new df, which has a balanced list of cell id's, w/ data"""
        self.X = pd.DataFrame(index=self.df_cuml.index, columns=cells) 
        self.X.update(self.df_cuml) #fill nan's with data where available
        
        
        """Standardize data; more stable if the selected subset of data is
        standardised, rather than using the original units
        WHY DO WE NEED TO DO THIS TO STABILISE THE EM?
        CHECK TO UNDERSTAND THIS BETTER"""
        #mu = self.X.mean().mean()
        #sigma = self.X.std().std() #this works well
        sigma = 1000 #use this to keep units consistent between gens
        #self.X = (self.X - mu)/sigma
        self.X = self.X/sigma
        
        """Generate 3-index labels for each covariance matrix element"""  
        
        self.cov_pattern = make_cov_pattern(cells)
        


def plot_radial_lineage(obj,index=[0,1,2,3,4],
    show_labels=False,output=None):
    """Plot a few sample lineages radially
    obj = lineage object such as worm or sim or pm
    index = which of the families to select"""
    
    
    nrows, ncols = 1, 5
    fig = plt.figure(figsize=(8,2)) #(w,h) in inches
    #nrows, ncols = 1, 1
    #fig = plt.figure(figsize=(8,8)) #(w,h) in inches
    
    gs = plt.GridSpec(nrows,ncols)
    gs.update(wspace=0., hspace=0.) 

    """ Establish colormap scaling """
    vmin = obj.X.iloc[index,:].min().min()
    vmax = obj.X.iloc[index,:].max().max()

    G = nx.Graph()

    for inum,i in enumerate(index):
        ped = obj.X.iloc[i,:] #select the family
        #ped = obj.X.mean() # average
        
        for i in ped.index:
            G.add_node(i)
            G.node[i]['x'] = ped[i] # fill node with the expression level
            if i != '1':
                G.add_edge(i[:-1],i) #add edge between node i and its parent
    
        """create dict of node positions in radial tree, {node:(x,y)}"""
        pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='twopi', args='') 
        node_list = G.nodes()
        nodecolor_list = [G.node[i]['x'] for i in node_list]
        
        """ if wormData object, then create a nodelabel dict with cell id's"""
        if hasattr(obj, 'binary_to_cellname'):
            nodelabel_dict = {}
            for i in G.nodes():
                try:
                    nodelabel_dict[i] = obj.binary_to_cellname[i] #{'1':'P0'}
                except:
                    nodelabel_dict[i] = 'x'
        else:
            nodelabel_dict = None

        """Print list of node values to check worm data """
        #for z1,z2 in zip(node_list,nodecolor_list):
        #    print(nodelabel_dict[z1],z2)
        
        gmax = max([len(id) for id  in ped.index])
        ax = fig.add_subplot(gs[0,inum],aspect='equal')
        
        ax.axis('off')
        ax.set_xticklabels([]) #remove ticklabels which otherwise take up space
        ax.set_yticklabels([])
        
        cmap = plt.get_cmap('viridis')
        nx.draw_networkx(G, pos, ax=ax, cmap=cmap, with_labels=show_labels, 
            node_size=-13*(gmax-5)+50*1, #crowded trees have smaller nodes
            linewidths=0, width=0.5, #linewidths=node border; width=edge width
            alpha=0.5, font_size=8, nodelist = node_list, 
            node_color=nodecolor_list, labels=nodelabel_dict,
            vmin=vmin, vmax=vmax)
    

    pnorm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=pnorm)
    sm._A = []
    cbar = fig.colorbar(sm,ax=ax, fraction=0.035, pad=0.0)
    cbar.ax.tick_params(labelsize=8)
    gs.tight_layout(fig)
    
    if output=='pdf': #bbox_inches='tight'
        plt.savefig('samples_'+'.pdf',format='pdf',bbox_inches='tight')

    return
    
def plot_vs_gen(lineage_obj,gmax=None,gmin=None,output=None):
    """Plots the response variable as a function of gen for all families
    lineage_obj = pm or sim or worm"""
    
    X = lineage_obj.X.copy()
    
    """Drop a family member if all of them are nan's"""
    X.dropna(axis=1,inplace=True,how='all')
    
    """Create list of generations that have data"""
    gen_list = [len(id) for id in X.columns] 
    
    #X.columns = pd.MultiIndex.from_tuples(
    #    [(id,g) for id,g in zip(X.columns,gen_list)])
    X.columns = gen_list
    X.columns.name = 'gen'
    X.index.name   = 'ped'

    if gmax == None:
        gmax = max(gen_list)
    if gmin == None:
        gmin = min(gen_list)

    X = X.loc[:,gmin:gmax]

    """List of pedigree names"""
    ped_list= X.index.tolist()  

    """Move generations predictor from columns to index"""
    X = X.stack()

    """Calculate relative plot displacement for each pedigree """    
    npeds = float(len(ped_list))
    if npeds > 1:
        spread = 0.35 #spread on either side of gen value
        delta = 2*spread/(npeds-1)
    else:
        spread = 0.
        delta = 0.
    
    ped_order = {ped: i for i,ped in enumerate(ped_list)}
    
    nrows, ncols = 1, 1
    fig = plt.figure(figsize=(5,5)) #(w,h) in inches
    gs = plt.GridSpec(nrows,ncols)
    ax = fig.add_subplot(gs[0,0])
    
    for i in ped_list:
        x_ped = X.xs(i,level='ped')
        gen = x_ped.index
        """Displace generation position to avoid pedigree overlap """
        gen_pos = gen - spread + ped_order[i]*delta

        ax.plot(gen_pos, x_ped.values, label=i, alpha=0.8, marker='o',
                  markeredgewidth=0.1, markersize=4, linestyle='None')
    
    """Vertical lines to separate generations"""
    for j in range(gmin-1, gmax+1):
        plt.axvline(x=j+0.5, axes=ax, color='k', linestyle=':')

    ax.set_xlabel('Generation')
    ax.set_ylabel('Phenotype')
    ax.legend(title='Family', 
        bbox_to_anchor=(0.99, 1.02)).get_frame().set_alpha(0.8)
    
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(12)
    
    gs.tight_layout(fig)

    if output=='pdf': #bbox_inches='tight'
        plt.savefig('gentrend_'+'.pdf',format='pdf',bbox_inches='tight')

    return
    
class MidpointNormalize(colors.Normalize):
	"""
	Set the colormap and centre the colorbar
	Normalise the colorbar so that colors diverge either side 
	of a prescribed midpoint value), 
	e.g. im=ax1.imshow(array,
	    norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
	http://chris35wills.github.io/matplotlib_diverging_colorbar/
	"""
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		# I'm ignoring masked values and all kinds of edge cases to make a
		# simple example...
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))	
    
def make_cov_pattern(cbin, h=0.8, sim=False):
    """Create labels to identify unique covariance matrix elements. 
        cbin = list of strings. Each string is a binary number 
            representing a lineal position
        sim = whether or not to generate a branching process cov matrix
        h = if generating a cov matrix, must enter the mother-daughter 
            correlation
        """

    def common_anc(node0, node1):
        """Returns the lowest common ancestor of node0 and node1 
        Uses the fact that a node's lineage is encoded in its name. Thus
        the common ancestor of '10101' and '10110001' is '101' """
        
        z = list(zip(node0, node1))  
        com = ''.join([i[0] for i in itertools.takewhile(
            lambda x: x[0]==x[1], z)])
        
        return com    

    """Covariance matrix labelling each unique cross element"""
    cov_pattern = pd.DataFrame(index=cbin, columns=cbin) 
    
    dist = pd.DataFrame(index=cbin, columns=cbin)
    
    if sim:
        kin = pd.DataFrame(index=cbin, columns=cbin)
        cov = pd.DataFrame(index=cbin, columns=cbin)
    
    for i in cbin:
        for j in cbin:
            mrca = common_anc(i,j) # name of mrca of i and j
            i_gen = len(i) # generation of i
            j_gen = len(j) # generation of j
            m_gen = len(mrca) # generation of mrca of i and j
        
            #swap i and j positions to ensure a symmetric matrix
            if i_gen > j_gen: 
                j_gen, i_gen = i_gen, j_gen
        
            cov_pattern.loc[i,j] = str(i_gen)+str(j_gen)+str(m_gen)
            
            if sim:
                dist.loc[i,j] = abs(i_gen - m_gen) + abs(j_gen - m_gen)
                kin.loc[i,j] = h**dist.loc[i,j]
                cov.loc[i,j] = kin.loc[i,j]
        
    if sim:
        return cov_pattern, cov
    else:
        return cov_pattern

def X_from_excel(file, gmax=5):
    """ Reads pedigree data and outputs a balanced table, 
    where rows are pedigrees and columns are lineal positions. 
    gmax is the maximum allowed number of generations"""  
    
    X_raw = pd.read_excel(file, index_col=[0,1])
    X_raw = X_raw.unstack()
    X_raw.columns = ['{}'.format(x[1]) for x in X_raw.columns]
    
    """ Create list of all cell id's within given generations
    Integer id's, e.g. [1,2,3,4,5,6,7]"""
    gmin = 1 #always start at generation 1
    cint = range(int(2**(gmin-1)),int(2**gmax)) 
        
    """Convert to binary id's: ['1','10','11','100','101','110','111']"""
    cbin  = [str(bin(i))[2:] for i in cint] 
    cells = cbin
    
    """Populate new df, which has a complete list of cell id's, w/ data"""
    X = pd.DataFrame(index=X_raw.index, columns=cells) 
    X.update(X_raw) #fill nan's with data where available

    return X
    
