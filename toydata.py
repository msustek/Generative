import numpy as np
N_SAMPLES = 50

def generate_ds_gauss():
    mean = np.array([-2, 2])
    cov = np.array([[1, 0.7], [0.7, 1]])  
    return sample_gauss(N_SAMPLES, mean, cov)


def generate_ds_square():
    size = 2
    var = 0.25
    width = 13
    height = 13
    covs = np.array([[var,  0.0], [ 0.0, var]])
    return sample_square_gauss(N_SAMPLES, size, width, height, covs)

def generate_ds_banana():
    mu_x, mu_y, var_x = 0, 0, 2 # mean and standard deviation for x distribution
    var_y_ratio = 1.0 / 25
    BAN_LEN = 0.5 # (BAN_LEN * |x|^BAN_CURV) => how much curved the banana will be
    BAN_CURV = 2
    return sample_banana(N_SAMPLES, mu_x, mu_y, var_x, var_y_ratio, BAN_LEN, BAN_CURV) 




# Gauss
def pdf_gauss(x, mu, cov):
    assert(mu.ndim == 1 and len(mu) == len(cov) and (cov.ndim == 1 or cov.shape[0] == cov.shape[1]))
    x = np.atleast_2d(x) - mu
    return np.exp(-0.5*(len(mu)*np.log(2 * np.pi) + np.linalg.slogdet(cov)[1] + np.sum(x.dot(np.linalg.inv(cov)) * x, axis=1)))

def sample_gauss(N, mu = np.array([0,0]), cov = np.eye(2)):
    return np.random.multivariate_normal(mu, cov, N).T


# GMM
def pdf_gmm(x, weights, mus, covs):
    assert(abs(sum(weights) - 1.0) < 1e-3)
    assert(len(weights) == len(mus) and len(mus) == len(covs))
    res = np.zeros(len(x))    
    for w, mu, cov in zip(weights,mus,covs):
        res += w * pdf_gauss(x, mu, cov)
    return res

def sample_gmm(N, weights, mus, covs):
    assert(abs(sum(weights) - 1.0) < 1e-3)
    assert(len(weights) == len(mus) and len(mus) == len(covs))
    samples = []
    for N_per_component, mu, cov in zip(np.random.multinomial(N, weights), mus,covs):
        samples.append(sample_gauss(N_per_component, mu, cov))
    samples = np.hstack(samples).T
    np.random.shuffle(samples)
    return samples.T


# Mixture
def pdf_mm(x, weights, pdf_funs, argss):
    assert(abs(sum(weights) - 1.0) < 1e-3)
    if callable(pdf_funs):
        pdf_funs = [pdf_funs]*len(weights)
    assert(len(weights) == len(argss) and len(pdf_funs) == len(argss))
    res = np.zeros(len(x))    
    for w, pdf_fun, args in zip(weights,pdf_funs, argss):
        res += w * pdf_fun(x, *args)
    return res 
    
def sample_mm(N, weights, sample_funs, argss):
    assert(abs(sum(weights) - 1.0) < 1e-3)
    if callable(sample_funs):
        sample_funs = [sample_funs]*len(weights)
    assert(len(weights) == len(argss) and len(argss) == len(sample_funs))  
    
    samples = []
    for N_per_component, sample_fun, args in zip(np.random.multinomial(N, weights), sample_funs, argss):
        samples.append(sample_fun(N_per_component, *args))
    samples = np.hstack(samples).T
    np.random.shuffle(samples)
    return samples.T


# Square Gauss
def check_for_square_gauss(size, covs, weights):
    comp = size*size
    if covs is None:
        covs = np.array([[[0.25,  0.0], [ 0.0, 0.25]]]*comp)
    elif covs.ndim == 2:
        covs = np.array([covs]*comp)
    if weights is None:
        weights = [1.0/comp]*comp
    assert(len(covs) == comp and comp == len(weights))
    assert(abs(sum(weights) - 1.0) < 1e-3)    
    return covs, weights

def get_means_for_square(size, width, height, bias_x, bias_y):
    dx = width / max(1,size-1)
    dy = height / max(1,size-1)
    start_x = -(size - 1.0)/2 * dx + bias_x
    start_y = -(size - 1.0)/2 * dy + bias_y
    m_x, m_y = np.meshgrid(np.arange(start_x, start_x + size*dx, dx), np.arange(start_y, start_y + size*dy, dy))
    return np.vstack([m_x.ravel(),m_y.ravel()]).T


def pdf_square_gauss(x, size = 2, width = 13, height = 13, covs = None, weights = None, bias_x = 0, bias_y = 0):
    covs, weights = check_for_square_gauss(size, covs, weights)    
    means = get_means_for_square(size, width, height, bias_x, bias_y)
    return pdf_gmm(x, weights, means, covs)
    

def sample_square_gauss(N, size = 2, width = 13, height = 13, covs = None, weights = None, bias_x = 0, bias_y = 0):
    covs, weights = check_for_square_gauss(size, covs, weights)    
    means = get_means_for_square(size, width, height, bias_x, bias_y)
    return sample_gmm(N, weights, means, covs)
 
    
    
# Banana
def get_banana_params(mu_x, mu_y, var_x, var_y_ratio, BAN_LEN, BAN_CURV, mid_x):
    if mid_x is None:
        mid_x = mu_x # middle of banana shape <=> can be different 
    cov = np.array([[var_x,  0.0], [ 0.0, var_x * var_y_ratio]]) # I expect them to be decorelated -> can be changed
    return mid_x, np.array([mu_x, mu_y]), cov

# (BAN_LEN * |x|^BAN_CURV) => how much curved the banana will be
def pdf_banana_grid(X,Y, mu_x = 0, mu_y = 0, var_x = 2, var_y_ratio = 0.04, BAN_LEN = 0.5, BAN_CURV = 1.5, mid_x = None, direction = 1):
    mid_x, mean, cov = get_banana_params(mu_x, mu_y, var_x, var_y_ratio, BAN_LEN, BAN_CURV, mid_x)

    POINTS = np.vstack([X.ravel(),Y.ravel()])
    POINTS[1] = POINTS[1] - direction*BAN_LEN*np.array([np.power(np.abs(X[0,:]-mid_x),BAN_CURV)]*X.shape[0]).ravel()
    return pdf_gauss(POINTS.T,mean,cov) 

# this is slower than grid variant
def pdf_banana(points, mu_x = 0, mu_y = 0, var_x = 2, var_y_ratio = 0.04, BAN_LEN = 0.5, BAN_CURV = 1.5, mid_x = None, direction = 1):
    mid_x, mean, cov = get_banana_params(mu_x, mu_y, var_x, var_y_ratio, BAN_LEN, BAN_CURV, mid_x)
    def f(x,y):
        return x,y - direction*BAN_LEN*np.power(np.abs(x-mid_x),BAN_CURV)
    f = np.vectorize(f)
    p = np.array(f(points.T[0],points.T[1])).T
    return pdf_gauss(p,mean,cov) 


def sample_banana(N, mu_x = 0, mu_y = 0, var_x = 2, var_y_ratio = 0.04, BAN_LEN = 0.5, BAN_CURV = 1.5, mid_x = None, direction = 1):
    mid_x, mean, cov = get_banana_params(mu_x, mu_y, var_x, var_y_ratio, BAN_LEN, BAN_CURV, mid_x)
    
    sample_x, sample_y = sample_gauss(N, mean, cov)
    sample_y = sample_y + direction*BAN_LEN*np.power(np.abs(sample_x-mid_x),BAN_CURV)
    return sample_x, sample_y
    

# 2 Bananas
def pdf_2bananas_grid(X, Y, dist_betw = 6.0, mu_x = 0, mu_y = 0, var_x = 2, var_y_ratio = 0.04, BAN_LEN = 0.5, BAN_CURV = 1.5, mid_x = None):
    Z1 = pdf_banana_grid(X, Y, mu_x, mu_y - dist_betw/2.0, var_x, var_y_ratio, BAN_LEN, BAN_CURV, mid_x, 1)
    Z2 = pdf_banana_grid(X, Y, mu_x, mu_y + dist_betw/2.0, var_x, var_y_ratio, BAN_LEN, BAN_CURV, mid_x, -1)
    return 0.5*Z1 + 0.5*Z2

# this is slower than grid variant
def pdf_2bananas(points, dist_betw, mu_x = 0, mu_y = 0, var_x = 2, var_y_ratio = 0.04, BAN_LEN = 0.5, BAN_CURV = 1.5, mid_x = None):
    Z1 = pdf_banana(points, mu_x, mu_y - dist_betw/2.0, var_x, var_y_ratio, BAN_LEN, BAN_CURV, mid_x, 1)
    Z2 = pdf_banana(points, mu_x, mu_y + dist_betw/2.0, var_x, var_y_ratio, BAN_LEN, BAN_CURV, mid_x, -1)
    return 0.5*Z1 + 0.5*Z2


def sample_2bananas(N, dist_betw, mu_x = 0, mu_y = 0, var_x = 2, var_y_ratio = 0.04, BAN_LEN = 0.5, BAN_CURV = 1.5, mid_x = None):
    p1 = [mu_x, mu_y - dist_betw/2.0, var_x, var_y_ratio, BAN_LEN, BAN_CURV, mid_x, 1]
    p2 = [mu_x, mu_y + dist_betw/2.0, var_x, var_y_ratio, BAN_LEN, BAN_CURV, mid_x, -1]
    return sample_mm(N, [0.5, 0.5], sample_banana, [p1,p2])        
    
    
# K-Bananas    

def pdf_kbananas_grid(X, Y, k, width, height, dist_betw = 6.0, mu_x = 0, mu_y = 0, var_x = 2, var_y_ratio = 0.04, BAN_LEN = 0.5, BAN_CURV = 1.5, mid_x = None):
    comp = k*k
    res = np.zeros(X.shape[0]*X.shape[1])
    for x,y in get_means_for_square(k, width, height, 0, 0):
        res += pdf_2bananas_grid(X, Y, dist_betw, mu_x + x, mu_y + y, var_x, var_y_ratio, BAN_LEN, BAN_CURV, mid_x)
    return res

def pdf_kbananas(points, k, width, height, dist_betw, mu_x = 0, mu_y = 0, var_x = 2, var_y_ratio = 0.04, BAN_LEN = 0.5, BAN_CURV = 1.5, mid_x = None):
    comp = k*k
    return pdf_mm(points, [1.0/comp]*comp, pdf_2bananas, [[dist_betw, mu_x + x, mu_y + y, var_x, var_y_ratio, BAN_LEN, BAN_CURV, mid_x] for x,y in get_means_for_square(k, width, height, 0, 0)])


def sample_kbananas(N, k, width, height, dist_betw, mu_x = 0, mu_y = 0, var_x = 2, var_y_ratio = 0.04, BAN_LEN = 0.5, BAN_CURV = 1.5, mid_x = None):
    comp = k*k
    return sample_mm(N, [1.0/comp]*comp, sample_2bananas, [[dist_betw, mu_x + x, mu_y + y, var_x, var_y_ratio, BAN_LEN, BAN_CURV, mid_x] for x,y in get_means_for_square(k, width, height, 0, 0)])        
    
