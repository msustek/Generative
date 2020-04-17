from toydata import *
from bokeh.plotting import figure

IMAGE_STEP = 0.02
SIZE = 10.0
XMIN, XMAX = YMIN, YMAX = -SIZE, SIZE


def show_distr_and_samples(Z, sample_x, sample_y, pdf_scale = "log", fig = None, draw = True, title = None):
    if fig is None:
        
        fig = figure(x_range=(XMIN, XMAX), y_range=(YMIN, YMAX), width=400, height=400, match_aspect=True,
               tooltips=[("x", "$x"), ("y", "$y"), ("pdf", "@image")], title=title)
        
        #figure(width=500, height=500, match_aspect=True,
        #       tooltips=[("x", "$x"), ("y", "$y"), ("pdf", "@image")], plot_width=700, plot_height=700)
    
    if Z is not None:
        Z = Z.reshape(grid_size,-1)
        if pdf_scale == "log":
            Z = np.log(Z + 1e-20)
        fig.image(image=[1-Z if INV_COL else Z], x=XMIN, y=YMIN, dw=XMAX-XMIN, dh=YMAX-YMIN, palette=PAL)

    if sample_x is not None and sample_y is not None:
        fig.circle(sample_x, sample_y, size=2, line_color="red", fill_alpha=0.8)
    if draw:
        show(fig)
    return fig

def get_distr_and_samples(distr_type = "Banana", draw_pdf = True, draw_samples = True):
    print(distr_type)
    Z = sample_x = sample_y = None

    
    # Gauss
    if distr_type == "gauss":
        mean = np.array([-2, 2])
        cov = np.array([[1, 0.7], [0.7, 1]])  

        if draw_pdf:
            Z = pdf_gauss(np.vstack([X.ravel(),Y.ravel()]).T,mean,cov)
        if draw_samples:
            sample_x, sample_y = sample_gauss(N_SAMPLES, mean, cov)

    # GMM
    elif distr_type == "GMM":
        weights = [0.9, 0.1]
        means = np.array([[1.0, 0.0],
                          [3.0, 1.7]])
        covs = np.array([[[1.0,  0.7], [ 0.7, 1.0]],
                         [[0.5, -0.2], [-0.2, 0.1]]])

        if draw_pdf:
            Z = pdf_gmm(np.vstack([X.ravel(),Y.ravel()]).T, weights, means, covs)
        #Z = pdf_mm(np.vstack([X.ravel(),Y.ravel()]).T, weights, pdf_gauss, [[mean, cov] for mean, cov in zip(means,covs)])    
        if draw_samples:
            sample_x, sample_y = sample_gmm(N_SAMPLES, weights, means, covs)
            #sample_x, sample_y = sample_mm(N_SAMPLES, weights, sample_gauss, [[mean, cov] for mean, cov in zip(means,covs)])


    elif distr_type == "square":
        size = 2
        var = 0.25
        width = 13
        height = 13
        covs = np.array([[var,  0.0], [ 0.0, var]])
        
        if draw_pdf:
            Z = pdf_square_gauss(np.vstack([X.ravel(),Y.ravel()]).T, size, width, height, covs)
        if draw_samples:
            sample_x, sample_y = sample_square_gauss(N_SAMPLES, size, width, height, covs)

            
    elif distr_type == "banana":
        mu_x, mu_y, var_x = 0, 0, 2 # mean and standard deviation for x distribution
        var_y_ratio = 1.0 / 25
        BAN_LEN = 0.5 # (BAN_LEN * |x|^BAN_CURV) => how much curved the banana will be
        BAN_CURV = 2
        
        if draw_pdf:
            Z = pdf_banana_grid(X, Y, mu_x, mu_y, var_x, var_y_ratio, BAN_LEN, BAN_CURV)
        #Z = pdf_banana(np.vstack([X.ravel(),Y.ravel()]).T, mu_x, mu_y, var_x, var_y_ratio, BAN_LEN, BAN_CURV)
        if draw_samples:
            sample_x, sample_y = sample_banana(N_SAMPLES, mu_x, mu_y, var_x, var_y_ratio, BAN_LEN, BAN_CURV) 

            
    elif distr_type == "2Bananas":    
        dist_betw = 3
        mu_x, mu_y, var_x = 0, 0, 2 # mean and standard deviation for x distribution
        var_y_ratio = 1.0 / 16
        BAN_LEN = 0.3 # (BAN_LEN * |x|^BAN_CURV) => how much curved the banana will be
        BAN_CURV = 1.5
        
        if draw_pdf:        
            Z = pdf_2bananas_grid(X, Y, dist_betw, mu_x, mu_y, var_x, var_y_ratio, BAN_LEN, BAN_CURV)
        
        #Z = pdf_2bananas(np.vstack([X.ravel(),Y.ravel()]).T, mu_x, var_x, var_y_ratio, BAN_LEN, BAN_CURV)
        if draw_samples:
            sample_x, sample_y = sample_2bananas(N_SAMPLES, dist_betw, mu_x, mu_y, var_x, var_y_ratio, BAN_LEN, BAN_CURV) 

            
    elif distr_type == "K-2Bananas":  
        K = 2
        width = 13
        height = 13
        dist_betw = 3
        mu_x, mu_y, var_x = 0, 0, 2 # mean and standard deviation for x distribution
        var_y_ratio = 1.0 / 16
        BAN_LEN = 0.3 # (BAN_LEN * |x|^BAN_CURV) => how much curved the banana will be
        BAN_CURV = 1.5
        
        if draw_pdf:
            Z = pdf_kbananas_grid(X, Y, K, width, height, dist_betw, mu_x, mu_y, var_x, var_y_ratio, BAN_LEN, BAN_CURV)
#        Z = pdf_kbananas(np.vstack([X.ravel(),Y.ravel()]).T, K, width, height, dist_betw, mu_x, mu_y, var_x, var_y_ratio, BAN_LEN, BAN_CURV)        
        if draw_samples:
            sample_x, sample_y = sample_kbananas(N_SAMPLES, K, width, height, dist_betw, mu_x, mu_y, var_x, var_y_ratio, BAN_LEN, BAN_CURV) 

    
    return Z, sample_x, sample_y
            

if __name__ == "__main__":
    #DISTR = "gauss"
    #DISTR = "GMM"
    #DISTR = "square"
    DISTR = "banana"
    #DISTR = "2Bananas"
    #DISTR = "K-2Bananas"

    PDF = True
    #PDF = False
    #PDF_scale = "log"
    PDF_scale = "normal" # can be anything - it reacts only on "log"

    SAMPLING = True
    #SAMPLING = False
    N_SAMPLES = 50

    if False:
        INV_COL = False
        PAL = "Viridis256"
        #PAL = "Inferno256"
    else:
        INV_COL = True
        PAL = "Greys256"

    x = np.arange(XMIN, XMAX, IMAGE_STEP)
    y = np.arange(YMIN, YMAX, IMAGE_STEP)
    grid_size = len(y)
    X, Y = np.meshgrid(x, y)
    show_distr_and_samples(*get_distr_and_samples(DISTR, PDF, SAMPLING), PDF_scale)