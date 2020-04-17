from bokeh.layouts import column
from bokeh.layouts import grid 
from bokeh.models import CustomJS, ColumnDataSource, Slider
from bokeh.plotting import figure, show
from bokeh.models import Legend, LegendItem
from bokeh.io import export_png, export_svgs
import os, torch
import numpy as np
from toydata import sample_gauss
from distr_visual import show_distr_and_samples

def show_loss(train_l, test_l = None):
    fig = figure()
    indices = list(range(1,len(train_l) + 1))
    x = [indices]
    y = [train_l]
    colors = "red"
    
    if test_l is not None:
        x.append(indices)
        y.append(test_l)
        colors = [colors, "blue"]
        
    r = fig.multi_line(x, y, color= colors)       
    legends = [LegendItem(label="Training", renderers=[r], index=0)]
    
    if test_l is not None:
        legends.append(LegendItem(label="Testing", renderers=[r], index=1))
        
    legend = Legend(items=legends)
    fig.add_layout(legend)
    fig.xaxis.axis_label = 'Epoch'
    fig.yaxis.axis_label = 'Loss'
    show(fig)
    
    
def get_detailed_loss(loss, title):
    loss, rec_l, kld_l = loss
    fig = figure(title = title)
    indices = list(range(1,len(loss) + 1))
    x = [indices, indices, indices]
    y = [loss, rec_l, kld_l]
    colors = ["black", "red", "green"]
        
    r = fig.multi_line(x, y, color= colors)       
    legends = [LegendItem(label="Rec + KLD", renderers=[r], index=0)]
    legends.append(LegendItem(label="Reconstr", renderers=[r], index=1))
    legends.append(LegendItem(label="KLD", renderers=[r], index=2))
    
    legend = Legend(items=legends)
    fig.add_layout(legend)
    fig.xaxis.axis_label = 'Epoch'
    fig.yaxis.axis_label = 'Loss'
    return fig

def show_detailed_loss(loss, title = "Training loss"):
    show(get_detailed_loss(loss, title))
    
def myellipse(mean, cov, size = 2, n_points = 100):
    d, v = np.linalg.eigh(cov)
    angles = np.linspace(0,2*np.pi, n_points)
    circle = np.c_[np.cos(angles), np.sin(angles)]
    contour = mean + circle@np.diag(size*np.sqrt(d))@v
    return contour


def get_animation_of_movement(start_points, end_mean, end_var, show_start = True, show_line = True, title=None, el_size=2, fix_figure = True, ret_col = True, show_var_at_start = False):
    n_points_per_el = 100 # each ellipse is constructed by joining this number of points
    
    if end_mean.ndim == 2:
        end_mean = end_mean.reshape(1,*end_mean.shape)
    if end_var.ndim == 2:
        end_var = end_var.reshape(1,*end_var.shape)
    x,y = start_points.T
    # contours of variance -> make it invisible in graph
    x_c, y_c = np.array([x]*n_points_per_el).T.tolist(), np.array([y]*n_points_per_el).T.tolist()
    
    
    if fix_figure:
        fig = show_distr_and_samples(None,None,None, pdf_scale = "normal", draw = False, title=title)
    else:
        fig = figure(width=400, height=400, match_aspect=True, title=title)

    
    cds = {"x_s":x, "y_s":y}    
    for i in range(len(end_mean)):
        x2_mean, y2_mean = end_mean[i].T
        x_d, y_d = x2_mean-x,y2_mean-y    
        contours = []
        for x_,y_,c_ in zip(x,y, end_var[i]):
            contours.append(myellipse([x_,y_],np.diag(c_),el_size,n_points_per_el))
        contours = np.array(contours)
        x_cs, y_cs= contours[:,:,0].tolist(),contours[:,:,1].tolist()
        
        cds[f"x{i}"], cds[f"y{i}"] = x2_mean, y2_mean#x, y
        cds[f"x_l{i}"], cds[f"y_l{i}"] = np.array([x,x]).T.tolist(), np.array([y,y]).T.tolist()
        if show_var_at_start:
            cds[f"x_c{i}"], cds[f"y_c{i}"] = (np.array(x_cs) + np.repeat(np.array(x_d)[:, np.newaxis], n_points_per_el, axis =-1)).tolist(), (np.array(y_cs) + np.repeat(np.array(y_d)[:, np.newaxis], n_points_per_el, axis =-1)).tolist()
        else:
            cds[f"x_c{i}"], cds[f"y_c{i}"] = x_c, y_c
        cds[f"x_cs{i}"], cds[f"y_cs{i}"] = x_cs, y_cs
        cds[f"x_d{i}"], cds[f"y_d{i}"] = x_d, y_d
        
    source = ColumnDataSource(data=cds)
    
    # elipse - variance 
    for i in range(len(end_mean)):
        fig.multi_line(xs=f'x_c{i}', ys=f'y_c{i}', source = source, color = "blue", alpha= 0.2)
        
    # line showing difference
    if show_line:
        for i in range(len(end_mean)):
            fig.multi_line(xs=f'x_l{i}', ys=f'y_l{i}', source = source, color="black", alpha=.4, line_width=1)
    
    # orig data
    if show_start:
        fig.circle('x_s', 'y_s', source = source, size=4, line_color="red", fill_color="red", alpha = 0.5, fill_alpha=0.5)
    
    # reconstructed sample
    for i in range(len(end_mean)):
        fig.circle(f'x{i}', f'y{i}', source=source, size=2, line_color="blue", alpha = 1.)


    slider = Slider(start=0.0, end=1.05, value=(1.05 if show_var_at_start else 1.0), step=.05, title="movement")
    js_string = js_string2 = js_string3 = js_string4 = ""
    for i in range(len(end_mean)):
        for var_name in ["x", "y", "x_l", "y_l", "x_c", "y_c", "x_cs", "y_cs", "x_d", "y_d" ]:
            js_string += f'{var_name}{i} = data["{var_name}{i}"]\n'
            
        js_string2 += f"x{i}[i] = x_s[i] + f*x_d{i}[i]\n"
        js_string2 += f"y{i}[i] = y_s[i] + f*y_d{i}[i]\n"
        js_string2 += f"x_l{i}[i][1] = x{i}[i]\n"
        js_string2 += f"y_l{i}[i][1] = y{i}[i]\n"
            
        js_string3 += f"x_c{i}[i][j] = x_cs{i}[i][j] + f*x_d{i}[i]\n"
        js_string3 += f"y_c{i}[i][j] = y_cs{i}[i][j] + f*y_d{i}[i]\n"
        
        js_string4 += f"x_c{i}[i][j] = x{i}[i]\n"
        js_string4 += f"y_c{i}[i][j] = y{i}[i]\n"


        
    #print(js_string, "\n\n", js_string2, "\n\n",js_string3, "\n\n",js_string4)
    update_curve = CustomJS(args=dict(source=source, slider=slider), code="""
        var data = source.data;
        var f = slider.value;
        var slider_over = f > 1.0 
        if (slider_over){
            f = 1.0
        }
        x_s = data['x_s']
        y_s = data['y_s']
        """ + js_string + """
        
        for (i = 0; i < x_s.length; i++) {
        """ + js_string2 + """
            for (j = 0; j < (x_c0[i]).length; j++) {
                if (slider_over){
        """ + js_string3 + """
                }else{
        """ + js_string4 + """
                }
            }
            //alpha[i] = 1-f
            //alpha_n[i] = f
        }

        // necessary becasue we mutated source.data in-place
        source.change.emit();
    """)
    slider.js_on_change('value', update_curve)
    if ret_col:
        return column(slider, fig)
    else:
        return fig

def introspect_model(model, data, data_ind, n_sampl, device):
    mus_x = []
    vars_x = []
    data_ind = torch.from_numpy(np.array(data_ind)).to(device)
    mu_z, logvar_z = model.encoder(torch.from_numpy(np.array(data)).to(device), data_ind)
    mu_z, logvar_z = mu_z.detach(), logvar_z.detach()
    
    for i in range(n_sampl):
        mu_x, logvar_x = model.decoder(model.reparameterize(mu_z, logvar_z), data_ind) 
        mus_x.append(mu_x.numpy())
        vars_x.append(np.exp(logvar_x.numpy()))
    return mu_z, logvar_z, mus_x, vars_x 

def get_model_introspect(model, tr_data, tr_data_ind, val_data, val_data_ind, device, n_sampl = 3, n_lat_sampl = 300, fix_fig = True, ret_col = True, show_var_at_start = False):
    with torch.no_grad():
        model.eval()

        tr_mu_z, tr_logvar_z, tr_mus_x, tr_vars_x = introspect_model(model, tr_data, tr_data_ind, n_sampl, device)
        val_mu_z, val_logvar_z, val_mus_x, val_vars_x = introspect_model(model, val_data, val_data_ind, n_sampl, device)
            
        z_samples = sample_gauss(n_lat_sampl).T
        d = model.decoder(torch.from_numpy(z_samples), None)
        x_samples = model.reparameterize(*d)
        
        fig1 = get_animation_of_movement(tr_data, tr_mu_z.numpy(), np.exp(tr_logvar_z.numpy()), False, False, title="TRAIN X -> Z - Latent space", fix_figure = fix_fig, ret_col = ret_col, show_var_at_start = show_var_at_start)
        fig2 = get_animation_of_movement(tr_data, np.array(tr_mus_x), np.array(tr_vars_x), title="TRAIN X -> X - Reconstruction", fix_figure = fix_fig, ret_col = ret_col, show_var_at_start = show_var_at_start)
        
        fig3 = get_animation_of_movement(val_data, val_mu_z.numpy(), np.exp(val_logvar_z.numpy()), False, False, title="VAL X -> Z - Latent space", fix_figure = fix_fig, ret_col = ret_col, show_var_at_start = show_var_at_start)
        fig4 = get_animation_of_movement(val_data, np.array(val_mus_x), np.array(val_vars_x), title="VAL X -> X - Reconstruction", fix_figure = fix_fig, ret_col = ret_col, show_var_at_start = show_var_at_start)
        
        fig5 = get_animation_of_movement(z_samples, d[0].numpy(), np.exp(d[1].numpy()), False, False, title="Z -> X - Sampling (mean, var)", fix_figure = fix_fig, ret_col = ret_col, show_var_at_start = show_var_at_start)
        fig6 = get_animation_of_movement(z_samples, x_samples.numpy(), np.zeros_like(x_samples), False, False, title="Z -> X - Sampling (samples)", fix_figure = fix_fig, ret_col = ret_col, show_var_at_start = show_var_at_start)
        return fig1, fig2, fig3, fig4, fig5, fig6
        
        
def show_model_introspect(model, tr_data, tr_data_ind, val_data, val_data_ind, device, n_sampl = 3, n_lat_sampl = 300, fix_fig = True):
    fig1, fig2, fig3, fig4, fig5, fig6 = get_model_introspect(model, tr_data, tr_data_ind, val_data, val_data_ind, device, n_sampl, n_lat_sampl, fix_fig)
    show(grid([[fig1,fig2], [fig3, fig4], [fig5, fig6]]))
    
    

def save_fig(fig, directory, name, sub_figs = None):
    export_png(fig, filename=os.path.join(directory, name + ".png"))
    if sub_figs is None:
        fig.output_backend = "svg"
        export_svgs(fig, filename=os.path.join(directory, name + ".svg"))
    else:
        # Bokeh does not support saving plot with more figures in svg format
        for i, sf in enumerate(sub_figs):
            sf.output_backend = "svg"
            export_svgs(sf, filename=os.path.join(directory, name + str(i+1) + ".svg"))
    
