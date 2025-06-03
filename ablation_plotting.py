import numpy
import torch
from CLA.utils.norm_functions import exp_map_normalize, stereo_normalize
from torch.nn.functional import normalize
import plotly.express as px
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
from scipy.stats import beta

cifar10_labels = {
    1: "Airplane",
    2: "Car",
    3: "Bird",
    4: "Cat",
    5: "Deer",
    6: "Dog",
    7: "Frog",
    8: "Horse",
    9: "Ship",
    10: "Truck"
}

def plot_on_sphere_animation(
        points_on_sphere,
        targets,
        marker_size = 1
        ):
    # ponts_on_shpere.shape = epoch_dim, sample_n_dim, vector_dim
    # the epoch_dim is used for the frames
    # the sample_n_dim equals the number of points
    # the vector_dim equals 3 for a 3d coordinate system

    epoch_idx = []
    concat_out = []
    for epoch, output in zip(range(len(points_on_sphere)), points_on_sphere):
        epoch_idx.append(torch.tensor([16*2**epoch]).repeat(output.shape[0]))
        concat_out.append(output)
    epoch_idx = torch.cat(epoch_idx)
    concat_out = torch.cat(concat_out)

    target_names = []
    for array in targets:
        for element in array:
            target_names.append(cifar10_labels[element.item()+1])


    output = {
        "x": concat_out[:,1],
        "y": concat_out[:,2],
        "z": concat_out[:,0],
        "target": target_names,
        "epoch": epoch_idx
    }
    output = pd.DataFrame(output)

    fig = px.scatter_3d(output, x='x', y='y', z='z', color="target", animation_frame="epoch", opacity=1, height=700)
    fig.update_traces(marker_size = marker_size)
    fig.update_layout(legend= {'itemsizing': 'constant'})
    fig.show()

import yaml
import os
def load_ablations(ablation_path):
    train_embeds = numpy.load(f"{ablation_path}/train_embeddings.npy", allow_pickle=True)
    train_targets = numpy.load(f"{ablation_path}/train_targets.npy", allow_pickle=True)
    test_embeds = numpy.load(f"{ablation_path}/test_embeddings.npy", allow_pickle=True)
    test_targets = numpy.load(f"{ablation_path}/test_targets.npy", allow_pickle=True)
    finetune_results = numpy.load(f"{ablation_path}/finetune_results.npy", allow_pickle=True)
    accuracies = numpy.load(f"{ablation_path}/accuracies.npy", allow_pickle=True)

    if "settings.yml" in os.listdir(ablation_path):
        try:
            with open(f"{ablation_path}/settings.yml") as file:
                settings = yaml.safe_load(file)
        except:
            settings = None
    else:
        settings = None

    return dict(
        train_embeds = train_embeds,
        train_targets = train_targets,
        test_embeds = test_embeds,
        test_targets = test_targets,
        fietune_results = finetune_results,
        accuracies = accuracies,
        settings = settings
    )

def plot_path_on_sphere(ablation_path, norm_function, test=False):
    ablations = load_ablations(ablation_path)
    if test:
        embeds = ablations["test_embeds"]
        targets = ablations["test_targets"]
    else:
        embeds = ablations["train_embeds"]
        targets = ablations["train_targets"]
    
    points_on_sphere = norm_function(torch.tensor(embeds.item()["projector"]), dim=-1)
    plot_on_sphere_animation(points_on_sphere, targets)


def plot_hist_on_sphere(ablation_dict, norm_function):

    all_on_sphere = norm_function(torch.tensor(ablation_dict["train_embeds"].item()["projector"]), dim=-1)

    fig = make_subplots(rows=3, cols=1)
    fig.add_trace(
        go.Histogram(x=all_on_sphere[-1][:,1], name="x"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Histogram(x=all_on_sphere[-1][:,2], name="y"),
        row=2, col=1
    )
    fig.add_trace(
        go.Histogram(x=all_on_sphere[-1][:,0], name="z"),
        row=3, col=1
    )
    fig.update_layout(xaxis=dict(range=[-1, 1]))
    fig.update_layout(xaxis2=dict(range=[-1, 1]))
    fig.update_layout(xaxis3=dict(range=[-1, 1]))
    fig.show()

def plot_path_hist_on_sphere(ablation_path, norm_function):
    ablation_dict = load_ablations(ablation_path)
    plot_hist_on_sphere(ablation_dict, norm_function)

def plot_hist_n_sphere(ablation_dict, norm_function, show_all=False):

    all_on_sphere = norm_function(torch.tensor(ablation_dict["train_embeds"].item()["projector"]), dim=-1)

    n = all_on_sphere.shape[-1]
    plotting_dims = []
    i = 1
    while i < n:
        plotting_dims.append(i-1)
        if show_all:
            i +=1
        else:
            i*=2
    plotting_dims.append(n-1)

    fig = make_subplots(rows=len(plotting_dims), cols=1)
    for row, dim in enumerate(plotting_dims):
        fig.add_trace(
            go.Histogram(x=all_on_sphere[-1][:,dim], name=str(dim)),
            row=row+1, col=1,
        )

    fig.update_xaxes({'range': (-1,1), 'autorange': False})
    if show_all:
        fig.update_layout(
            height=5000
        )
    
    fig.show()

def hist_sphere_unifrom_dist(dim=128, sample_n=10_000, norm_p=2, show_up_to=1):
    points = torch.normal(0,1,(1, sample_n, dim))
    points_on_sphere = normalize(points, dim=-1, p=norm_p)

    fig = figure_hist_embed_coordinates(points_on_sphere, show_up_to=show_up_to)

    k = dim
    a, b = (k-1)/2, (k-1)/2
    x = np.linspace(-1,1, 100)
    beta_dist = beta(a,b)
    beta_max = beta_dist.pdf(0.5)
    max_hist_value = np.histogram(points_on_sphere[0])[0].max()

    fig.add_trace(
        go.Scatter(x=x, y=(beta_dist.pdf((x+1)/2)/beta_max)*max_hist_value)
    )

    fig.update_xaxes({'range': (-1,1), 'autorange': False})
    fig.update_layout(
        title = f"{dim}-Sphere"
    )

    fig.show()

def figure_hist_embed_coordinates(embed, show_all=False, show_up_to=-1, epoch=-1):
    
    n = embed.shape[-1]
    plotting_dims = []
    i = 1
    number_of_axis_shown = 0
    while i < n:
        plotting_dims.append(i-1)
        number_of_axis_shown +=1
        if show_all or show_up_to > 0:
            i +=1
            if i > show_up_to:
                break
        else:
            i*=2
    if show_up_to < 1:
        plotting_dims.append(n-1)
    

    fig = make_subplots(rows=len(plotting_dims), cols=1)
    for row, dim in enumerate(plotting_dims):
        fig.add_trace(
            go.Histogram(x=embed[epoch,:,dim], name=str(dim)),
            row=row+1, col=1,
        )
    all_x = []
    for dat in fig.data:
        all_x += list(dat["x"])
    fig.update_xaxes({'range': (min(all_x), max(all_x)), 'autorange': False})
    
    fig.update_layout(
        height=number_of_axis_shown * 50 + 200
    )
    return fig

def plot_hist_coordinates(folder_name, what_output="projector", show_all=False, show_up_to=-1, epoch=-1):
    ablation_dict = load_ablations(get_ablation_path(folder_name))
    coordinates = torch.tensor(ablation_dict["train_embeds"].item()[what_output])

    fig = figure_hist_embed_coordinates(coordinates,  show_all=show_all, show_up_to=show_up_to, epoch=epoch)
    
    fig.update_layout(
        title=folder_name+": "+what_output+", epoch position:"+str(epoch)
    )
    if what_output == "on_sphere":
        fig.update_xaxes({'range': (-1,1), 'autorange': False})
    
    fig.show()

def plot_hist_norm_on_sphere(on_sphere):
    normalized1 = on_sphere.norm(dim=-1, p=1)
    normalized3 = on_sphere.norm(dim=-1, p=3)
    fig = make_subplots(rows=2, cols=1)
    fig.add_trace(
        go.Histogram(x=normalized1, name="L1 norm"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Histogram(x=normalized3, name="L3 norm"),
        row=2, col=1,
    )

    n = on_sphere.shape[-1]
    l1_max = n/(n**(1/2))
    print("L1 norm max:", l1_max)
    print("L3 norm max:", 1)
    fig.update_layout(xaxis=dict(range=[0, l1_max]))
    fig.update_layout(xaxis2=dict(range=[0, 1]))

    fig.show()

def plot_norm_hist(projector_out, name):
    normalized2 = projector_out.norm(dim=-1, p=2)
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(
        go.Histogram(x=normalized2, name=name),
        row=1, col=1,
    )
    
    fig.add_vline(x=normalized2.mean(), line_width=3, line_dash="dash", line_color="black")
    fig.update_layout(
        title=name
    )
    fig.update_layout(xaxis=dict(range=[0, normalized2.max()]))
    fig.show()
    print("L2 norm mean:", normalized2.mean().item())

def path_plot_norm_hists(ablation_path, test=False, epoch=-1):
    ablation_dict = load_ablations(ablation_path)
    projector_embeds = torch.tensor(ablation_dict["test_embeds" if test else "train_embeds"].item()["projector"][epoch])
    backbone_embeds = torch.tensor(ablation_dict["test_embeds" if test else "train_embeds"].item()["backbone"][epoch])

    plot_norm_hist(backbone_embeds, "backbone")
    plot_norm_hist(projector_embeds, "projector")

import numpy as np

def animate_norm_hist(embeds):
    normalized = embeds.norm(dim=-1)
    max_norm = normalized.max()
    # Create figure
    fig = go.Figure()
    nbinsx = 300

    max_y = 0
    # Add traces, one for each slider step
    for i, epoch in enumerate(normalized):
        counts, bins = np.histogram(epoch, bins=300)
        bins = 0.5*(bins[:-1] + bins[1:])
        fig.add_trace(
            go.Bar(
            visible=False,
            x=bins,
            y=counts,
            name=f"epoch {16 * 2**i}"
            ))
        y = np.histogram(epoch, bins=nbinsx)[0].max()
        if y > max_y:
            max_y = y
        
        fig.add_vline(x=epoch.mean(), line_width=3, line_dash="dash", line_color="black", visible=False)

    # Make 0th trace visible
    fig.data[0].visible = True

    # Create and add slider
    steps = []
    for i in range(len(fig.data)):
        step = dict(
            label = f"{16 * 2**i}",
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                {"title": "Epoch: " + str(16*2**i)}],  # layout attribute
            #args2=[{"visible": [False] * len(fig.data)}]
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        #step["args2"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Epoch: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders
    )
    fig.update_xaxes({'range': (0,max_norm), 'autorange': False})
    fig.update_yaxes({'range': (0,max_y*1.1), 'autorange': False})
    fig.update_layout(barmode='group', bargap=0.1,bargroupgap=0.0)

    fig.show()


def animate_norm_hist(embeds, name="Histogram", epoch_names=[]):
    normalized = embeds.norm(dim=-1)
    max_norm = normalized.max()
    # Create figure
    fig = go.Figure()
    nbins = 300
    bin_width = max_norm/nbins
    bins = np.linspace(0,max_norm, nbins)

    max_y = 0
    for epoch in normalized:
        y = np.histogram(epoch, bins=bins)[0].max()
        if y > max_y:
            max_y = y
    # Add traces, one for each slider step
    data_per_step = normalized

    # Initial histogram and mean line
    initial_data = data_per_step[0]
    mean_val = initial_data.mean()
    def name_frame(i, data=None):
        if epoch_names:
            return epoch_names[i]
        else:
            return f"{16 * 2**i}"

    fig = go.Figure(
        data=[
            go.Histogram(x=initial_data, opacity=0.6, marker_color='royalblue',
                                 xbins=dict( # bins used for histogram
                                    start=0,
                                    end=max_norm,
                                    size=bin_width
                                    ),
                                    name="Histogram"
                                ),
            go.Scatter(x=[mean_val, mean_val], y=[0, max_y], mode='lines',
                    line=dict(color='red', dash='dash'), name='Mean')
        ],
        layout=go.Layout(
            xaxis=dict(range=[0, max_norm]),
            yaxis=dict(range=[0, max_y]),
            sliders=[{
                "active": 0,
                "pad": {"t": 50},
                "currentvalue": {"prefix": "Epoch: "},
                "steps": [{
                    "method": "animate",
                    "label": name_frame(i),
                    "args": [[ name_frame(i)], {"frame": {"duration": 0, "redraw": True},
                                            "mode": "immediate"}]
                } for i in range(len(data_per_step))]
            }],
        ),
        frames=[
            go.Frame(
                data=[
                    go.Histogram(x=data, opacity=0.6, marker_color='royalblue',
                                 xbins=dict( # bins used for histogram
                                    start=0,
                                    end=max_norm,
                                    size=bin_width
                                    ),
                                ),
                    go.Scatter(x=[data.mean()]*2, y=[0, max_y], mode='lines',
                            line=dict(color='red', dash='dash'))
                ],
                name= name_frame(i)
            ) for i, data in enumerate(data_per_step)
        ]
    )
    fig.update_layout(
        title=name
    )

    fig.show()

def get_ablation_path(exp_name, model_str="simclr", dataset="cifar10", log_dir="scratch"):
    return f"outputs/{dataset}/{model_str}/scratch/{exp_name}/ablations"


def animate_path_hist(ablation_path, test=False, what_output="projector", name="Histogram", epoch_names=[]):
    ablation_dict = load_ablations(ablation_path)
    embeds = torch.tensor(ablation_dict["test_embeds" if test else "train_embeds"].item()[what_output])

    animate_norm_hist(embeds, name, epoch_names)


def animate_folder_hist(folder_name, test=False, what_output="projector", name=None, epoch_names=[], plot_train_and_test=False):
    name = name+": "+what_output if name else folder_name+": "+what_output
    if plot_train_and_test:
        print("Train")
        animate_path_hist(get_ablation_path(folder_name),False,what_output,name, epoch_names)
        print("Test")
        animate_path_hist(get_ablation_path(folder_name),True,what_output,name, epoch_names)
    else:
        animate_path_hist(get_ablation_path(folder_name),test,what_output,name, epoch_names)