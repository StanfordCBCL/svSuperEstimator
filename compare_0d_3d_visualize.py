import pickle
from svsuperestimator import visualizer

model_name = "0091_0001"

report = visualizer.Report()

with open(f"results/{model_name}.pickle", "rb") as ff:
    data = pickle.load(ff)

# print(data["0d"]["pressure_systole"])

plot_opts = {
    "width": 750,
    "height": 750,
    "xaxis_title": "3D",
    "yaxis_title": "0D",
    "title": "0D-3D without optimization",
}

plot_opts_opt = {
    "width": 750,
    "height": 750,
    "xaxis_title": "3D",
    "yaxis_title": "0D optimized",
    "title": "0D-3D with optimization",
}

report.add("Systolic pressure")
pres_plot = visualizer.Plot2D(**plot_opts)
pres_plot.add_point_trace(
    x=data["3d"]["pressure_systole"], y=data["0d"]["pressure_systole"], name=""
)

pres_plot_opt = visualizer.Plot2D(**plot_opts_opt)
pres_plot_opt.add_point_trace(
    x=data["3d"]["pressure_systole"],
    y=data["0d_opt"]["pressure_systole"],
    name="",
)

report.add([pres_plot, pres_plot_opt])

report.to_html("results/" + model_name + ".html", title=model_name)
