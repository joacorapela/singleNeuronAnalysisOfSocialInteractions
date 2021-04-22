
import sys
import pdb
import argparse
import yaml
import numpy as np
import plotly.graph_objects as go

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--neurons_labels", help="neuron label", default="131B,34B,66A,74A,58A,50A")
    parser.add_argument("--interactions", help="interaction to plot", default="[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]")
    parser.add_argument("--params_filename", help="spikes filename pattern", default="../../../../data/120120/Parameters_13042019.yml")
    parser.add_argument("--spike_times_filenames_pattern", help="spike times filenames pattern", default="../../../../data/120120/Neurons_BLA/{:s}.npy")
    parser.add_argument("--fig_filename_pattern", help="spikes rates figure filename pattern", default="../../figures/spikesTimesOfInteractions_{:s}.{:s}")
    args = parser.parse_args()

    neurons_labels = np.array([neuron_label for neuron_label in args.neurons_labels.split(",")])
    interactions = np.array([int(str) for str in args.interactions[1:-1].split(",")])-1
    params_filename = args.params_filename
    spike_times_filenames_pattern = args.spike_times_filenames_pattern
    fig_filename_pattern = args.fig_filename_pattern

    with open(params_filename, "r") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    interactions_types = params["interactionType"]
    interactions_start_times = params["InteractionStart"]
    interactions_stop_times = params["InteractionStop"]

    for neuron_label in neurons_labels:
        print("Processing neuron {:s}".format(neuron_label))

        spikes_times_filename = spike_times_filenames_pattern.format(neuron_label)
        spikes_times = np.load(spikes_times_filename)
        spikes_times = spikes_times.astype(dtype=np.float)

        fig = go.Figure()
        for i, interaction in enumerate(interactions):
            interaction_start_time = interactions_start_times[interaction]*1e3
            interaction_stop_time = interactions_stop_times[interaction]*1e3
            interaction_spike_times = spikes_times[np.logical_and(interaction_start_time<=spikes_times[:,0], spikes_times[:,0]<=interaction_stop_time), 0]
            interaction_spike_times -= interaction_start_time
            number_spikes = len(interaction_spike_times)
            interaction_duration_secs = (interaction_stop_time-interaction_start_time)/1e3
            spike_rate = number_spikes/interaction_duration_secs
            legend = "{:s} ({:.2f} spikes/sec)".format(interactions_types[interaction], spike_rate)
            trace = go.Scatter(x=interaction_spike_times,
                               y=i*np.ones(len(interaction_spike_times)),
                               mode="markers", name=legend)
            fig.add_trace(trace)
        fig.update_xaxes(title_text="Time (msec)")
        fig.update_yaxes(showticklabels=False)

        html_fig_filename = fig_filename_pattern.format(neuron_label, "html")
        png_fig_filename = fig_filename_pattern.format(neuron_label, "png")
        fig.write_html(html_fig_filename)
        fig.write_image(png_fig_filename)

    pdb.set_trace()
if __name__=="__main__":
    main(sys.argv)
