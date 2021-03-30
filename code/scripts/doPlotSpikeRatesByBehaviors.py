
import sys
import pdb
import argparse
import numpy as np
import plotly.graph_objects as go

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--spike_rates_by_behaviors_filename",
                        help="filename containing the spikes rates by behavior",
                        default="../../results/spikesRatesByBehaviors.npz")
    parser.add_argument("--spike_rates_fig_filename_pattern",
                        help="spikes rates figure filename pattern",
                        default="../../figures/spikesRatesByBehaviors{:s}.{:s}")
    args = parser.parse_args()

    spike_rates_by_behaviors_filename = args.spike_rates_by_behaviors_filename
    fig_filename_pattern = args.spike_rates_fig_filename_pattern

    load_res = np.load(file=spike_rates_by_behaviors_filename)
    spike_rates_by_behaviors = load_res["spike_rates_by_behaviors"]
    neurons_labels = load_res["neurons_labels"]
    behaviors_labels = load_res["behaviors_labels"]

    mean_spike_rastes_across_behaviors = np.mean(spike_rates_by_behaviors,
                                                 axis=1)
    # to allow broadcasting
    mean_spike_rastes_across_behaviors = \
            np.expand_dims(a=mean_spike_rastes_across_behaviors, axis=1)
    normalized_spike_rates_by_behaviors = \
            spike_rates_by_behaviors/mean_spike_rastes_across_behaviors

    fig = go.Figure()
    trace = go.Heatmap(z=spike_rates_by_behaviors,
                       x=behaviors_labels,
                       y=neurons_labels)
    fig.add_trace(trace)
    fig.write_image(fig_filename_pattern.format("Unnormalized", "png"))
    fig.write_html(fig_filename_pattern.format("Unnormalized", "html"))
    fig.show()

    fig = go.Figure()
    trace = go.Heatmap(z=normalized_spike_rates_by_behaviors,
                       x=behaviors_labels,
                       y=neurons_labels)
    fig.add_trace(trace)
    fig.write_image(fig_filename_pattern.format("Normalized", "png"))
    fig.write_html(fig_filename_pattern.format("Normalized", "html"))
    fig.show()

    pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)
