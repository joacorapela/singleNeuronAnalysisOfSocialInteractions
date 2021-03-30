
import sys
import pdb
import argparse
import numpy as np
import plotly.graph_objects as go
sys.path.append("../src")
import utils

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--neuron_label",
                        help="labels to neuron",
                        default="131A")
    parser.add_argument("--behaviors_labels",
                        help="behavioral labels to include",
                        default='[rice2]')
    parser.add_argument("--interactions",
                        help="interactions to include",
                        default='[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]')
    parser.add_argument("--maxISIinHist",
                        help="maximum ISI in histogram (msec)",
                        type=float, default=500)
    parser.add_argument("--ISIsHistBinSize",
                        help="bin size for ISI histogram (msec)",
                        type=float, default=1.0)
    parser.add_argument("--maxISIforHistPlot",
                        help="max ISI for histogram plot (msec)",
                        type=float, default=200)
    parser.add_argument("--bouttimes_filenames_pattern_pattern",
                        help="bouttimes filename pattern pattern",
                        default="../../../../data/120120/Behavior/*_int{:d}_bouttimes.npz")
    parser.add_argument("--spikes_times_filenames_pattern",
                        help="spikes times filename pattern",
                        default="../../../../data/120120/Neurons_BLA/{:s}.npy")
    parser.add_argument("--figure_filename_pattern",
                        help="figure filename pattern",
                        default="../../figures/ISIs_{:s}_{:s}_{:s}.{:s}")
    args = parser.parse_args()

    neuron_label = args.neuron_label
    behaviors_labels = args.behaviors_labels[1:-1].split(",")
    interactions = [int(str) for str in args.interactions[1:-1].split(",")]
    maxISIinHist = args.maxISIinHist
    ISIsHistBinSize = args.ISIsHistBinSize
    maxISIforHistPlot = args.maxISIforHistPlot
    bouttimes_filenames_pattern_pattern = args.bouttimes_filenames_pattern_pattern
    spikes_times_filenames_pattern = args.spikes_times_filenames_pattern
    figure_filename_pattern = args.figure_filename_pattern

    spikes_times_filename = spikes_times_filenames_pattern.format(neuron_label)
    spikes_times = np.load(spikes_times_filename)
    ISIs = utils.get_ISIs_for_behaviors_in_interactions(spikes_times=spikes_times,
                                                        behaviors_labels=behaviors_labels,
                                                        interactions=interactions,
                                                        bouttimes_filenames_pattern_pattern=bouttimes_filenames_pattern_pattern)

    ISIsHistBins = np.arange(0, maxISIinHist, ISIsHistBinSize)
    ISIsCounts, _ = np.histogram(ISIs, ISIsHistBins)
    ISIsHist = ISIsCounts/len(ISIs)
    htmlFigFilename = figure_filename_pattern.format(args.neuron_label, args.behaviors_labels, args.interactions, "html")
    pngFigFilename = figure_filename_pattern.format(args.neuron_label, args.behaviors_labels, args.interactions, "png")
    traceISIsHist = go.Bar(x=ISIsHistBins, y=ISIsHist)
    fig = go.Figure()
    fig.add_trace(traceISIsHist)
    fig.update_xaxes(title_text="ISI (msec)", range=[0.0, maxISIforHistPlot])
    fig.update_yaxes(title_text="Probability")
    fig.write_html(htmlFigFilename)
    fig.write_image(pngFigFilename)

    pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)
