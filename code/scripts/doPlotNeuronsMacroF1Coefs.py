
import sys
import pdb
import argparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--neurons_labels_f1Coefs_filename",
                        help="filename containing the neurons labels and f1 coefficients",
                        default="../../logs/interactions_decodings_grouped.csv")
    parser.add_argument("--fig_filename_pattern",
                        help="f1 coefficients figure filename pattern",
                        default="../../figures/f1_coefs.{:s}")
    args = parser.parse_args()

    neurons_labels_f1Coefs_filename = args.neurons_labels_f1Coefs_filename
    fig_filename_pattern= args.fig_filename_pattern

    neurons_labels_f1Coefs = pd.read_csv(neurons_labels_f1Coefs_filename).to_numpy()
    sort_indices = np.argsort(-neurons_labels_f1Coefs[:,1])
    neurons_labels_sorted = neurons_labels_f1Coefs[sort_indices, 0]
    neurons_f1Coefs_sorted = neurons_labels_f1Coefs[sort_indices, 1]

    htmlFigFilename = fig_filename_pattern.format("html")
    pngFigFilename = fig_filename_pattern.format("png")
    traceF1Coefs = go.Bar(x=neurons_labels_sorted, y=neurons_f1Coefs_sorted)
    fig = go.Figure()
    fig.add_trace(traceF1Coefs)
    fig.update_xaxes(title_text="Neuron Label")
    fig.update_yaxes(title_text="Macro-F1 Coefficient")
    fig.write_html(htmlFigFilename)
    fig.write_image(pngFigFilename)

    pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)
