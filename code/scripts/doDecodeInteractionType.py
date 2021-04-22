
import sys
import pdb
import argparse
import yaml
import numpy as np
import plotly.graph_objects as go
import sklearn.model_selection
import sklearn.metrics
import plotly.express as px
sys.path.append("../src")
import utils
import classifiers
import probabilisticModels
import statMetrics

def get_interaction_group_index(interaction_type, interactions_groups):
    for interaction_group_index, interaction_group in enumerate(interactions_groups):
        if interaction_type in interaction_group:
            return interaction_group_index
    raise ValueError("interaction type {:s} not found", interaction_type)

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--likelihood_model",
                        help="likelihood model",
                        default="inverse_Gaussian")
                        # default="exponential")
    parser.add_argument("--plot_confusion_matrix",
                        help="plot confusion matrix",
                        action="store_false")
    parser.add_argument("--neurons_labels",
                        help="labels to neurons",
                        default="__all__")
    parser.add_argument("--interactions_nros",
                        help="interaction to plot",
                        default="[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]")
    parser.add_argument("--params_filename",
                        help="parameters filename",
                        default="../../../../data/120120/Parameters_13042019.yml")
    parser.add_argument("--percentage_train",
                        help="percentage of ISIs for training",
                        type=float,
                        default=0.8)
    parser.add_argument("--nResamples",
                        help="number of resamples to build the confusion matrix",
                        type=int,
                        default=100)
    parser.add_argument("--min_ISI",
                        help="minimum allowed ISI",
                        type=float,
                        default=1e-6)
    parser.add_argument("--spikes_times_filenames_dir",
                        help="spikes times filename directory",
                        default="../../../../data/120120/Neurons_BLA/")
    parser.add_argument("--results_filename_pattern",
                        help="results filename pattern",
                        default="../../results/interactions_decodings_{:s}_{:s}.npz")
    parser.add_argument("--fig_filename_pattern",
                        help="figure filename pattern",
                        default="../../figures/confusionMatrix_grouped_{:s}_{:s}.{:s}")
    parser.add_argument("--decodings_log_filename",
                        help="decodings log filename",
                        default="../../logs/interactions_decodings_grouped.csv")
    args = parser.parse_args()

    interactions_groups = [["NovelMale1_1", "NovelMale1_2",
                            "NovelMale2_1", "NovelMale2_2",
                            "FamiliarMale1_1", "FamiliarMale1_2"],
                           ["Female1_1", "Female1_2",
                            "Female2_1", "Female2_2",
                            "Female3_1", "Female3_2"],
                           ["Rice_1", "Rice_2"],
                           ["Toy_1", "Toy_2"]]
    interactions_groups_names = ["Male", "Female", "Food", "Object"]
    nInteractions_groups = len(interactions_groups_names)

    likelihood_model = args.likelihood_model
    plot_confusion_matrix = args.plot_confusion_matrix
    neurons_labels = [neuron_label for neuron_label in args.neurons_labels.split(",")]
    interactions_nros = np.array([int(str) for str in args.interactions_nros[1:-1].split(",")])-1
    params_filename = args.params_filename
    percentage_train = args.percentage_train
    nResamples = args.nResamples
    min_ISI = args.min_ISI
    spikes_times_filenames_dir = args.spikes_times_filenames_dir
    results_filename_pattern = args.results_filename_pattern
    fig_filename_pattern = args.fig_filename_pattern
    decodings_log_filename = args.decodings_log_filename

    if neurons_labels[0]=="__all__":
        neurons_labels = utils.get_all_neurons_labels(spikes_times_filenames_dir=spikes_times_filenames_dir)

    for neuron_label in neurons_labels:
        spikes_times_filename = "{:s}/{:s}.npy".format(spikes_times_filenames_dir, neuron_label)
        spikes_times = np.load(spikes_times_filename)
        spikes_times = spikes_times.astype(float)
        if likelihood_model=="exponential":
            model_class = probabilisticModels.Exponential
        elif likelihood_model=="inverse_Gaussian":
            model_class = probabilisticModels.InverseGaussian
        else:
            raise ValueError("Invalid likelihood_model={:s}".format(likelihood_model))

        with open(params_filename, "r") as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
        interactions_types = params["interactionType"]
        interactions_start_times = params["InteractionStart"]
        interactions_stop_times = params["InteractionStop"]
        nInteractions = len(interactions_types)

        interactions_spike_times = utils.get_spike_times_in_interactions(spikes_times=spikes_times, interactions_nros=interactions_nros, interactions_start_times=interactions_start_times, interactions_stop_times=interactions_stop_times)
        # interactions_ISIs = [np.diff(interaction_spike_times) for interaction_spike_times in interactions_spike_times]
        interactions_ISIs = [None]*nInteractions
        for i in range(nInteractions):
            interaction_ISIs = np.diff(interactions_spike_times[i])
            indices = np.where(interaction_ISIs<min_ISI)[0]
            if len(indices)>0:
                interaction_ISIs[indices] = min_ISI
            interactions_ISIs[i] = interaction_ISIs

        confusion_matrix = np.zeros((nInteractions_groups, nInteractions_groups))
        classifier = classifiers.NaiveBayes()
        interactions_labels = np.array([str(i) for i in range(nInteractions)])
        for i in range(nResamples):
            suffled_interactions_ISIs = [np.random.permutation(interaction_ISIs) for interaction_ISIs in interactions_ISIs]
            train_ISIs = [interaction_ISIs[:round(len(interaction_ISIs)*percentage_train)] for interaction_ISIs in suffled_interactions_ISIs]
            test_ISIs = [interaction_ISIs[round(len(interaction_ISIs)*percentage_train):] for interaction_ISIs in suffled_interactions_ISIs ]
            classifier.train(x=train_ISIs, y=interactions_labels, model_class=model_class)

            for interaction_test_index, interaction_test_ISIs in enumerate(test_ISIs):
                classified_interaction = classifier.classify(x=interaction_test_ISIs)
                classified_interaction_index = np.where(interactions_labels==classified_interaction)[0][0]
                classified_interaction_group_index = get_interaction_group_index(interaction_type=interactions_types[classified_interaction_index], interactions_groups=interactions_groups)
                interaction_test_group_index = get_interaction_group_index(interaction_type=interactions_types[interaction_test_index], interactions_groups=interactions_groups)
                confusion_matrix[classified_interaction_group_index, interaction_test_group_index] += 1

        nInteractions_per_group = np.array([len(interaction_group) for interaction_group in interactions_groups])
        confusion_matrix_normalization_matrix = np.diag(1.0/(nResamples*nInteractions_per_group))
        normalized_confusion_matrix = np.matmul(confusion_matrix, confusion_matrix_normalization_matrix)

        confusion_matrix_metrics = statMetrics.get_multiclass_confusion_matrix_metrics(confusion_matrix=confusion_matrix)

        aString = "{:s}, {:.02f}\n".format(neuron_label, confusion_matrix_metrics[1][2])
        with open(decodings_log_filename, "a") as f:
            f.write(aString)
        print(aString)

        if plot_confusion_matrix:
            interactions_groups_indices = np.arange(nInteractions_groups)
            fig = px.imshow(normalized_confusion_matrix,
                            labels=dict(y="Decoded Interaction", x="True Interaction"),
                            x=interactions_groups_indices,
                            y=interactions_groups_indices,
                            zmin=0.0, zmax=1.0)
            fig.update_layout(
                title = "Macro F1: {:.02f}".format(confusion_matrix_metrics[1][2]),
                xaxis = dict(
                    tickmode = 'array',
                    tickvals = interactions_groups_indices,
                    ticktext = interactions_groups_names,
                ),
                yaxis = dict(
                    tickmode = 'array',
                    tickvals = interactions_groups_indices,
                    ticktext = interactions_groups_names,
                ),
            )

            htmlFigFilename = fig_filename_pattern.format(neuron_label, likelihood_model, "html")
            pngFigFilename = fig_filename_pattern.format(neuron_label, likelihood_model, "png")
            fig.write_html(htmlFigFilename)
            fig.write_image(pngFigFilename)

    pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)
