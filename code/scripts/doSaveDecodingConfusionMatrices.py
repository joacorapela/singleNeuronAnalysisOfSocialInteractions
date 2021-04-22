
import sys
import os
import pdb
import glob
import argparse
import numpy as np
import plotly.express as px
sys.path.append("../src")
import utils
import classifiers
import probabilisticModels

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--likelihood_model",
                        help="likelihood model",
                        default="inverse_Gaussian")
                        # default="exponential")
    parser.add_argument("--behaviors_labels",
                        help="behavioral labels",
                        default='[nonsocial,headtail,conspecific]')
                        # default='[approach,following,headhead,headtail,conspecific,rice1,rice2]')
    parser.add_argument("--interactions",
                        help="interaction numbers",
                        default="[1,3,5,6,7,8,9,11,13,14,15]")
                        # default="[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]")
    parser.add_argument("--bouttimes_filenames_pattern_pattern",
                        help="bouttimes filename pattern pattern",
                        default="../../../../data/120120/Behavior/*_int{:d}_bouttimes.npz")
    parser.add_argument("--spikes_times_filenames_pattern",
                        help="spikes times filename pattern",
                        default="../../../../data/120120/Neurons_BLA/*.npy")
    parser.add_argument("--decodings_log_filename_pattern",
                        help="decodings log filename pattern",
                        default="../../logs/decodings_{:s}.log")
    parser.add_argument("--confusion_matrix_filename_pattern",
                        help="confusion matrix filename pattern",
                        default="../../results/confusionMatrix_{:s}_{:s}.npz")
    parser.add_argument("--fig_filename_pattern",
                        help="figure filename pattern",
                        default="../../figures/confusionMatrix_{:s}_{:s}.{:s}")
    args = parser.parse_args()

    likelihood_model = args.likelihood_model
    behaviors_labels = args.behaviors_labels[1:-1].split(",")
    interactions = [int(str) for str in args.interactions[1:-1].split(",")]
    bouttimes_filenames_pattern_pattern = args.bouttimes_filenames_pattern_pattern
    spikes_times_filenames_pattern = args.spikes_times_filenames_pattern
    decodings_log_filename_pattern = args.decodings_log_filename_pattern
    confusion_matrix_filename_pattern = args.confusion_matrix_filename_pattern
    fig_filename_pattern = args.fig_filename_pattern

    decodings_log_filename = decodings_log_filename_pattern.format(likelihood_model)
    spikes_times_filenames = glob.glob(spikes_times_filenames_pattern)
    if likelihood_model=="exponential":
        model_class = probabilisticModels.Exponential
    elif likelihood_model=="inverse_Gaussian":
        model_class = probabilisticModels.InverseGaussian
    else:
        raise ValueError("Invalid likelihood_model={:s}".format(likelihood_model))

    nBehaviors = len(behaviors_labels)
    confusion_matrix = np.zeros((nBehaviors, nBehaviors))
    classifier = classifiers.NaiveBayes()
    for i, spikes_times_filename in enumerate(spikes_times_filenames):
        print("Processing {:s}".format(spikes_times_filename))
        neuron_label = os.path.splitext(os.path.basename(spikes_times_filename))[0]
        spikes_times = np.load(spikes_times_filename)
        spikes_times = spikes_times.astype(float)

        for test_behavior_index, test_behavior_label in enumerate(behaviors_labels):
            for j, test_interaction in enumerate(interactions):
                train_interactions = np.delete(interactions, j)
                train_ISIs = utils.get_ISIs_by_behavior_in_interactions(spikes_times=spikes_times,
                                                                        behaviors_labels=behaviors_labels,
                                                                        interactions=train_interactions,
                                                                        bouttimes_filenames_pattern_pattern=bouttimes_filenames_pattern_pattern)
                test_ISIs = utils.get_ISIs_for_behaviors_in_interactions(spikes_times=spikes_times,
                                                                         behaviors_labels=[test_behavior_label],
                                                                         interactions=[test_interaction],
                                                                         bouttimes_filenames_pattern_pattern=bouttimes_filenames_pattern_pattern)
                if test_ISIs is not None:
                    classifier.train(x_by_class=train_ISIs, model_class=model_class)
                    classified_behavior = classifier.classify(x=test_ISIs)
                    if classified_behavior is not None:
                        classified_behavior_index = np.where(np.array(behaviors_labels)==classified_behavior)[0][0]
                        confusion_matrix[test_behavior_index, classified_behavior_index] += 1
                        aString = "Test interaction={:02d}. True test behavior={:s}, Classified test behavoir={:s}".format(test_interaction, test_behavior_label, classified_behavior)

        row_sums = np.sum(confusion_matrix, axis=1)
        normalized_confusion_matrix = np.matmul(np.diag(1/row_sums), confusion_matrix)
        neuron_rank = np.diag(normalized_confusion_matrix).sum()
        aString = "{:s}\t{:f}\n".format(neuron_label, neuron_rank)
        with open(decodings_log_filename, "a") as f:
            f.write(aString)
        print(aString)
        confusion_matrix_filename = confusion_matrix_filename_pattern.format(neuron_label, likelihood_model)
        np.savez(confusion_matrix_filename, confusion_matrix=confusion_matrix, normalized_confusion_matrix=normalized_confusion_matrix, behaviors_labels=behaviors_labels)
        fig = px.imshow(normalized_confusion_matrix,
                        labels=dict(x="Decoded Behavior", y="True Behavior", color="Proportion"),
                        x=behaviors_labels,
                        y=behaviors_labels,
                        zmin=0.0, zmax=1.0)
        htmlFigFilename = fig_filename_pattern.format(neuron_label, likelihood_model, "html")
        pngFigFilename = fig_filename_pattern.format(neuron_label, likelihood_model, "png")
        fig.write_html(htmlFigFilename)
        fig.write_image(pngFigFilename)
    pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)
