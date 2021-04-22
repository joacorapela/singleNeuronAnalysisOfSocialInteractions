
import sys
import pdb
import argparse
import numpy as np
import plotly.graph_objects as go
sys.path.append("../src")
import utils
import classifiers
import probabilisticModels

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--likelihood_model",
                        help="likelihood model",
                        default="exponential")
    parser.add_argument("--neuron_label",
                        help="labels to neuron",
                        default="131A")
    parser.add_argument("--train_behaviors_labels",
                        help="behavioral labels for training",
                        default='[approach,following,headhead,headtail,conspecific,rice1,rice2]')
                        # default='[approach,following,headhead,headtail,conspecific,rice1,rice2]')
    parser.add_argument("--train_interactions",
                        help="interaction numbers for training",
                        default="[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]")
                        # default="[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]")
    parser.add_argument("--test_behavior_label",
                        help="behavioral label for testing",
                        default='headhead')
    parser.add_argument("--test_interactions",
                        help="interaction numbers for testing",
                        default="[16]")
    parser.add_argument("--bouttimes_filenames_pattern_pattern",
                        help="bouttimes filename pattern pattern",
                        default="../../../../data/120120/Behavior/*_int{:d}_bouttimes.npz")
    parser.add_argument("--spikes_times_filenames_pattern",
                        help="spikes times filename pattern",
                        default="../../../../data/120120/Neurons_BLA/{:s}.npy")
    parser.add_argument("--decodings_log_filename",
                        help="decodings log filename",
                        default="../../logs/decodings.log")
    args = parser.parse_args()

    likelihood_model = args.likelihood_model
    neuron_label = args.neuron_label
    train_behaviors_labels = args.train_behaviors_labels[1:-1].split(",")
    train_interactions = [int(str) for str in args.train_interactions[1:-1].split(",")]
    test_behavior_label = args.test_behavior_label
    test_interactions = [int(str) for str in args.test_interactions[1:-1].split(",")]
    bouttimes_filenames_pattern_pattern = args.bouttimes_filenames_pattern_pattern
    spikes_times_filenames_pattern = args.spikes_times_filenames_pattern
    decodings_log_filename = args.decodings_log_filename

    spikes_times_filename = spikes_times_filenames_pattern.format(neuron_label)
    spikes_times = np.load(spikes_times_filename)
    spikes_times = spikes_times.astype(float)
    train_ISIs = utils.get_ISIs_by_behavior_in_interactions(spikes_times=spikes_times,
                                                            behaviors_labels=train_behaviors_labels,
                                                            interactions=train_interactions,
                                                            bouttimes_filenames_pattern_pattern=bouttimes_filenames_pattern_pattern)
    test_ISIs = utils.get_ISIs_for_behaviors_in_interactions(spikes_times=spikes_times,
                                                             behaviors_labels=[test_behavior_label],
                                                             interactions=test_interactions,
                                                             bouttimes_filenames_pattern_pattern=bouttimes_filenames_pattern_pattern)
    classifier = classifiers.NaiveBayes()
    classifier.train(x_by_class=train_ISIs,
                     # model_class=probabilisticModels.Exponential)
                     model_class=probabilisticModels.InverseGaussian)
    classified_behavior = classifier.classify(x=test_ISIs)
    aString = "True test behavior={:s}, Classified test behavoir={:s}".format(test_behavior_label, classified_behavior)
    with open(decodings_log_filename, "a") as f:
        f.write(aString)
    print(aString)

    pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)
