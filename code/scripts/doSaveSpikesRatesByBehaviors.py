
import sys
import pdb
import os
import glob
import argparse
import ast
import numpy as np
import pandas as pd
sys.path.append("../src")
import utils

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--behaviors_labels",
                        help="behavioral labels to include",
                        default='["nonsocial", "approach", "following", "headhead", "headtail", "conspecific", "rice1", "rice2"]')
    parser.add_argument("--bouts_times_filenames_pattern",
                        help="bout times filename pattern",
                        default="../../../../../data/120120/Behavior/*_int*_bouttimes.npz")
    parser.add_argument("--spikes_filenames_pattern",
                        help="spikes filename pattern",
                        default="../../../../../data/120120/Neurons_BLA/*[A-Z].npy")
    parser.add_argument("--save_filename",
                        help="filename to save the spikes rates by behavior",
                        default="../../results/spikesRatesByBehaviors.npz")
    args = parser.parse_args()

    ordered_behaviors_labels = ast.literal_eval(args.behaviors_labels)
    behaviors_labels = set(ordered_behaviors_labels)
    bouts_times_filenames_pattern = args.bouts_times_filenames_pattern
    spikes_filenames_pattern = args.spikes_filenames_pattern
    save_filename = args.save_filename

    bouts_times_filenames = glob.glob(bouts_times_filenames_pattern)
    spikes_filenames = glob.glob(spikes_filenames_pattern)
    n_neurons = len(spikes_filenames)
    neurons_labels = [None]*n_neurons
    files_numbers = np.zeros(n_neurons)
    files_letters = [None]*n_neurons

    bout_times_by_behavior = utils.get_bout_times_by_behavior(
        behaviors_labels=behaviors_labels,
        bouts_times_filenames=bouts_times_filenames)
    total_time_by_behavior = utils.get_total_time_by_behavior(
        bout_times_by_behavior=bout_times_by_behavior)
    spike_rates_by_behaviors = np.empty((n_neurons, len(ordered_behaviors_labels)))
    for i, spikes_filename in enumerate(spikes_filenames):
        neurons_labels[i] = os.path.splitext(os.path.basename(spikes_filename))[0]
        files_numbers[i] = int(neurons_labels[i][:-1])
        files_letters[i] = neurons_labels[i][-1]
        spikes_times = np.load(spikes_filename)
        spike_times_by_behavior_and_bout = utils.get_spike_times_by_behavior_and_bout(
            bout_times_by_behavior=bout_times_by_behavior,
            spikes_times=spikes_times)
        nro_spikes_by_behavior = utils.get_nro_spikes_by_behavior(
            spike_times_by_behavior_and_bout=spike_times_by_behavior_and_bout)
        spike_rates_by_behaviors[i,:] = utils.get_spike_rate_by_behavior(
            nro_spikes_by_behavior=nro_spikes_by_behavior,
            total_time_by_behavior=total_time_by_behavior,
            ordered_behaviors_labels=ordered_behaviors_labels)
    df = pd.DataFrame({"number": files_numbers, "letter": files_letters})
    sdf = df.sort_values(by=["number", "letter"])
    indices = sdf.index.to_list()
    neurons_labels = [neurons_labels[i] for i in indices]
    spike_rates_by_behaviors = spike_rates_by_behaviors[indices, :]

    np.savez(file=save_filename,
             spike_rates_by_behaviors=spike_rates_by_behaviors,
             neurons_labels=neurons_labels,
             behaviors_labels=ordered_behaviors_labels)
    pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)
