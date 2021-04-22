import pdb
import numpy as np

def get_bout_times_by_behavior(behaviors_labels,
                                  bouts_times_filenames):
    bout_times_by_behavior = dict(zip(behaviors_labels, [None]*len(behaviors_labels)))

    for bouts_times_filename in bouts_times_filenames:
        bouts_times_dict = np.load(bouts_times_filename)
        behaviors_labels_to_include = behaviors_labels & set(bouts_times_dict.keys())
        for behavioral_label in behaviors_labels_to_include:
            if bout_times_by_behavior[behavioral_label] is None:
                # bout times saved in secs but I want them in msecs
                bout_times_by_behavior[behavioral_label] = \
                        bouts_times_dict[behavioral_label]*1000
            else:
                # bout times saved in secs but I want them in msecs
                bout_times_for_behavior = bouts_times_dict[behavioral_label]*1000
                if bout_times_for_behavior.shape[0] > 0:
                    bout_times_by_behavior[behavioral_label] = \
                            np.vstack((bout_times_by_behavior[behavioral_label],
                                       bout_times_for_behavior))
    return bout_times_by_behavior

def get_spike_times_by_behavior_and_bout(bout_times_by_behavior, spikes_times):
    behaviors_labels = bout_times_by_behavior.keys()
    spike_times_by_behavior = dict(zip(behaviors_labels, [None]*len(behaviors_labels)))
    for behavioral_label in behaviors_labels:
        bout_times_for_behavior = bout_times_by_behavior[behavioral_label]
        n_bouts = bout_times_for_behavior.shape[0]
        spike_times_for_behavior = [None]*n_bouts
        for i in range(n_bouts):
            indices = np.logical_and(bout_times_for_behavior[i, 0] <= spikes_times,
                                     spikes_times < bout_times_for_behavior[i, 1])
            spike_times_for_behavior_and_bout = spikes_times[indices]
            spike_times_for_behavior[i] = spike_times_for_behavior_and_bout
        spike_times_by_behavior[behavioral_label] = spike_times_for_behavior
    return spike_times_by_behavior

def get_total_time_by_behavior(bout_times_by_behavior):
    total_time_by_behavior = {}
    for behavioral_label, bout_times in bout_times_by_behavior.items():
        total_time_for_behavior = (bout_times[:,1]-bout_times[:,0]).sum()
        total_time_by_behavior[behavioral_label] = total_time_for_behavior
    return total_time_by_behavior

def get_nro_spikes_by_behavior(spike_times_by_behavior_and_bout):
    nro_spikes_by_behavior = {}
    for behavioral_label, spike_times_for_bouts in spike_times_by_behavior_and_bout.items():
        nro_spikes_for_behavior = 0
        for spike_times_for_bout in spike_times_for_bouts:
            nro_spikes_for_behavior += len(spike_times_for_bout)
        nro_spikes_by_behavior[behavioral_label] = nro_spikes_for_behavior
    return nro_spikes_by_behavior

def get_spike_rate_by_behavior(nro_spikes_by_behavior, total_time_by_behavior, ordered_behaviors_labels):
    spike_rates = np.empty(len(ordered_behaviors_labels))
    for i, behavioral_label in enumerate(ordered_behaviors_labels):
        # division by 1e3 becausetotal_time_by_behavior in msec
        spike_rates[i] = nro_spikes_by_behavior[behavioral_label]/(total_time_by_behavior[behavioral_label]/1e3)
    return spike_rates
