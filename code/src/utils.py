import numpy as np

def getBoutSamplesByBehavior(behavioralLabels, boutSamplesFilenames):
    boutTimesByBehavior = dict(zip(behavioralLabels, [None]*len(behavioralLabels)))

    for boutSamplesFilename in boutSamplesFilenames:
        boutSamples = np.load(boutSamplesFilename)
        behavioralLabelsToInclude = behavioralLabels & set(boutSamples.keys())
        for behavioralLabel in behavioralLabelsToInclude:
            if boutTimesByBehavior[behavioralLabel] is None:
                boutTimesByBehavior[behavioralLabel] = boutSamples[behavioralLabel]
            else:
                boutTimesByBehavior[behavioralLabel] = np.vstack(boutTimesByBehavior[behavioralLabel], boutSamples[behavioralLabel])
    return boutTimesByBehavior

def getSpikeSamplesByBoutAndBehavior(boutsSamplesByBehavior, spikesSamples):
    behavioralLabels = boutsSamplesByBehavior.keys()
    spikeSamplesByBehavior = dict(zip(behavioralLabels, [None]*len(behavioralLabels)))
    for behavioralLabel in behavioralLabels:
        boutSamplesForBehavior = boutsSamplesByBehavior[behavioralLabel]
        spikeSamplesForBehavior = [None]*boutSamples.size(0)
        for i in range(len(spikeSamplesForBehavior)):
            spikeSamplesForBoutAndBehavior = spikesSamples[boutSamplesForBehavior[i,0]<=spikesSamples & spikesSamples<boutSamplesForBehavior[i,1]]
            spikeSamplesForBehavior.append(spikeSamplesForBoutAndBehavior)
        spikeSamplesByBehavior[behavioralLabel] = spikeSamplesForBoutAndBehavior
    return spikeSamplesByBehavior

