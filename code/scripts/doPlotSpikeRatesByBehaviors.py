
import sys
import glob
import argparse
import numpy as np

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--spikesFilenamesPattern", help="spikes filename pattern", default="../../../data/120120/Neurons_BLA/*[A-Z].npy")
    parser.add_argument("--behavioralLabels", help="behavioral labels to include", default='["nonsocial", "approach", "following", "headhead", "headtail", "conspecific", "rice1", "rice2"]')
    parser.add_argument("--spikesRatesFigFilenamePattern", help="spikes rates figure filename pattern", default="../figures/spikesRates{:s}.{:s}")
    parser.add_argument("--sRate", help="sample rate", type=float, default=1e3)
    parser.add_argument("--binSizeSecs", help="bin size (sec)", type=float, default=0.05)
    args = parser.parse_args()

    spikesFilenamesPattern = args.spikesFilenamesPattern
    spikesRatesFigFilenamePattern = args.spikesRatesFigFilenamePattern
    behavioralLabels = ast.literal_eval(args.behavioralLabels)
    sRate = float(args.sRate)
    binSizeSecs = float(args.binSizeSecs)

    spikesFilenames = glob.glob(spikesFilenamesPattern)
    nNeurons = len(spikesFilenames)
    filesNumbers = np.zeros(nNeurons)
    filesLetters = [None]*nNeurons
    spikeRates = np.zeros(nNeurons)
    neuronsLabels = np.empty(nNeurons, dtype=object)
    recordingLength = 0

    boutSamplesByBehavior = utils.getBoutSamplesByBehavior(behavioralLabels=behavioralLabels, boutSamplesFilenames)

    for i, spikesFilename in enumerate(spikesFilenames):
        neuronsLabels[i] = os.path.splitext(os.path.basename(spikesFilename))[0]
        filesNumbers[i] = int(neuronsLabels[i][:-1])
        filesLetters[i] = neuronsLabels[i][-1]
        spikesSamples = np.load(spikesFilename)*sRate
        ISIs = np.diff(spikesTimes[:,0])/sRate
        counts, _ = np.histogram(ISIs, ISIsHistBins)
        ISIsHists[i,:] = counts/len(ISIs)
        recordingLength = max(recordingLength, spikesTimes[-1])
        spikeRates[i] = len(spikesTimes)
if __name__=="__main__":
    main(sys.argv)
