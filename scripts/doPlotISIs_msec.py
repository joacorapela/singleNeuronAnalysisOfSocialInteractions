import sys
import pdb
import os
import glob
import argparse
import numpy as np
import plotly.graph_objects as go

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--spikesFilenamesPattern", help="spikes filename pattern", default="../../../data/120120/Neurons_BLA/*[A-Z].npy")
    parser.add_argument("--ISIsHistFigFilenamePattern", help="ISI  figure filename pattern", default="../figures/ISIsHist_neuron{:s}.{:s}")
    parser.add_argument("--sRate", help="sample rate", type=float, default=1e3)
    parser.add_argument("--maxISIinHist", help="maximum ISI in histogram", type=float, default=500.0) #msec
    parser.add_argument("--ISIsHistBinSize", help="bin size for ISI histogram", type=float, default=1.0) #msec
    parser.add_argument("--maxISIforHistPlot", help="max ISI for histogram plot (sec)", type=float, default=200) # msec
    parser.add_argument("--nNeurons", help="number of neurons to plot", type=int, default=20)
    args = parser.parse_args()

    spikesFilenamesPattern = args.spikesFilenamesPattern
    ISIsHistFigFilenamePattern = args.ISIsHistFigFilenamePattern
    sRate = args.sRate
    maxISIinHist = args.maxISIinHist
    ISIsHistBinSize = args.ISIsHistBinSize
    maxISIforHistPlot = args.maxISIforHistPlot
    nNeurons = args.nNeurons

    ISIsHistBinsMsec = np.arange(0, maxISIinHist, ISIsHistBinSize)

    spikesFilenames = glob.glob(spikesFilenamesPattern)
    if nNeurons>0:
        spikesFilenames = spikesFilenames[:nNeurons]
    else:
        nNeurons = len(spikesFilenames)
    filesNumbers = np.zeros(nNeurons)
    filesLetters = [None]*nNeurons
    ISIsHists = np.empty((nNeurons, len(ISIsHistBinsMsec)-1))
    neuronsLabels = np.empty(nNeurons, dtype=object)
    for i, spikesFilename in enumerate(spikesFilenames):
        neuronsLabels[i] = os.path.splitext(os.path.basename(spikesFilename))[0]
        filesNumbers[i] = int(neuronsLabels[i][:-1])
        filesLetters[i] = neuronsLabels[i][-1]
        spikesSamples = np.load(spikesFilename)
        ISIs = np.diff(spikesSamples[:, 0])
        counts, _ = np.histogram(ISIs, ISIsHistBinsMsec)
        ISIsHists[i,:] = counts/len(ISIs)

    for i in range(nNeurons):
        print("Plotting ISIs histogram for neuron {:s}".format(neuronsLabels[i]))
        htmlFigFilename = ISIsHistFigFilenamePattern.format(neuronsLabels[i], "html")
        pngFigFilename = ISIsHistFigFilenamePattern.format(neuronsLabels[i], "png")
        fig = go.Figure([go.Bar(x=ISIsHistBinsMsec, y=ISIsHists[i,:])])
        fig.update_xaxes(title_text="ISI (msec)", range=[0.0, maxISIforHistPlot])
        fig.update_yaxes(title_text="Probability")
        fig.write_html(htmlFigFilename)
        fig.write_image(pngFigFilename)
    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
