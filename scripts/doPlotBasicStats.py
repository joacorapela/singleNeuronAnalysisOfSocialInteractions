import sys
import pdb
import os
import glob
import argparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--spikesFilenamesPattern", help="spikes filename pattern", default="../../../data/120120/Neurons_BLA/*[A-Z].npy")
    parser.add_argument("--spikesRatesFigFilenamePattern", help="spikes rates figure filename pattern", default="../figures/spikesRates.{:s}")
    parser.add_argument("--ISIsHistFigFilenamePattern", help="ISI  figure filename pattern", default="../figures/ISIsHist_neuron{:s}.{:s}")
    parser.add_argument("--sRate", help="sample rate", type=float, default=1e3)
    parser.add_argument("--maxISIinHist", help="maximum ISI in histogram", type=float, default=0.5)
    parser.add_argument("--ISIsHistBinSize", help="bin size for ISI histogram", type=float, default=1e-3)
    parser.add_argument("--maxISIforHistPlot", help="max ISI for histogram plot", type=float, default=0.2)
    args = parser.parse_args()

    spikesFilenamesPattern = args.spikesFilenamesPattern
    spikesRatesFigFilenamePattern = args.spikesRatesFigFilenamePattern
    ISIsHistFigFilenamePattern = args.ISIsHistFigFilenamePattern
    sRate = args.sRate
    maxISIinHist = args.maxISIinHist
    ISIsHistBinSize = args.ISIsHistBinSize
    maxISIforHistPlot = args.maxISIforHistPlot

    ISIsHistBins = np.arange(0, maxISIinHist, ISIsHistBinSize)

    spikesFilenames = glob.glob(spikesFilenamesPattern)
    nNeurons = len(spikesFilenames)
    filesNumbers = np.zeros(nNeurons)
    filesLetters = [None]*nNeurons
    ISIsHists = np.empty((nNeurons, len(ISIsHistBins)-1))
    spikeRates = np.zeros(nNeurons)
    neuronsLabels = np.empty(nNeurons, dtype=object)
    recordingLength = 0
    for i, spikesFilename in enumerate(spikesFilenames):
        neuronsLabels[i] = os.path.splitext(os.path.basename(spikesFilename))[0]
        filesNumbers[i] = int(neuronsLabels[i][:-1])
        filesLetters[i] = neuronsLabels[i][-1]
        spikesTimes = np.load(spikesFilename)
        ISIs = np.diff(spikesTimes[:,0])/sRate
        counts, _ = np.histogram(ISIs, ISIsHistBins)
        ISIsHists[i,:] = counts/len(ISIs)
        recordingLength = max(recordingLength, spikesTimes[-1])
        spikeRates[i] = len(spikesTimes)
    spikeRates /= recordingLength/sRate
    df = pd.DataFrame({"number": filesNumbers, "letter": filesLetters})
    sdf = df.sort_values(by=["number", "letter"])
    indices = sdf.index.to_list()
    neuronsLabels = neuronsLabels[indices]
    spikeRates = spikeRates[indices]

    htmlFigFilename = spikesRatesFigFilenamePattern.format("html")
    pngFigFilename = spikesRatesFigFilenamePattern.format("png")
    fig = go.Figure([go.Bar(x=neuronsLabels, y=spikeRates)])
    fig.update_yaxes(title_text="Spike Rate (Hz)")
    fig.update_xaxes(title_text="Neuron Label")
    fig.write_html(htmlFigFilename)
    fig.write_image(pngFigFilename)

    for i in range(nNeurons):
        print("Plotting ISIs histogram for neuron {:s}".format(neuronsLabels[i]))
        htmlFigFilename = ISIsHistFigFilenamePattern.format(neuronsLabels[i], "html")
        pngFigFilename = ISIsHistFigFilenamePattern.format(neuronsLabels[i], "png")
        fig = go.Figure([go.Bar(x=ISIsHistBins, y=ISIsHists[i,:])])
        fig.update_xaxes(title_text="ISI (sec)", range=[0.0, maxISIforHistPlot])
        fig.update_yaxes(title_text="Probability")
        fig.write_html(htmlFigFilename)
        fig.write_image(pngFigFilename)
        pdb.set_trace()
    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
