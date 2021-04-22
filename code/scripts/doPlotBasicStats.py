import sys
import pdb
import os
import glob
import argparse
import numpy as np
import scipy.stats
import pandas as pd
import plotly.graph_objects as go

def autocorr(x, lags):
    xcorr = np.correlate(x-x.mean(), x-x.mean(), 'full')
    xcorr = xcorr[xcorr.size//2:]/xcorr.max()
    return xcorr[:lags+1]

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--spikesFilenamesPattern", help="spikes filename pattern", default="../../../../data/120120/Neurons_BLA/*[A-Z].npy")
    parser.add_argument("--spikesRatesFigFilenamePattern", help="spikes rates figure filename pattern", default="../../figures/spikesRates.{:s}")
    parser.add_argument("--ISIsHistFigFilenamePattern", help="ISI  figure filename pattern", default="../../figures/ISIsHist_neuron{:s}.{:s}")
    parser.add_argument("--incFigFilenamePattern", help="spike counts figure filename pattern", default="../../figures/inc_neuron{:s}.{:s}")
    parser.add_argument("--incAutocorrsFigFilenamePattern", help="spike counts figure filename pattern", default="../../figures/inc_autocorrs_neuron{:s}.{:s}")
    parser.add_argument("--isisAutocorrsFigFilenamePattern", help="spike counts figure filename pattern", default="../../figures/isis_autocorrs_neuron{:s}.{:s}")
    parser.add_argument("--sRate", help="sample rate", type=float, default=1e3)
    parser.add_argument("--maxISIinHist", help="maximum ISI in histogram (msec)", type=float, default=500)
    parser.add_argument("--ISIsHistBinSize", help="bin size for ISI histogram (msec)", type=float, default=1.0)
    parser.add_argument("--maxISIforHistPlot", help="max ISI for histogram plot (msec)", type=float, default=200)
    parser.add_argument("--incBinSize", help="bin size for spike counts (msec)", type=float, default=200.0)
    parser.add_argument("--autocorrsIncBinSize", help="bin size for increments autocorrelations (msec)", type=float, default=1.0)
    parser.add_argument("--autocorrsIncNLags", help="number of lags for increments autocorrelations", type=int, default=100)
    parser.add_argument("--autocorrsIncMaxTime", help="maximum time to use to compute increments autocorrelations (msec)", type=float, default=1*60*1e3) # 1 minute default
    parser.add_argument("--autocorrsISIsNLags", help="number of lags for ISIs autocorrelations", type=int, default=20)
    parser.add_argument("--nISIsForAutocorr", help="number of ISIs to use to compute ISIs autocorrelations", type=int, default=1000)
    args = parser.parse_args()

    spikesFilenamesPattern = args.spikesFilenamesPattern
    spikesRatesFigFilenamePattern = args.spikesRatesFigFilenamePattern
    ISIsHistFigFilenamePattern = args.ISIsHistFigFilenamePattern
    incFigFilenamePattern = args.incFigFilenamePattern
    incAutocorrsFigFilenamePattern = args.incAutocorrsFigFilenamePattern
    isisAutocorrsFigFilenamePattern = args.isisAutocorrsFigFilenamePattern
    sRate = args.sRate
    maxISIinHist = args.maxISIinHist
    ISIsHistBinSize = args.ISIsHistBinSize
    maxISIforHistPlot = args.maxISIforHistPlot
    incBinSize = args.incBinSize
    autocorrsIncBinSize = args.autocorrsIncBinSize
    autocorrsIncNLags = args.autocorrsIncNLags
    autocorrsIncMaxTime = args.autocorrsIncMaxTime
    autocorrsISIsNLags = args.autocorrsISIsNLags
    nISIsForAutocorr = args.nISIsForAutocorr

    ISIsHistBins = np.arange(0, maxISIinHist, ISIsHistBinSize)

    # spikesFilenames = glob.glob(spikesFilenamesPattern)
    spikesFilenames = [
        "../../../../data/120120/Neurons_BLA/131A.npy",
        "../../../../data/120120/Neurons_BLA/34B.npy",
        "../../../../data/120120/Neurons_BLA/66A.npy",
        "../../../../data/120120/Neurons_BLA/74A.npy",
        "../../../../data/120120/Neurons_BLA/58A.npy",
        "../../../../data/120120/Neurons_BLA/50A.npy"
    ]
    nNeurons = len(spikesFilenames)
    filesNumbers = np.zeros(nNeurons)
    filesLetters = [None]*nNeurons
    ISIsHists = np.empty((nNeurons, len(ISIsHistBins)-1))
    expModelPDF = np.empty((nNeurons, len(ISIsHistBins)-1))
    invGaussModelPDF = np.empty((nNeurons, len(ISIsHistBins)-1))
    spikeRates = np.zeros(nNeurons)
    fanoFactors = np.zeros(nNeurons)
    incBins = None
    autocorrsIncBins = None
    autocorrsISIs = None
    neuronsLabels = np.empty(nNeurons, dtype=object)
    recordingLength = 0
    for i, spikesFilename in enumerate(spikesFilenames):
        print("Processing {:s}".format(spikesFilename))
        neuronsLabels[i] = os.path.splitext(os.path.basename(spikesFilename))[0]
        filesNumbers[i] = int(neuronsLabels[i][:-1])
        filesLetters[i] = neuronsLabels[i][-1]
        spikeTimes = np.load(spikesFilename)
        # ISIs = np.diff(spikeTimes[:,0])/sRate
        ISIs = np.diff(spikeTimes[:,0])
        ISIs[np.where(ISIs==0)[0]] = 1.0 # fixing problem due to storing spike times in milliseconds
        ISIsCounts, _ = np.histogram(ISIs, ISIsHistBins)
        ISIsHists[i,:] = ISIsCounts/len(ISIs)
        if incBins is None:
            incBins = np.arange(0, spikeTimes.max(), incBinSize)
            inc = np.zeros((nNeurons, len(incBins)-1))
            N = inc.shape[1]
            shape = (N-1)/2
            scale = 2/(N-1)
            fanoFactors95CI = scipy.stats.gamma.ppf([.025, .975], shape, scale=scale)
        if autocorrsIncBins is None:
            autocorrsIncBins = np.arange(0, autocorrsIncMaxTime, autocorrsIncBinSize)
            autocorrsIncSig = 2.0/np.sqrt(len(autocorrsIncBins))
            autocorrsInc = np.zeros((nNeurons, autocorrsIncNLags+1))
            autocorrsISIsSig = np.zeros(nNeurons)
        neuronInc, _ = np.histogram(spikeTimes, incBins)
        inc[i,:] = neuronInc
        fanoFactors[i] = neuronInc.var()/neuronInc.mean()
        oneNeuronAutocorrsInc, _ = np.histogram(spikeTimes, autocorrsIncBins)
        autocorrsInc[i,:] = autocorr(oneNeuronAutocorrsInc, autocorrsIncNLags)
        if autocorrsISIs is None:
            autocorrsISIs = np.zeros((nNeurons, autocorrsISIsNLags+1))
        if len(ISIs)>nISIsForAutocorr:
            ISIsForAutocorr = ISIs[:nISIsForAutocorr]
        else:
            ISIsForAutocorr = ISIs
        autocorrsISIsSig[i] = 2.0/np.sqrt(len(ISIsForAutocorr))
        autocorrsISIs[i,:] = autocorr(ISIsForAutocorr, autocorrsISIsNLags)
        lbda = 1.0/ISIs.mean()
        expModelPDF[i,:] = lbda*np.exp(-lbda*ISIsHistBins[:-1])*ISIsHistBinSize
        # inverse Gaussian
        mu = ISIs.mean()
        lbda = 1/(1/ISIs-1/mu).mean()
        bins = ISIsHistBins[:-1]
        model = (
            np.sqrt(lbda/2/np.pi/bins**3)*
            np.exp(-lbda*(bins-mu)**2/2/mu**2/bins)*ISIsHistBinSize
        )
        model[0] = 0
        invGaussModelPDF[i,:] = model
        #
        recordingLength = max(recordingLength, spikeTimes[-1])
        spikeRates[i] = len(spikeTimes)
    # spikeRates /= recordingLength/sRate
    spikeRates /= recordingLength
    # df = pd.DataFrame({"number": filesNumbers, "letter": filesLetters})
    # sdf = df.sort_values(by=["number", "letter"])
    # indices = sdf.index.to_list()
    indices = np.argsort(-spikeRates)
    neuronsLabels = neuronsLabels[indices]
    ISIsHists = ISIsHists[indices,:]
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
        traceISIsHist = go.Bar(x=ISIsHistBins, y=ISIsHists[i,:], name="empirical")
        traceExpModelPDF = go.Scatter(x=ISIsHistBins, y=expModelPDF[i,:], mode="lines", name="exponential model")
        traceInvGaussModelPDF = go.Scatter(x=ISIsHistBins, y=invGaussModelPDF[i,:], mode="lines", name="invGauss model")
        fig = go.Figure()
        fig.add_trace(traceISIsHist)
        fig.add_trace(traceExpModelPDF)
        fig.add_trace(traceInvGaussModelPDF)
        fig.update_xaxes(title_text="ISI (msec)", range=[0.0, maxISIforHistPlot])
        fig.update_yaxes(title_text="Probability")
        fig.write_html(htmlFigFilename)
        fig.write_image(pngFigFilename)

        print("Plotting increments for neuron {:s}".format(neuronsLabels[i]))
        htmlFigFilename = incFigFilenamePattern.format(neuronsLabels[i], "html")
        pngFigFilename = incFigFilenamePattern.format(neuronsLabels[i], "png")
        # fig = go.Figure([go.Bar(x=incBins/1000, y=inc[i,:])])
        trace = go.Scatter(x=incBins/1000, y=inc[i,:], mode="lines+markers")
        fig = go.Figure()
        fig.add_trace(trace)
        fig.update_xaxes(title_text="Time (sec)")
        fig.update_yaxes(title_text="Spike Count")
        fig.update_layout(title="Fano Factor {:02f}, 95% CI ({:.02f}, {:.02f})".format(fanoFactors[i], fanoFactors95CI[0], fanoFactors95CI[1]))
        fig.write_html(htmlFigFilename)
        fig.write_image(pngFigFilename)

        print("Plotting increments autocorrelations for neuron {:s}".format(neuronsLabels[i]))
        htmlFigFilename = incAutocorrsFigFilenamePattern.format(neuronsLabels[i], "html")
        pngFigFilename = incAutocorrsFigFilenamePattern.format(neuronsLabels[i], "png")
        lagsSecs = np.arange(autocorrsIncNLags)*autocorrsIncBinSize
        traceAutocorrelation = go.Scatter(x=lagsSecs, y=autocorrsInc[i,:], mode="markers")
        fig = go.Figure()
        fig.add_trace(traceAutocorrelation)
        fig.add_hline(y=autocorrsIncSig, line_dash="dash")
        fig.add_hline(y=-autocorrsIncSig, line_dash="dash")
        fig.update_layout(yaxis_range=[-.1,.1])
        fig.update_xaxes(title_text="Lag (msec)")
        fig.update_yaxes(title_text="Autocorrelation")
        fig.write_html(htmlFigFilename)
        fig.write_image(pngFigFilename)

        print("Plotting ISIs autocorrelations for neuron {:s}".format(neuronsLabels[i]))
        htmlFigFilename = isisAutocorrsFigFilenamePattern.format(neuronsLabels[i], "html")
        pngFigFilename = isisAutocorrsFigFilenamePattern.format(neuronsLabels[i], "png")
        traceAutocorrelation = go.Scatter(x=np.arange(autocorrsISIsNLags), y=autocorrsISIs[i,:], mode="markers")
        fig = go.Figure()
        fig.add_trace(traceAutocorrelation)
        fig.add_hline(y=autocorrsISIsSig[i], line_dash="dash")
        fig.add_hline(y=-autocorrsISIsSig[i], line_dash="dash")
        fig.update_layout(yaxis_range=[-.2,.2])
        fig.update_xaxes(title_text="Number of Spikes in the Past")
        fig.update_yaxes(title_text="Autocorrelation")
        fig.write_html(htmlFigFilename)
        fig.write_image(pngFigFilename)

        pdb.set_trace()
    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
