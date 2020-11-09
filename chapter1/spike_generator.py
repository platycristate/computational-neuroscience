import matplotlib.pyplot as plt
import numpy as np
import random
plt.style.use('seaborn-whitegrid')


def HomogeneousPoisson(rate, duration, dt=1e-3):
    '''
    Homogeneous Poisson process spike generator,
    implemented by dividing time into bins of and
    calculating probability of spike generation at each bin

    Return: array with spike times
    '''
    NoBins = int(duration / dt)
    time = np.random.uniform(0, 1, NoBins)

    # choose elements that have x_rand less than prob. of firing a spike
    spikes_indices = np.nonzero(rate * dt > time)

    # convert indices to time
    spikes = (np.array(spikes_indices) / NoBins) * duration

    return spikes.flatten()

def HomogeneousPoissonEfficient(rate, duration):
    '''
    Hoomogeneous Poisson process spike generator,
    implemented by estimating interspike intervals.

    Return: array with spike times
    '''
    spikes = [0]

    while spikes[-1] < duration:
        spikes.append( spikes[-1] - np.log(np.random.rand()) / rate )

    return np.array( spikes[1:-1] )

def NonhomogeneousPoisson(rate_func, duration):
    '''
    Nonhomogeneous Poisson process spike generator
    with time-dependent firing rate

    Return: array with spike times
    '''
    r_max = np.max( rate_func( np.linspace(0, duration, duration*1000 ) ))
    spikes = [0]
    while spikes[-1] < duration:
        spikes.append( spikes[-1] - np.log(np.random.rand()) / r_max )
    ToBeRemoved = []
    for i  in range(1, len(spikes)):
        if rate_func(spikes[i])/r_max < np.random.rand():
            ToBeRemoved.append(i)
    spikes_thinned = np.delete( np.array(spikes), ToBeRemoved )
    return spikes_thinned


def HomogeneousPoissonRefractory(rate, duration, tau):
    '''
    Homogeneous Poisson spike generator,
    with dynamic refractoriness after the spike
    '''
    spikes = [0]
    while spikes[-1] < duration:
        spikes.append(spikes[-1] - np.log(np.random.rand()) / rate)
    ToBeRemoved = []
    for i in range(1, len(spikes)):
        t = spikes[i]
        t_prev = spikes[i-1]
        # calculating how much rate has recovered after previous spike
        new_rate = rate_recovery(t - t_prev, r0=rate, tau_ref=tau)
        x_rand = random.random()
        if new_rate / rate < x_rand:
            ToBeRemoved.append(i)
    spikes = np.delete( np.array(spikes), ToBeRemoved )
    return spikes

def rate_recovery(t, r0, tau_ref):
    '''
    Introduces exponential recovering of firing rate,
    after firing a spike
    Should be used with Inhomogeneous Poisson process
    '''
    tau_ref /= 1000 # converts to seconds
    return r0 * (1 - np.exp(-t/tau_ref) )

def PlotTrials(process=HomogeneousPoissonEfficient, rate=40, duration=1, trials=20):
    plt.figure(figsize=(8, 5), dpi=100)
    NeuralResponses = []
    for i in range(trials):
        NeuralResponses.append( process(10, 1) )

    plt.eventplot(NeuralResponses, linelengths=0.5)
    plt.xlabel('time [ms]')
    plt.ylabel('trial number')

def distribution_spike_counts(spikes, step_interval=100, bindwidth=1, plot=False):
    spikes_counts = spike_count(spikes, step_interval)
    if plot:
        plt.figure(dpi=100)
        plt.hist(spikes_counts, bins='auto',
                color='purple', ec='black')
        plt.xlabel('no. of spikes in the interval')
    else:
        return spikes_counts

def ISI_distribution(spikes):
    plt.figure(dpi=100)
    isi = np.diff(spikes) * 1000
    plt.hist(isi, bins='auto', color='purple', ec='black')
    plt.xlabel('ISI [ms]', fontsize=16)
    plt.ylabel('no. of intervals', fontsize=16)

def spike_count(spikes, step_interval=100):
    '''
    Calculates the number of spikes over duration of spikes
    with a given step interval
    '''
    spikes_counts = []
    step_interval /= 1000 # converts to seconds
    start = 0
    end = step_interval
    while end <= np.max(spikes):
        spikes_counts.append( len(spikes[ (spikes > start) & (spikes <= end) ] ) )
        start += step_interval
        end += step_interval
    return np.array(spikes_counts)

def autocorrelation(spikes, time_lag=100, dt=1e-2):
    '''
    Computes autocorrelation histogram, by dividing
    time into bins and calculating. For each bin we calculate the no. of
    times 2 spikes are separated by some ISI (ms) (x-axis).
    '''
    duration = np.max(spikes)
    time_lag /= 1000 # convert to seconds
    NoBins = int( time_lag/dt)
    CountIntervals = []
    IBS = []
    for i in range(len(spikes)):
        # counting differences in time between current spike and all subsequent spikes
        intervals = spikes - spikes[i]
        IBS.extend(intervals)
    IBS = np.array(IBS)
    for m in range(-NoBins, NoBins+1):
        lower = m*dt - (dt/2)
        upper = m*dt + (dt/2)
        InBin = np.count_nonzero( (IBS >= lower) & (IBS < upper) )
        InBin = (InBin/duration) - ((len(spikes)*len(spikes) * dt) / (duration**2))
        CountIntervals.append(InBin)

    CountIntervals = np.array(CountIntervals)
    BinsBoundaries = np.array([m*dt for m in range(-NoBins, NoBins+1)]) * 1000
    return BinsBoundaries, CountIntervals

def fano(spikes):
    '''
    Computes Fanos factors for different intervals over
    which spikes are counted.
    duration/interval = no. of counts
    F = Var(X) / mean(X)
    '''
    fanos = []
    spikes = np.array(spikes)
    for t in range(1, 100): # ms
        spikes_counts = spike_count(spikes, step_interval=t)
        fanos.append( np.var(spikes_counts) / np.mean(spikes_counts) )
    return fanos

def coefficient_variation(data):
    '''
    C_v = std(X) / mean(X)
    '''
    return np.std(data) / np.mean(data)
