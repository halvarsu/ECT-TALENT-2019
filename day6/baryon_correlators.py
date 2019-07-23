from uncertainties import unumpy as unp
import matplotlib.pyplot as plt
import uncertainties as uc
import pandas as pd
import numpy as np 
import glob

def load_data_const_vol(folder):
    import glob

    data = []
    for file in glob.glob(folder + '/*'):
        if '-a-' not in file and '-1-' not in file:
            continue
        df = pd.read_csv(file, index_col=0)
        cfg_num = file[-9:-4]
        line = df['p000']
        line.name = cfg_num
        data.append(line)
        #plt.semilogy(df['p000'])
    data = pd.concat(data,axis=1).sort_index(axis=1)
    return data
print('1')
def bootstrap(values, plateau1= None, plateau2 = None, K = 50, ax=None):
    Nsamples, Nt = values.shape
    indx = np.arange(Nsamples)
    fvalues = []
    times = np.arange(Nt)

    # plateaus

    mask_a = plateau1 if plateau1 is not None else slice(6,20)
    mask_b = plateau2 if plateau2 is not None else slice(-13,-5)
    times_a = times[1:][mask_a]
    times_b = times[1:][mask_b]

    ma_vals = []
    mb_vals = []

    np.random.seed(0)
    for i in range(K):
        choice = np.random.randint(Nsamples, size=Nsamples)
        avg    = np.average(values[choice], axis=0)

        fval   = np.log(avg[:-1]/avg[1:])

        mass_a = np.polyfit(times_a, fval[mask_a], deg=0)
        ma_vals.append(mass_a)
        mass_b = np.polyfit(times_b, fval[mask_b], deg=0)
        mb_vals.append(mass_b)

        fvalues.append(fval)

    fvalues = np.array(fvalues)

    if ax is None:
        ax = plt.gca()

    mean = np.mean(fvalues, axis=0)
    std = np.std(fvalues, axis=0)
    ax.plot(times[1:], mean)
    ax.fill_between(times[1:], mean-std, mean+std, alpha = 0.5)
    Ma = uc.ufloat(np.abs(np.mean(ma_vals)), np.std(ma_vals))
    Mb = uc.ufloat(np.abs(np.mean(mb_vals)), np.std(mb_vals))

    ax.plot(times_a, Ma.n*np.ones_like(times_a))
    ax.plot(times_b, -Mb.n*np.ones_like(times_b))
    
    return Ma, Mb

def bootstrap_v2(values, plateau1= None, plateau2 = None, K = 50, ax=None):
    Nsamples, Nt = values.shape
    indx = np.arange(Nsamples)
    times = np.arange(Nt)

    # plateaus

    mask_a = plateau1 if plateau1 is not None else slice(6,20)
    mask_b = plateau2 if plateau2 is not None else slice(-13,-5)
    times_a = times[1:][mask_a]
    times_b = times[1:][mask_b]

    ma_vals = []
    mb_vals = []

    np.random.seed(0)
    fvalues = np.log(values[:,:-1]/values[:,1:])
    samples = []
    print(fvalues.shape)

    for i in range(K):
        choice = np.random.randint(Nsamples, size=Nsamples)
        avg    = np.mean(fvalues[choice], axis=0)
        std    = np.std(fvalues[choice], axis=0)

        samples.append([avg, std])

    samples = np.array(samples)
    return samples
