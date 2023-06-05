import os
import numpy as np
import toml
import csv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd


class Settings:
    volume: int
    kappa: float
    ensemble: str

    def __init__(self, volume, kappa, ensemble):
        self.volume = volume
        self.kappa = kappa
        self.ensemble = ensemble

    @staticmethod
    def from_config(config):
        volume = config["model"]["volume"]
        kappa = config["model"]["kappa_0"]
        ensemble = config["model"]["ensemble"]
        return Settings(volume, kappa, ensemble)


class DataSet:
    data: np.ndarray
    settings: Settings

    def __init__(self, data, settings):
        self.data = data
        self.settings = settings

    # note: currently only returns data from first file with matching settings
    @classmethod
    def load(cls, dirname, filename):
        vec = not (filename == "volume" or filename == "vertex_count")
        root = os.getcwd()
        output_path = os.path.join(root, "..", "output")
        dirs = list(os.scandir(output_path))

        for directory in dirs:
            path = os.path.join(output_path, directory.name)
            config = toml.load(os.path.join(path, "config.toml"))
            settings = Settings.from_config(config)
            if directory.name == dirname:
                full_path = os.path.join(path, f"{filename}.csv")
                if vec:
                    data = cls._read_vec_data(full_path)
                else:
                    data = np.loadtxt(full_path, dtype=int)
                return cls(data, settings)

    @classmethod
    def _read_vec_data(cls, path):
        # read data and find max number of columns
        with open(path, 'r') as file:
            datareader = csv.reader(file)
            data = []
            max_cols = 0

            for row in datareader:
                data.append(row)
                max_cols = max(max_cols, len(row))

        # pad data with zeroes s.t. all columns are of equal length
        for i in range(len(data)):
            row = data[i]
            row = np.array(row, dtype=int)
            row = np.pad(row, (0, max_cols - len(row)), 'constant')
            data[i] = row

        # convert to numpy array
        data = np.array(data)
        return data

    def transform(self, trans=""):
        # extract data
        old_data = self.data

        # apply transformation
        if trans == "":
            new_data = self
        elif trans == "max":
            new_data = self._get_max(old_data)
        elif trans == "parity":
            new_data = self._parity(old_data)

        # return new DataSet instance
        return DataSet(new_data, self.settings)

    @staticmethod
    def _get_max(data):
        shape = np.shape(data)
        if len(shape) == 2:
            (n_series, n_hist) = shape
            index_mask = np.tile(np.arange(n_hist), (n_series, 1))
            masked_data = np.where(data > 0, index_mask, 0)
            return np.max(masked_data, axis=1)
        else:
            return np.zeros_like(data)
    
    @staticmethod
    def _parity(data):
        if len(np.shape(data)) == 2:
            even = np.sum(data[:, ::2], axis=1)
            total = np.sum(data, axis=1)
            parity = 2 * even / total - 1
            return parity
        else:
            return np.zeros_like(data)

    def bootstrap(self, observable, b):
        # load (meta)data from instance
        data = self.data
        settings = self.settings

        # compute the observable from resampled data sets
        values = np.empty(b)
        for i in range(b):
            resampled_data = self._resample(data)
            new_value = self._compute(resampled_data, observable)
            values[i] = new_value

        # compute the error and return data point
        tcorr = _tcorr(data)
        value = self._compute(data, observable)
        error = np.sqrt(tcorr) * np.std(values, ddof=1)
        return DataPoint(value, error, settings)

    @staticmethod
    def _resample(data):
        # generate random indices and index into the array
        n = data.size
        indices = np.random.choice(n, size=n)
        return data[indices]

    @staticmethod
    def _compute(data, observable):

        if observable == "mean":
            return np.mean(data)
        elif observable == "std":
            return np.std(data, ddof=1)
        elif observable == "tcorr":
            return _tcorr(data)

        
    def observe(self, observable):
        data = self.data
        settings = self.settings
        value = self._compute(data, observable)
        return DataPoint(value, 0.0, settings)
    
    def histogram(self):
        data = self.data
        settings = self.settings
        shape = data.shape
        if len(shape) == 1:
            hist = np.bincount(data)
        elif len(shape) == 2:
            hist = np.sum(data, axis=0) / shape[0]
        else:
            raise ValueError
        return DataHist(hist, settings)



class DataPoint:
    value: float
    error: float
    settings: Settings

    def __init__(self, value, error, settings):
        self.value = value
        self.error = error
        self.settings = settings

class DataHist:
    hist: np.ndarray
    settings: Settings

    def __init__(self, hist, settings):
        self.hist = hist
        self.settings = settings


def correlation_analysis(series):

    # plot the trace
    plt.plot(series)
    plt.title("Trace")
    plt.xlabel(r"$\tau$")
    plt.show()

    # plot the autocovariance
    autocov = _autocov(series)
    tau = _tcorr(series)
    plt.plot(autocov)
    plt.xlim((0, 3 * tau))
    plt.ylim((0, 1))
    plt.axhline(np.exp(-1.), linestyle='--')
    plt.title("Autocovariance")
    plt.xlabel(r"$\tau$")
    plt.ylabel(
        r"$\langle (x(t) - \langle x \rangle) (x(t + \tau) - \langle x \rangle) \rangle / \sigma^2$")
    plt.show()


def _autocov(data):
    tmax = np.size(data)
    delta = data - np.mean(data)
    autocov = np.array([
        np.dot(delta[:(tmax - t)], delta[t:]) / (tmax - t)
        for t in range(tmax)
    ])

    # normalise
    if autocov[0] == 0.0:
        autocov[0] = 1.0
    else:
        autocov = autocov / autocov[0]
    return autocov


def _tcorr(data):
    autocov = _autocov(data)
    try:
        tcorr = np.argwhere(autocov < np.exp(-1))[0][0]
    except IndexError:
        tcorr = 1
    return tcorr

def process_volume_kappa(data):
    
    # load the data into usable arrays
    values = np.empty_like(data, dtype=float)
    errors = np.empty_like(data, dtype=float)
    kappas = np.empty_like(data, dtype=float)
    volumes = np.empty_like(data, dtype=int)
    labels = []
    for i in range(len(data)):
        values[i] = data[i].value
        errors[i] = data[i].error
        kappas[i] = data[i].settings.kappa
        volumes[i] = data[i].settings.volume
        if not volumes[i] in labels:
            labels.append(volumes[i])
    labels.sort()

    # define data frame
    df = pd.DataFrame()

    # add each volume to the dataframe 
    for volume in labels:

        # filter the data
        mask = volumes == volume
        x = kappas[mask]
        y = values[mask]
        yerr = errors[mask]

        # sort data
        perm = np.argsort(x)
        x = x[perm]
        y = y[perm]
        yerr = yerr[perm]

        # add data to df
        df["kappa_" + str(volume)] = x
        df["value_" + str(volume)] = y
        df["error_" + str(volume)] = yerr

    return df

def write_data_frame(df, name):
    root = os.getcwd()
    path = os.path.join(root, "plots", name + ".csv")
    df.to_csv(path, index=False)

def get_peak_data(df):
    volumes = [200, 400, 800, 1600, 3200, 6400]
    final_volumes = []
    kappas = []
    values = []
    

    for volume in volumes:
        x_data = df["kappa_" + str(volume)].to_numpy()
        y_data = df["value_" + str(volume)].to_numpy()
        i_max = np.argmax(y_data)

        # make sure the peak is not on the boundary
        if 0 < i_max < len(x_data) - 1:
        
            # assign coordinates
            x0 = x_data[i_max - 1]
            x1 = x_data[i_max]
            x2 = x_data[i_max + 1]
            y0 = y_data[i_max - 1]
            y1 = y_data[i_max]
            y2 = y_data[i_max + 1]

            # assign deltas
            dx01 = x0 - x1
            dx12 = x1 - x2
            dx20 = x2 - x0
            dy01 = y0 - y1
            dy12 = y1 - y2
            dy20 = y2 - y0

            # compute peak
            temp = x0 * dy12 + x1 * dy20 + x2 * dy01
            x = (x0 ** 2 * dy12 + x1 ** 2 * dy20 + x2 ** 2 * dy01) / (2 * temp)
            y = y0 - (dy01 * dx20 ** 2 + dy20 * dx01 ** 2) ** 2 / (4 * dx01 * dx20 * dx12 * temp)

            # add to list
            kappas.append(x)
            values.append(y)
            final_volumes.append(volume)

    return pd.DataFrame({"volume": final_volumes, "kappa": kappas, "value": values})



def volume_kappa_plot(df, exponent, invert_cmap=True, sigma=1.0, title=""):

    # prepare some plotting variables
    fig, ax = plt.subplots(1)
    cmap = cm.get_cmap('viridis')
    volumes = [200, 400, 800, 1600, 3200, 6400]

    # plot each volume line
    for volume in volumes:

        # scale the data
        norm = volume ** exponent
        x = df["kappa_" + str(volume)].to_numpy()
        y = df["value_" + str(volume)].to_numpy() / norm
        yerr = sigma * df["error_" + str(volume)].to_numpy() / norm

        # define color and depth
        depth = volumes.index(volume) / (len(volumes) - 1)
        if invert_cmap:
            cmap_index = 1 - depth
        else:
            cmap_index = depth
        depth = np.floor(depth * 1000)

        # plot 
        ax.errorbar(x, y, yerr=yerr, label=str(volume), color=cmap(cmap_index))

    # deduplicate legend
    ax.legend(
        *[*zip(*{l: h for h, l in zip(*ax.get_legend_handles_labels())}.items())][::-1])
    
    # title and axis labels
    plt.title(rf"{title} ($\nu = {exponent}$)")
    plt.xlabel(r"$\kappa$")
    plt.ylabel(r"$\mathcal{O} N_3^{-\nu}$")
    plt.show()

def hist_plot(data, volume, xmax, yscale, invert_cmap=True, title=""):

    hist_list = np.empty_like(data, dtype=np.ndarray)
    kappas = np.empty_like(data, dtype=float)
    volumes = np.empty_like(data, dtype=int)
    for i in range(len(data)):
        hist_list[i] = data[i].hist
        kappas[i] = data[i].settings.kappa
        volumes[i] = data[i].settings.volume

    mask = volumes == volume
    kappa_min = np.min(kappas[mask])
    kappa_max = np.max(kappas[mask])

    # prepare some plotting variables
    fig, ax = plt.subplots(1)
    cmap = cm.get_cmap('viridis')

    
    for i in range(len(data)):
        if mask[i]:
            kappa = data[i].settings.kappa
            hist_data = data[i].hist
            x = np.repeat(np.arange(len(hist_data) + 1), 2)
            y = np.repeat(hist_data, 2)
            y = np.insert(y, 0, 0.0)
            y = np.append(y, 0.0)
            y = y / np.sum(hist_data)
            y = yscale * y + kappa

            # define color and depth
            depth = (kappa - kappa_min) / (kappa_max - kappa_min)
            if invert_cmap:
                cmap_index = 1 - depth
            else:
                cmap_index = depth
            depth = np.floor((1 - depth) * 1000)

            # plot
            plt.fill_between(x, y, y2=kappa, color=cmap(cmap_index), zorder=depth)
            plt.hlines(kappa, 0, volume, colors=[cmap(cmap_index)], zorder=depth)
            

    # title and axis labels
    plt.title(f"{title} (N3 = {volume})")
    plt.ylabel(r"$\kappa$")
    ax.set_xlim(0, xmax)
    plt.show()