import numpy as np
import numpy.random as rng
import scipy.stats as stats

class RandomWalk:
    def __init__(self, size):
        self.growth = 0.0
        self.size = size

    def get_data(self):
        x = rng.rand(self.size) - 0.5 + self.growth

        return x
    
class TrendFollower:
    def __init__(self, size):
        self.trend = np.zeros(size)
        self.momentum = np.full(size, 0.5)

    def follow(self, x):
        for i, security in np.ndenumerate(x):
            index = i[0]
            self.trend[index] = self.momentum[index] * self.trend[index] + (1 - self.momentum[index]) * security

        return self.trend

class StdDecisionMaker:
    def __init__(self, window_size, num_sec):
        self.window_size = window_size
        self.num_sec = num_sec
        self.data = []
        self.std = 0

    def get_size(self):
        return min(len(self.data), self.window_size)


    def decide(self, x):
        self.data.append(x)
        
        f = self.fit_line()
        std = self.get_std(f)
        diff = np.asarray([f[i](self.get_size()) for i in range(self.num_sec)]) - x
        q = diff - std
        
        return q.clip(min=0)

        
    def fit_line(self):
        size = self.get_size()
        x = np.vstack([np.linspace(1, size, num=size) for i in range(self.num_sec)]).T
        y = np.stack(self.data[-size:])
        funcs = []

        for i in range(self.num_sec):
            result = stats.linregress(x[:, i], y[:, i])
            func = lambda x: result.slope*x + result.intercept
            funcs.append(func)

        return funcs

    def get_std(self, funcs):
        diffs = []
        stds = []

        size = self.get_size()
        for i in range(self.num_sec):
            for j in range(size):
                expected_value = funcs[i](j)
                real_value = self.data[j][i]
                diff = expected_value - real_value
                diffs.append(diff)

            std = np.std(diffs)
            stds.append(std)

        return np.asarray(stds)
    
class CorrelationBuyer:
    def __init__(self):
        self.data = []

    def correlate(self, x):
        self.data.append(x)

        vals = np.stack(self.data)
        cov = np.corrcoef(vals, rowvar=False)

        if type(cov) == np.float64:
            corrs = 1.0
        else:    
            for i in range(int(cov.shape[0])):
                corrs = np.matrix(cov) * np.matrix(x).T

        return np.array(corrs).squeeze()
    
class Simulator:
    def simulate(self, num_sec, length):
        randomWallStreet = RandomWalk(num_sec)
        trendFollower = TrendFollower(num_sec)
        stdMaker = StdDecisionMaker(20, num_sec)
        corrBuyer = CorrelationBuyer()

        data = np.zeros(num_sec)

        for i in range(length):
            x = randomWallStreet.get_data()
            trend = trendFollower.follow(x)
            std = stdMaker.decide(x)
            corr = corrBuyer.correlate(x) / 20

            datapoint = np.add(np.add(np.add(x, trend), std), corr)
            data = np.append(data, datapoint, axis=0)

        data = np.reshape(data, (num_sec, -1))

        return data
    
    def cum_sim(self, num_sec, length):
        return np.cumsum(self.simulate(num_sec, length), axis=1)
