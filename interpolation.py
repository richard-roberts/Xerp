import numpy as np


class ScatteredDataInterpolation:

    def __init__(self, source, target, sigma):
        self.source = [np.array(v) for v in source]
        self.target = [np.array(v) for v in target]
        self.sigma = sigma

        n = len(self.source)
        A = np.matrix(np.zeros((n, n)))
        for (i, pose_i) in enumerate(self.source):
            for (j, pose_j) in enumerate(self.source):
                dist = np.linalg.norm(pose_i - pose_j)
                A[i, j] = self.kernel(dist)
        
        b = np.vstack(self.target)
        self.weights = (A.T * A).I * A.T * b

    def kernel(self, dist):
        numer = -np.power(dist, 2)
        denom = 2 * np.power(self.sigma, 2)
        return np.exp(numer / denom)

    def interpolate(self, pose):
        vs = []
        for s in self.source:
            dist = np.linalg.norm(pose - s)
            vs.append(self.kernel(dist))

        result = np.matrix(vs) * self.weights
        return result.tolist()[0]
