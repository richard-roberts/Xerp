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

    def interpolate_one(self, pose):
        vs = []
        for s in self.source:
            dist = np.linalg.norm(pose - s)
            vs.append(self.kernel(dist))

        result = np.matrix(vs) * self.weights
        return result.tolist()[0]

    def interpolate_many(self, poses):
        m = []
        for pose in poses:
            row = []
            for s in self.source:
                dist = np.linalg.norm(pose - s)
                row.append(self.kernel(dist))
            m.append(row)

        result = np.matrix(m) * self.weights
        return result   

    def interpolate_many_fast(self, poses):
        n_s = len(self.source)
        n_p = len(poses)
        P = np.array([poses,] * n_s).transpose((1, 0, 2))
        S = np.array([self.source,] * n_p)
        PS = P - S
        normed = np.sqrt((PS * PS).sum(axis=2))
        guass = np.exp(-np.power(normed, 2) / (2 * np.power(self.sigma, 2)))
        result = np.matrix(guass) * self.weights
        return result.tolist()
    