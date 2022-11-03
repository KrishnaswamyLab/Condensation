import os, copy, errno

import numpy as np
import numpy.matlib

from scipy.io import savemat
from scipy.spatial.distance import pdist

class ContractionClustering:

    def __init__(self, data, channels, savepath, **kwargs):

        self.sampleSize = np.shape(data)[0]

        # Data points at the current contraction step
        self.dataContracted = copy.deepcopy(data)

        # Sequence of data points at each contraction step
        self.contractionSequence = []

        # Sample indices at the current contraction step
        self.sampleIndices = dict()
        for ii in range(self.sampleSize):
            self.sampleIndices[ii] = [ii]

        self.channels = channels

        self.destination = None
        if os.path.isdir(savepath):
            self.destination = savepath
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), savepath)

        self.currentSigma = np.Inf
        if "initialSigma" in kwargs:
            self.currentSigma = kwargs.get("initialSigma")
        else:
            distanceMatrix = pdist(data, 'euclidean')
            self.currentSigma = np.mean(distanceMatrix) - np.std(distanceMatrix)

        # Populate expert labels
        if "expertLabels" in kwargs:
            assert(self.sampleSize == len(kwargs.get("expertLabels")))
            self.expertLabels = kwargs.get("expertLabels")
        else:
            self.expertLabels = np.ones((self.sampleSize,1))

        # Calculate epsilon
        self.epsilon = np.max(np.max(self.dataContracted)-np.min(self.dataContracted))/10000

        self.iteration = 0
        self.maxNumContractionSteps = 1
        self.numDiffusionSteps = 3

        self.eigenvalueSequence = []
        self.clusterStats = {}
        self.currentEigenvectors = []
        self.currentEigenvalues = []
        self.clusterAssignments = []

        self.normalizedAffinityMatrix = []
        self.normalizedAffinityMatrixInitialSigma = []
        self.weights = []

    def contract(self):

        self.steps(self.maxNumContractionSteps)

    def steps(self, nsteps=1):

        if self.iteration == 0:
            self.iteration = 1

        for itr in range(self.iteration, self.iteration+nsteps):

            self.iteration = itr
            self.contractionSequence.append(self.inflateClusters())
            self.calcAffinities()
            self.saveAffinityMatrix()
            self.spectralDecompose()
            self.performContractionMove()
            self.mergeEpsilonClusters()
            self.assignClusters()
            self.controlSigma()
            if self.checkTerminationCondition():
                break

    def inflateClusters(self):
        inflatedContractionMatrix = None
        for i in range(self.dataContracted.shape[0]):
            currentSampleIndices = self.sampleIndices[i]
            inflatedContractionMatrix[currentSampleIndices,:] = np.matlib.repmat(self.dataContracted[i,:], len(currentSampleIndices), 1)
        return inflatedContractionMatrix

    def saveAffinityMatrix(self):
        savemat(self.destination + str(self.iteration)+"-affinity-matrix.mat", 
                {"affinity": self.normalizedAffinityMatrix, "epsilon": self.epsilon})
