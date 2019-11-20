#
# LSST Data Management System
# Copyright 2014 LSST Corporation.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.newhoughtransform
import numpy as np
import scipy
from skimage.feature import canny
from sklearn.cluster import KMeans
import warnings

__all__ = ["FindSatellitesConfig", "FindSatellitesTask"]


class LineProfile:
    """Construct and/or fit a model for a linear satellite trail.

    This assumes a simple model for a satellite trail, in which the trail
    follows a straight line in pixels space, with a Moffat-shaped profile. The
    model is fit to data using a Newton-Raphson style minimization algorithm.
    The initial guess for the line parameters is assumed to be fairly accurate,
    so only a narrow band of pixels around the initial line estimate is used in
    fitting the model, which provides a significant speed-up over using all the
    data. The class can also be used just to construct a model for the data with
    a line following the given coordinates.
    """
    def __init__(self, data, weights, mask, invSigmaInit, line=None):
        """
        Parameters
        ----------
        data : np.ndarray
            2d array of data
        weights : np.ndarray
            2d array of weights
        mask : np.ndarray
            2d array with mask
        invSigmaInit : float
            Initial guess for inverse of Moffat sigma parameter
        line : np.ndarray (optional)
            Guess for position of line, with datatype names `rho`, `theta`, and
            `sigma`
        """
        self.data = data
        self.weights = weights
        self.invSigmaInit = invSigmaInit
        sigma = abs(self.invSigmaInit**-1)
        ymax, xmax = data.shape
        xrange = np.arange(xmax) - xmax / 2.
        yrange = np.arange(ymax) - ymax / 2.
        self.rhoMax = ((0.5 * ymax)**2 + (0.5 * xmax)**2)**0.5
        self.xmesh, self.ymesh = np.meshgrid(xrange, yrange)
        self.mask = (weights != 0)

        if line is not None:
            # Only fit pixels within 5 sigma of the estimated line
            self.initLine = line
            radtheta = np.deg2rad(line['theta'])
            distance = (np.cos(radtheta) * self.xmesh + np.sin(radtheta) * self.ymesh - line['rho'])
            m = (abs(distance) < 5 * abs(sigma))
            self.lineMask = np.copy(self.mask) & m
        else:
            self.lineMask = np.copy(self.mask)

        self.lineMaskSize = self.lineMask.sum()
        self.maskData = data[self.lineMask]
        self.maskWeights = weights[self.lineMask]
        self.mxmesh = self.xmesh[self.lineMask]
        self.mymesh = self.ymesh[self.lineMask]

    def setLineMask(self, line):
        """Set mask around the image region near the line

        Parameters
        ----------
        line : np.ndarray
            Array with parameters `rho`, `theta`, and `invSigma`
        """
        radtheta = np.deg2rad(line['theta'])
        distance = (np.cos(radtheta) * self.xmesh + np.sin(radtheta) * self.ymesh - line['rho'])
        m = (abs(distance) < (5 / line['invSigma']))
        self.lineMask = np.copy(self.mask) & m
        self.lineMaskSize = self.lineMask.sum()
        self.maskData = self.data[self.lineMask]
        self.maskWeights = self.weights[self.lineMask]
        self.mxmesh = self.xmesh[self.lineMask]
        self.mymesh = self.ymesh[self.lineMask]

    def makeMaskedProfile(self, rho, theta, invSigma, fitFlux=True):
        """Construct the line model in the masked region and calculate its
        derivatives

        Parameters
        ----------
        rho : float
            Distance of line from center of image
        theta : float
            Angle of line
        invSigma : float
            Inverse Moffat sigma parameter of line
        fitFlux : bool
            Fit the amplitude of the line profile to the data

        Returns
        -------
        model : np.ndarray
            Model in the masked region
        dModel : np.ndarray
            Derivative of the model in the masked region
        """
        # Calculate distance between pixels and line
        radtheta = np.deg2rad(theta)
        costheta = np.cos(radtheta)
        sintheta = np.sin(radtheta)
        distance = (costheta * self.mxmesh + sintheta * self.mymesh - rho)
        distanceSquared = distance**2

        # Calculate partial derivatives of distance
        drad = np.pi / 180
        dDistanceSqdRho = 2 * distance * (-np.ones_like(self.mxmesh))
        dDistanceSqdTheta = (2 * distance * (-sintheta * self.mxmesh + costheta * self.mymesh) * drad)

        # Use pixel-line distances to make Moffat profile
        profile = (1 + distanceSquared * invSigma**2)**-2.5
        dProfile = -2.5 * (1 + distanceSquared * invSigma**2)**-3.5

        if fitFlux:
            # Calculate line flux from profile and data
            f = ((self.maskWeights * self.maskData * profile).sum() / (self.maskWeights * profile**2).sum())
        else:
            # Approximately normalize the line
            f = invSigma**-1
        if np.isnan(f):
            f = 0

        model = f * profile

        # Calculate model derivatives
        dModeldRho = f * dProfile * dDistanceSqdRho * invSigma**2
        dModeldTheta = f * dProfile * dDistanceSqdTheta * invSigma**2
        dModeldSigma = f * dProfile * distanceSquared * 2 * invSigma

        dModel = np.array([dModeldRho, dModeldTheta, dModeldSigma])
        return model, dModel

    def makeProfile(self, rho, theta, invSigma, fitFlux=True):
        """Construct the line profile model

        Parameters
        ----------
        rho : float
            Distance of line from center of image
        theta : float
            Angle of line
        invSigma : float
            Inverse Moffat sigma parameter of line
        fitFlux : bool
            Fit the amplitude of the line profile to the data

        Returns
        -------
        finalModel : np.ndarray
            Model for line profile
        """
        model, dmodel = self.makeMaskedProfile(rho, theta, invSigma, fitFlux=fitFlux)
        finalModel = np.zeros_like(self.data)
        finalModel[self.lineMask] = model
        return finalModel

    def lineChi2(self, x, grad=True):
        """Construct the chi2 between the data and the model

        Parameters
        ----------
        x : np.ndarray
            Rho, theta, and inverse sigma parameters of the line model
        grad : bool (optional)
            Whether or not to return the gradient and hessian

        Returns
        -------
        reducedChi : float
            Reduced chi2 of the model
        reducedDChi : np.ndarray
            Derivative of the chi2 with respect to rho, theta, invSigma
        reducedHessianChi : np.ndarray
            Hessian of the chi2 with respect to rho, theta, invSigma
        """
        rho, theta, invSigma = x
        # Calculate chi2
        model, dModel = self.makeMaskedProfile(rho, theta, invSigma)
        chi2 = (self.maskWeights * (self.maskData - model)**2).sum()
        if not grad:
            return chi2.sum() / self.lineMaskSize

        # Calculate derivative and Hessian of chi2
        derivChi2 = ((-2 * self.maskWeights * (self.maskData - model))[None, :] * dModel).sum(axis=1)
        hessianChi2 = (2 * self.maskWeights * dModel[:, None, :] * dModel[None, :, :]).sum(axis=2)

        reducedChi = chi2 / self.lineMaskSize
        reducedDChi = derivChi2 / self.lineMaskSize
        reducedHessianChi = hessianChi2 / self.lineMaskSize
        return reducedChi, reducedDChi, reducedHessianChi

    def fit(self, dChi2Tol=0.1):
        """Perform Newton-Raphson minimization to find line parameters

        Parameters
        ----------
        dChi2Tol : float (optional)
            Change in Chi2 tolerated for fit convergence

        Returns
        -------
        outline : np.ndarray
            Coordinates and inverse width of fit line
        chi2 : float
            Reduced Chi2 of model fit to data
        fitSuccess : bool
            Boolean where `True` corresponds to a successful  fit
        """

        x = np.array([self.initLine['rho'], self.initLine['theta'], self.invSigmaInit])
        dChi2 = 1
        iter = 0
        oldChi2 = 0
        fitSuccess = True
        while abs(dChi2) > dChi2Tol:
            chi2, b, A = self.lineChi2(x)
            if chi2 == 0:
                break
            dChi2 = oldChi2 - chi2
            cholesky = scipy.linalg.cho_factor(A)
            dx = scipy.linalg.cho_solve(cholesky, b)

            def line_search(c):
                testx = x - c * dx
                return self.lineChi2(testx, grad=False)
            factor, fmin, _, _ = scipy.optimize.brent(line_search, full_output=True, tol=0.05)
            x -= factor * dx
            if x[0] > 1.5 * self.rhoMax:
                fitSuccess = False
                break
            oldChi2 = chi2
            iter += 1
        outline = np.core.records.fromarrays(x, dtype=[('rho', float), ('theta', float), ('invSigma', float)])
        outline.invSigma = abs(outline.invSigma)
        return outline, chi2, fitSuccess


class FindSatellitesConfig(pexConfig.Config):
    """Configuration parameters for `FindSatellitesTask`
    """
    minimumKernelHeight = pexConfig.Field(
        doc="minimum height of the satellite finding kernel relative to the tallest kernel",
        dtype=float,
        default=0.0,
    )
    absMinimumKernelHeight = pexConfig.Field(
        doc="minimum absolute height of the satellite finding kernel",
        dtype=float,
        default=5,
    )
    clusterMinimumSize = pexConfig.Field(
        doc="minimum size in pixels of detected clusters",
        dtype=int,
        default=50,
    )
    clusterMinimumDeviation = pexConfig.Field(
        doc="allowed deviation (in pixels) from a straight line for a detected "
            "line",
        dtype=int,
        default=2,
    )
    delta = pexConfig.Field(
        doc="stepsize in angle-radius parameter space",
        dtype=float,
        default=0.2,
    )
    nSigmas = pexConfig.Field(
        doc="Number of sigmas from center of kernel to include in voting "
            "procedure",
        dtype=float,
        default=2,
    )
    rhoBin = pexConfig.Field(
        doc="Binsize in pixels for position parameter rho when finding "
            "clusters of detected lines",
        dtype=float,
        default=30,
    )
    thetaBin = pexConfig.Field(
        doc="Binsize in degrees for angle parameter theta when finding "
            "clusters of detected lines",
        dtype=float,
        default=2,
    )
    minImageSignaltoNoise = pexConfig.Field(
        doc="Boundary in signal-to-noise between non-detections and detections "
            "for making a binary image from the original input image",
        dtype=float,
        default=5,
    )
    invSigma = pexConfig.Field(
        doc="Moffat sigma parameter (in units of pixels) describing the "
            "profile of the satellite trail",
        dtype=float,
        default=10.**-1,
    )
    imageBinning = pexConfig.Field(
        doc="Number of pixels by which to bin image",
        dtype=int,
        default=2,
    )
    footprintThreshold = pexConfig.Field(
        doc="Threshold at which to determine edge of line, in units of the line"
            "profile maximum",
        dtype=float,
        default=0.01
    )
    dChi2Tolerance = pexConfig.Field(
        doc="Absolute difference in Chi2 between iterations of line profile"
            "fitting that is acceptable for convergence",
        dtype=float,
        default=0.1
    )


class FindSatellitesTask(pipeBase.Task):
    """Find satellite trails or other straight lines in image data.

    Satellites passing through the field of view of the telescope leave a
    bright trail in images. This class uses the Kernel Hough Transform (KHT)
    (Fernandes and Oliveira, 2007), implemented in `lsst.houghtransform`. The
    procedure works by taking a binary image, either provided as put or produced
    from the input data image, using a Canny filter to make an image of the
    edges in the original image, then running the KHT on the edge image. The KHT
    identifies clusters of non-zero points, breaks those clusters of points into
    straight lines, keeps clusters with a size greater than the user-set
    threshold, then performs a voting procedure to find the best-fit coordinates
    of any straight lines. Given the results of the KHT algorithm, clusters of
    lines are identified and grouped (generally these correspond to the two
    edges of a satellite trail) and a profile is fit to the satellite trail in
    the original (non-binary) image.
    """

    ConfigClass = FindSatellitesConfig
    _DefaultName = "findSatellitesConfig"

    def __init__(self, *args, **kwargs):
        """Tmp init description
        """
        pipeBase.Task.__init__(self, *args, **kwargs)

    @pipeBase.timeMethod
    def run(self, maskedImage, binaryImage=None):
        """Find and mask satellite trails in a masked image

        Parameters
        ----------
        maskedImage : `lsst.afw.image.Exposure`
            The image in which to search for satellite trails.
        binaryImage : `np.ndarray` (optional)
            Optional binary image of significant detections in `maskedImage`

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Result struct with components:

            - ``originalLines``: lines identified by kernel hough transform
            - ``lineClusters``:  lines grouped into clusters in rho-theta space
            - ``lines``: final result for lines after line-profile fit
            - ``mask``: 2-d boolean mask where detected lines=0
        """
        if binaryImage is None:
            maskData = self.processMaskedImage(maskedImage)
        else:
            maskData = binaryImage.array
            if not (maskData.shape == maskedImage.image.array.shape):
                raise ValueError("maskedImage and binaryImage data arrays must have same shape")
            if not (maskData == maskData.astype(bool)).all():
                raise ValueError("binaryImage must be composed of 0s and 1s or booleans")

        self.edges = self.cannyFilter(maskData)
        self.lines = self.runKHT(self.edges)

        if (binaryImage is None) and (self.config.imageBinning is not None):
            # Renormalize line coordinates to original image:
            self.lines['rho'] *= self.config.imageBinning
            tmpMaskData = maskData.repeat(self.config.imageBinning, axis=0)
            maskData = tmpMaskData.repeat(self.config.imageBinning, axis=1)
        if len(self.lines) == 0:
            lineMask = np.zeros_like(maskData, dtype=bool)
            fitLines = np.empty(0, dtype=[('rho', float), ('theta', float), ('votes', int)])
            clusters = np.empty(0, dtype=[('rho', float), ('theta', float), ('votes', int)])
        else:
            clusters = self.findClusters(self.lines)
            fitLines, lineMask = self.fitProfile(clusters, maskedImage)

        # The output mask is the intersection of the fit satellite trails and the image detections
        outputMask = lineMask & maskData.astype(bool)

        return pipeBase.Struct(
            lines=fitLines,
            lineClusters=clusters,
            originalLines=self.lines,
            mask=outputMask,
        )

    def processMaskedImage(self, maskedImage, checkMethod=False):
        """Make binary image array from maskedImage object

        Convert input to a binary image by setting all data with signal-to-noise
        below some threshold to zero, and all data above the threshold to one.
        If the imageBinning config parameter has been set, this procedure will
        be preceded by a weighted binning of the data.

        Parameters
        ----------
        maskedImage : `lsst.afw.image.Exposure`
            Image to be (optionally) binned and converted

        Returns
        -------
        out_data : `np.ndarray`
            2-d binary image of pixels above the signal-to-noise threshold.
        """
        binning = self.config.imageBinning

        data = maskedImage.image.array
        weights = 1 / maskedImage.variance.array
        mask = maskedImage.getMask()

        detectionMask = ((mask.array & mask.getPlaneBitMask("DETECTED")))
        badMask = ((mask.array & mask.getPlaneBitMask("NO_DATA")) |
                   (mask.array & mask.getPlaneBitMask("INTRP")) |
                   (mask.array & mask.getPlaneBitMask("BAD")) |
                   (mask.array & mask.getPlaneBitMask("SAT")) |
                   (mask.array & mask.getPlaneBitMask("EDGE"))).astype(bool)

        fitMask = detectionMask.astype(bool) & ~(badMask.astype(bool))
        fitData = np.copy(data)
        fitData[~fitMask] = 0
        fitWeights = np.copy(weights)
        fitWeights[~fitMask] = 0

        if binning is not None:
            # Do weighted binning:
            ymax, xmax = fitData.shape
            if (ymax % binning == 0) & (xmax % binning == 0) & (not checkMethod):
                # Faster binning method
                binNumeratorReshape = (fitData * fitWeights).reshape(ymax // binning, binning,
                                                                     xmax // binning, binning)
                binDenominatorReshape = fitWeights.reshape(binNumeratorReshape.shape)
                binnedNumerator = binNumeratorReshape.sum(axis=3).sum(axis=1)
                binnedDenominator = binDenominatorReshape.sum(axis=3).sum(axis=1)
            else:
                # Slower binning method when (image shape mod binsize) != 0
                warnings.warn('Using slow binning method--consider choosing a binsize that evenly divides '
                              'into the image size, so that %s mod self.config.imageBinning  == 0 ' % ymax +
                              'and %s mod self.config.imageBinning == 0' % xmax)
                xarray = np.arange(xmax)
                yarray = np.arange(ymax)
                xmesh, ymesh = np.meshgrid(xarray, yarray)
                xbins = np.arange(0, xmax + binning, binning)
                ybins = np.arange(0, ymax + binning, binning)
                numerator = fitWeights * fitData
                binnedNumerator, *rest = scipy.stats.binned_statistic_2d(ymesh.ravel(), xmesh.ravel(),
                                                                         numerator.ravel(), statistic='sum',
                                                                         bins=(ybins, xbins))
                binnedDenominator, *rest = scipy.stats.binned_statistic_2d(ymesh.ravel(), xmesh.ravel(),
                                                                           fitWeights.ravel(),
                                                                           statistic='sum',
                                                                           bins=(ybins, xbins))
            binnedData = np.zeros_like(binnedNumerator)
            ind = binnedDenominator != 0
            binnedData[ind] = binnedNumerator[ind] / binnedDenominator[ind]
            binnedWeight = binnedDenominator
            binMask = (binnedData * binnedWeight**0.5) > self.config.minImageSignaltoNoise
            outputData = np.zeros_like(binnedData)
            outputData[binMask] = 1
        else:
            mask = (fitMask * fitWeights**0.5) > self.config.minImageSignaltoNoise
            outputData = np.zeros_like(fitData)

        return outputData

    def cannyFilter(self, image):
        """Apply a canny filter to the data in order to detect edges

        Parameters
        ----------
        image : `np.ndarray`
            2-d image data on which to run filter

        Returns
        -------
        cannyData : `np.ndarray`
            2-d image of edges found in input image
        """

        filterData = np.array(image, dtype=np.uint8)
        cannyData = canny(filterData, low_threshold=0, high_threshold=1, sigma=0.1)
        return cannyData

    def runKHT(self, image):
        """Run Kernel Hough Transform on image.

        Parameters
        ----------
        image : `np.ndarray`
            2-d image data on which to detect lines

        Returns
        -------
        lines : `np.ndarray'
            Array containing line parameter fields ``rho`` and ``theta``
        """

        lines = lsst.newhoughtransform.find_satellites(image, self.config.clusterMinimumSize,
                                                       self.config.clusterMinimumDeviation, self.config.delta,
                                                       self.config.minimumKernelHeight, self.config.nSigmas,
                                                       self.config.absMinimumKernelHeight)

        return lines

    def findClusters(self, lines):
        """Group lines that are close in parameter space and likely describe
        the same satellite trail

        Parameters
        ----------
        lines : `np.ndarray`
            Array with fields "rho" and "theta"

        Returns
        -------
        result : `np.ndarray`
            Array with fields "rho" and "theta"
        """

        # Renormalize variables so that expected standard deviation in a
        # cluster is 1.
        x = lines['rho'] / self.config.rhoBin
        y = lines['theta'] / self.config.thetaBin
        X = np.array([x, y]).T
        nClusters = 1

        # Put line parameters in clusters by starting with all in one, then
        # subdividing until the parameters of each cluster have std. dev=1
        while True:
            kmeans = KMeans(n_clusters=nClusters).fit(X)
            clusterStandardDeviations = np.zeros((nClusters, 2))
            for c in range(nClusters):
                inCluster = X[kmeans.labels_ == c]
                clusterStandardDeviations[c] = np.std(inCluster, axis=0)
            # Are cluster rhos and thetas all close?
            if (clusterStandardDeviations <= 1).all():
                break
            nClusters += 1

        # The cluster centers are final line estimates
        finalClusters = kmeans.cluster_centers_.T

        # Sum votes from lines going into clusters:
        clusterVotes = np.zeros(nClusters)
        for c in range(nClusters):
            ind = kmeans.labels_ == c
            clusterVotes[c] = lines['votes'][ind].sum()
        finalClustersInfo = np.vstack([finalClusters, clusterVotes])
        result = np.core.records.fromarrays(finalClustersInfo, dtype=[('rho', float), ('theta', float),
                                                                      ('votes', int)])

        # Rescale cluster centers
        result['rho'] *= self.config.rhoBin
        result['theta'] *= self.config.thetaBin

        return result

    def fitProfile(self, lines, maskedImage):
        """Fit the profile of the satellite trail.

        Given the initial parameters of detected lines, fit a model for the
        satellite trail to the original (non-binary image). The assumed model
        is a straight line with a Moffat profile.

        Parameters
        ----------
        lines : `np.ndarray`
            Array of lines with fields "rho" and "theta"
        maskedImage : `lsst.afw.image.Exposure`
            Original image to be used to fit profile of satellite trail.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Result struct with components:

            - ``linefits``: np.ndarray with fields "rho", "theta", and "sigma"
            - ``finalMask``: 2d np.ndarray with detected satellite trails
                masked out.
        """
        invSigmaInit = self.config.invSigma
        data = maskedImage.image.array
        weights = maskedImage.variance.array**-1
        ymax, xmax = data.shape
        xrange = np.arange(xmax) - xmax / 2.
        yrange = np.arange(ymax) - ymax / 2.
        xmesh, ymesh = np.meshgrid(xrange, yrange)
        mask = (weights != 0) & np.isfinite(weights)

        lineFits = []
        finalLineMasks = [np.zeros(data.shape, dtype=bool)]
        for line in lines:
            lineModel = LineProfile(data, weights, mask, invSigmaInit, line=line)
            # Skip any lines that do not cover any data (sometimes happens because of chip gaps)
            if lineModel.lineMaskSize == 0:
                continue

            fit, chi2, fitSuccess = lineModel.fit(dChi2Tol=self.config.dChi2Tolerance)
            sigma = 1 / fit.invSigma

            # Initial estimate should be quite close: fit is deemed unsuccessful if rho or theta
            # change more than the allowed bin in rho or theta:
            if ((abs(fit.rho - line.rho) > 2 * self.config.rhoBin) or
               (abs(fit.theta - line.theta) > 2 * self.config.thetaBin)):
                fitSuccess = False

            if not fitSuccess:
                continue

            # Make mask
            lineModel.setLineMask(fit)
            finalModel = lineModel.makeProfile(fit.rho, fit.theta, fit.invSigma)
            # Take absolute value, as trails are allowed to be negative
            finalModelMax = (abs(finalModel)).max()
            finalLineMask = abs(finalModel) > self.config.footprintThreshold
            lineFits.append([fit.rho, fit.theta, sigma, chi2, finalModelMax])
            finalLineMasks.append(finalLineMask)

        if len(lineFits) == 0:
            lineFits = np.empty(0, dtype=[('rho', float), ('theta', float), ('sigma', float), ('chi2', float),
                                          ('fluxMax', float)])
        else:
            lineFits = np.core.records.fromarrays(np.array(lineFits).T, dtype=[('rho', float),
                                                                               ('theta', float),
                                                                               ('sigma', float),
                                                                               ('chi2', float),
                                                                               ('fluxMax', float)])

        finalMask = np.array(finalLineMasks).any(axis=0)

        return lineFits, finalMask
