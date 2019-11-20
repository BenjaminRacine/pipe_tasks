import unittest
import numpy as np

import lsst.utils.tests

import lsst.afw.image
from lsst.pipe.tasks.findSatellites import FindSatellitesTask, LineProfile


class TestFindSatellites(lsst.utils.tests.TestCase):

    def setUp(self):
        self.fst = FindSatellitesTask()
        self.fst.config.dChi2Tolerance = 1e-10
        self.testx = 500
        self.testy = 600
        self.exposure = lsst.afw.image.ExposureF(self.testy, self.testx)
        self.exposure.maskedImage.image.array = np.random.randn(self.testx, self.testy).astype(np.float32)
        self.exposure.maskedImage.variance.array = np.ones((self.testx, self.testy)).astype(np.float32)

    def test_input(self):
        """Test that binaryImage optional argument is binary and right shape"""
        wrongshape_binaryImage = lsst.afw.image.ImageU(self.testx, self.testy + 100)
        with self.assertRaises(ValueError):
            self.fst.run(self.exposure, wrongshape_binaryImage)

        notbool_binaryImage = lsst.afw.image.ImageF(self.testy, self.testx)
        notbool_binaryImage.array = np.random.random((self.testx, self.testy)).astype(np.float32)
        with self.assertRaises(ValueError):
            self.fst.run(self.exposure, notbool_binaryImage)

    def test_binning(self):
        """Test the two binning methods and the no-binning method"""

        self.assertEqual(self.testx % self.fst.config.imageBinning, 0)
        self.assertEqual(self.testy % self.fst.config.imageBinning, 0)

        reshapeBinning = self.fst.processMaskedImage(self.exposure)
        with self.assertWarns(Warning):
            scipyBinning = self.fst.processMaskedImage(self.exposure, checkMethod=True)
        self.assertAlmostEqual(reshapeBinning.tolist(), scipyBinning.tolist())

        self.fst.config.imageBinning = None
        nobinImage = self.fst.processMaskedImage(self.exposure)
        self.assertEqual(nobinImage.shape, self.exposure.image.array.shape)
        self.assertEqual(nobinImage.tolist(), nobinImage.astype(bool).tolist())

    def test_canny(self):
        """Test that Canny filter returns binary of equal shape"""

        zeroExposure = lsst.afw.image.ExposureF(self.testy, self.testx)
        cannyZeroExposure = self.fst.cannyFilter(zeroExposure.image.array)
        self.assertEqual(cannyZeroExposure.tolist(), zeroExposure.image.array.tolist())

        processedImage = self.fst.processMaskedImage(self.exposure)
        cannyNonZero = self.fst.cannyFilter(processedImage)
        self.assertEqual(cannyNonZero.tolist(), cannyNonZero.astype(bool).tolist())

    def test_runkht(self):
        """Test the whole thing"""

        # Empty image:
        zeroArray = np.zeros((self.testx, self.testy))
        zeroLines = self.fst.runKHT(zeroArray)
        self.assertEqual(len(zeroLines), 0)
        testExposure = self.exposure.clone()
        result = self.fst.run(testExposure)
        self.assertEqual(len(result.lines), 0)
        self.assertEqual(result.mask.tolist(), zeroArray.tolist())

        # Make image with line and check that input line is recovered:
        testExposure = self.exposure.clone()
        input = np.array((150, 45, 3), dtype=[('rho', float), ('theta', float), ('sigma', float)])
        lineProfile = LineProfile(testExposure.image.array, testExposure.variance.array**-1,
                                  np.zeros_like(testExposure.image.array), input['sigma']**-1,)
        testData = lineProfile.makeProfile(input['rho'], input['theta'], input['sigma']**-1, fitFlux=False)
        testExposure.maskedImage.image.array = testData
        binaryImage = lsst.afw.image.ImageF(testExposure.getWidth(), testExposure.image.getHeight())
        binaryImage.array = (abs(testData) > 0.1 * (abs(testData)).max()).astype(np.float32)

        result = self.fst.run(testExposure, binaryImage=binaryImage)
        self.assertEqual(len(result.lines), 1)
        self.assertLess(abs(input['rho'] - result.lines[0]['rho']), 0.01)
        self.assertLess(abs(input['theta'] - result.lines[0]['theta']), 0.01)
        self.assertLess(abs(input['sigma'] - result.lines[0]['sigma']), 0.01)


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
