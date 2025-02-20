import unittest

from astropy.units import Quantity

import xga
from xga.sources import GalaxyCluster
from xga.generate.esass import evtool_image
from xga.products.phot import Image
from xga.products.spec import Spectrum
from xga.generate.esass import srctool_spectrum

from .. import SRC_ALL_TELS


class TestGalaxyCluster(unittest.TestCase):
    def test_r500(self):
        assert SRC_ALL_TELS.r500 == Quantity(500, 'kpc')