#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 4/24/26, 1:37 PM. Copyright (c) The Contributors.

from unittest import TestCase

from xga.utils import MISSION_XY_UNITS, TELESCOPES


class TestUtils(TestCase):
    def test_mission_xy_units(self) -> None:
        """
        Tests that all missions supported by XGA have coordinate units defined in MISSION_XY_UNITS.
        """
        for tel in TELESCOPES:
            with self.subTest(telescope=tel):
                self.assertIn(
                    tel,
                    MISSION_XY_UNITS,
                    f"Coordinate units have not been defined for {tel}, please add them "
                    "to MISSION_XY_UNITS in xga.utils",
                )
                self.assertIn(
                    "skyxy",
                    MISSION_XY_UNITS[tel],
                    f"Sky XY unit not defined for {tel} in MISSION_XY_UNITS in xga.utils",
                )
                self.assertIn(
                    "detxy",
                    MISSION_XY_UNITS[tel],
                    f"Detector XY unit not defined for {tel} in MISSION_XY_UNITS in xga.utils",
                )
