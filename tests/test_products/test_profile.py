#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 5/20/26, 1:00 PM. Copyright (c) The Contributors.

"""Tests for 1D profile viewing and thermodynamic profile products."""

import os
import unittest
from typing import Any
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
from astropy.constants import k_B, m_p
from astropy.cosmology import FlatLambdaCDM
from astropy.units import Quantity, Unit, UnitConversionError

from xga.exceptions import XGAFitError
from xga.models.density import KingProfile1D
from xga.models.temperature import SimpleVikhlininTemperature1D
from xga.products.profile import (
    APECNormalisation1D,
    BaryonFraction,
    GasDensity3D,
    GasTemperature3D,
    HydrostaticMass,
    ProjectedGasTemperature1D,
    SpecificEntropy,
    ThermalPressure,
)
from xga.sourcetools.misc import rad_to_ang
from xga.utils import MEAN_MOL_WEIGHT

from .. import MISC_OUTPUT_TESTS


class TestProfileView(unittest.TestCase):
    """Test cases for 1D profile and aggregate 1D profile view saving."""

    tp_one: ProjectedGasTemperature1D
    tp_two: ProjectedGasTemperature1D
    tp_agg: Any

    @classmethod
    def setUpClass(cls) -> None:
        """Initialise synthetic projected gas temperature profiles and their aggregate for view tests."""
        radii = Quantity([100, 200, 300, 400], "kpc")
        deg_radii = rad_to_ang(radii, z=0.16)

        cls.tp_one = ProjectedGasTemperature1D(
            radii,
            Quantity([1, 3, 4, 3.4], "keV"),
            Quantity([0, 0], "deg"),
            "magic-galaxy-cluster",
            "an-obsid",
            "an-inst",
            deg_radii=deg_radii,
        )

        cls.tp_two = ProjectedGasTemperature1D(
            radii,
            Quantity([10, 8, 7.1, 3.4], "keV"),
            Quantity([1, 1], "deg"),
            "another-magic-galaxy-cluster",
            "an-obsid",
            "an-inst",
            deg_radii=deg_radii,
        )

        cls.tp_agg = cls.tp_one + cls.tp_two

    def test_default_view(self) -> None:
        """Test saving a default view of a single 1D profile."""
        test_out_path = os.path.join(MISC_OUTPUT_TESTS, self.id())
        os.makedirs(test_out_path, exist_ok=True)

        self.tp_one.save_view(os.path.join(test_out_path, "tp_one_default.png"))

    def test_default_aggregate_view(self) -> None:
        """Test saving a default view of an aggregate 1D profile."""
        test_out_path = os.path.join(MISC_OUTPUT_TESTS, self.id())
        os.makedirs(test_out_path, exist_ok=True)

        self.tp_agg.save_view(os.path.join(test_out_path, "tp_one_two_agg_default.png"))

    def test_draw_rad_val_view(self) -> None:
        """Test saving a single 1D profile view with vertical radius and horizontal value annotations."""
        test_out_path = os.path.join(MISC_OUTPUT_TESTS, self.id())
        os.makedirs(test_out_path, exist_ok=True)

        self.tp_one.save_view(
            os.path.join(test_out_path, "tp_one_draw_rad.png"), draw_rads={r"$R_{2500}$": Quantity(134, "kpc")}
        )

        self.tp_one.save_view(
            os.path.join(test_out_path, "tp_one_draw_val.png"), draw_vals={r"$T_{2500}$": Quantity(2.3, "keV")}
        )

        self.tp_one.save_view(
            os.path.join(test_out_path, "tp_one_draw_val_werr.png"),
            draw_vals={r"$T_{2500}$": Quantity([2.3, 0.5, 0.1], "keV")},
        )

        self.tp_one.save_view(
            os.path.join(test_out_path, "tp_one_draw_rad_val.png"),
            draw_rads={r"$R_{2500}$": Quantity(134, "kpc")},
            draw_vals={r"$T_{2500}$": Quantity(2.3, "keV")},
        )

    def test_draw_rad_val_aggregate_view(self) -> None:
        """Test saving an aggregate 1D profile view with vertical radius and horizontal value annotations."""
        test_out_path = os.path.join(MISC_OUTPUT_TESTS, self.id())
        os.makedirs(test_out_path, exist_ok=True)

        self.tp_agg.save_view(
            os.path.join(test_out_path, "tp_agg_draw_rad.png"), draw_rads={r"$R_{2500}$": Quantity(134, "kpc")}
        )

        self.tp_agg.save_view(
            os.path.join(test_out_path, "tp_agg_draw_val.png"), draw_vals={r"$T_{2500}$": Quantity(2.3, "keV")}
        )

        self.tp_agg.save_view(
            os.path.join(test_out_path, "tp_agg_draw_val_werr.png"),
            draw_vals={r"$T_{2500}$": Quantity([2.3, 0.5, 0.1], "keV")},
        )

        self.tp_agg.save_view(
            os.path.join(test_out_path, "tp_agg_draw_rad_val.png"),
            draw_rads={r"$R_{2500}$": Quantity(134, "kpc")},
            draw_vals={r"$T_{2500}$": Quantity(2.3, "keV")},
        )


class TestThermodynamicProfiles(unittest.TestCase):
    """Synthetic unit tests for BaseProfile1D radius ordering and thermodynamic profiles."""

    z: float = 0.16
    centre: Quantity
    src_name: str = "synthetic-cluster"
    obs_id: str = "0123456789"
    inst: str = "pn"
    num_samples: int = 200

    r_fine: Quantity
    r_fine_err: Quantity
    deg_fine: Quantity

    r_coarse: Quantity
    r_coarse_err: Quantity
    deg_coarse: Quantity

    r_extended: Quantity
    r_extended_err: Quantity
    deg_extended: Quantity

    temp_model_ref: SimpleVikhlininTemperature1D
    dens_model_ref: KingProfile1D

    @classmethod
    def setUpClass(cls) -> None:
        """Initialise analytical reference models and radial grids for synthetic thermodynamic profile tests."""
        cls.centre = Quantity([0.0, 0.0], "deg")

        # 6-bin fine grid covering annulus bounds [70, 430] kpc
        cls.r_fine = Quantity([100.0, 160.0, 220.0, 280.0, 340.0, 400.0], "kpc")
        cls.r_fine_err = Quantity(np.full(len(cls.r_fine), 30.0), "kpc")
        cls.deg_fine = rad_to_ang(cls.r_fine, z=cls.z)

        # 3-bin coarse grid covering identical outer annulus bounds [70, 430] kpc
        cls.r_coarse = Quantity([130.0, 250.0, 370.0], "kpc")
        cls.r_coarse_err = Quantity(np.full(len(cls.r_coarse), 60.0), "kpc")
        cls.deg_coarse = rad_to_ang(cls.r_coarse, z=cls.z)

        # 7-bin extended grid where the outermost bin (460 kpc) lies beyond the coarse outer bound (430 kpc)
        cls.r_extended = Quantity([100.0, 160.0, 220.0, 280.0, 340.0, 400.0, 460.0], "kpc")
        cls.r_extended_err = Quantity(np.full(len(cls.r_extended), 30.0), "kpc")
        cls.deg_extended = rad_to_ang(cls.r_extended, z=cls.z)

        # Analytical reference models that decrease monotonically across [70, 500] kpc so hydrostatic masses are > 0
        cls.temp_model_ref = SimpleVikhlininTemperature1D(
            x_unit="kpc",
            y_unit="keV",
            cust_start_pars=[
                Quantity(50.0, "kpc"),
                Quantity(1.0, ""),
                Quantity(3.0, "keV"),
                Quantity(6.0, "keV"),
                Quantity(200.0, "kpc"),
                Quantity(1.0, ""),
            ],
        )
        cls.dens_model_ref = KingProfile1D(
            x_unit="kpc",
            y_unit="1/cm^3",
            cust_start_pars=[
                Quantity(0.5, ""),
                Quantity(150.0, "kpc"),
                Quantity(1e-2, "1/cm^3"),
            ],
        )

    def _make_temp_model(self) -> SimpleVikhlininTemperature1D:
        """Create a fitted SimpleVikhlininTemperature1D instance with deterministic parameter posteriors."""
        model = SimpleVikhlininTemperature1D(
            x_unit="kpc",
            y_unit="keV",
            cust_start_pars=self.temp_model_ref.start_pars,
        )
        rng = np.random.default_rng(42)
        model.par_dists = [
            Quantity(rng.normal(p.value, max(abs(p.value) * 1e-4, 1e-6), self.num_samples), p.unit)
            for p in model.model_pars
        ]
        model.success = True
        model.fit_method = "mcmc"
        return model

    def _make_dens_model(self) -> KingProfile1D:
        """Create a fitted KingProfile1D instance with deterministic parameter posteriors."""
        model = KingProfile1D(
            x_unit="kpc",
            y_unit="1/cm^3",
            cust_start_pars=self.dens_model_ref.start_pars,
        )
        rng = np.random.default_rng(43)
        model.par_dists = [
            Quantity(rng.normal(p.value, max(abs(p.value) * 1e-4, 1e-6), self.num_samples), p.unit)
            for p in model.model_pars
        ]
        model.success = True
        model.fit_method = "mcmc"
        return model

    def _make_temp_profile(
        self,
        radii: Quantity,
        radii_err: Quantity,
        deg_radii: Quantity,
        projected: bool = False,
        with_model: bool = False,
    ) -> GasTemperature3D | ProjectedGasTemperature1D:
        """Construct a synthetic 3D or projected temperature profile evaluated from the reference model."""
        vals = self.temp_model_ref(radii).flatten()
        vals_err = vals * 1e-3
        prof_cls = ProjectedGasTemperature1D if projected else GasTemperature3D
        prof = prof_cls(
            radii,
            vals,
            self.centre,
            self.src_name,
            self.obs_id,
            self.inst,
            radii_err=radii_err,
            values_err=vals_err,
            deg_radii=deg_radii,
        )
        if with_model:
            prof.add_model_fit(self._make_temp_model(), "mcmc")
        return prof

    def _make_dens_profile(
        self,
        radii: Quantity,
        radii_err: Quantity,
        deg_radii: Quantity,
        with_model: bool = False,
    ) -> GasDensity3D:
        """Construct a synthetic 3D gas number density profile evaluated from the reference King model."""
        vals = self.dens_model_ref(radii).flatten()
        vals_err = vals * 1e-3
        apec_norm = APECNormalisation1D(
            radii,
            Quantity(np.full(len(radii), 1e-3), "cm^-5"),
            self.centre,
            self.src_name,
            self.obs_id,
            self.inst,
            radii_err=radii_err,
            values_err=Quantity(np.full(len(radii), 1e-5), "cm^-5"),
            deg_radii=deg_radii,
        )
        prof = GasDensity3D(
            radii,
            vals,
            self.centre,
            self.src_name,
            self.obs_id,
            self.inst,
            "onion",
            apec_norm,
            radii_err=radii_err,
            values_err=vals_err,
            deg_radii=deg_radii,
        )
        if with_model:
            prof.add_model_fit(self._make_dens_model(), "mcmc")
        return prof

    def test_base_profile_non_ascending_radii(self) -> None:
        """Verify BaseProfile1D rejects descending or duplicate radii and deg_radii on declaration."""
        vals = Quantity([4.0, 3.5, 3.0, 2.5], "keV")

        # Descending radii must raise ValueError
        desc_radii = Quantity([400.0, 300.0, 200.0, 100.0], "kpc")
        with self.assertRaises(ValueError):
            ProjectedGasTemperature1D(
                desc_radii,
                vals,
                self.centre,
                self.src_name,
                self.obs_id,
                self.inst,
                deg_radii=rad_to_ang(desc_radii, z=self.z),
            )

        # Duplicate radii must raise ValueError
        dup_radii = Quantity([100.0, 200.0, 200.0, 300.0], "kpc")
        with self.assertRaises(ValueError):
            GasTemperature3D(
                dup_radii,
                vals,
                self.centre,
                self.src_name,
                self.obs_id,
                self.inst,
                deg_radii=rad_to_ang(dup_radii, z=self.z),
            )

        # Ascending radii paired with descending deg_radii must also raise ValueError
        asc_radii = Quantity([100.0, 200.0, 300.0, 400.0], "kpc")
        with self.assertRaises(ValueError):
            ProjectedGasTemperature1D(
                asc_radii,
                vals,
                self.centre,
                self.src_name,
                self.obs_id,
                self.inst,
                deg_radii=rad_to_ang(desc_radii, z=self.z),
            )

    def test_data_driven_equal_radii(self) -> None:
        """Test HydrostaticMass, SpecificEntropy, and ThermalPressure with identical input radial grids."""
        test_out_path = os.path.join(MISC_OUTPUT_TESTS, self.id())
        os.makedirs(test_out_path, exist_ok=True)

        temp_3d = self._make_temp_profile(self.r_fine, self.r_fine_err, self.deg_fine, projected=False)
        temp_proj = self._make_temp_profile(self.r_fine, self.r_fine_err, self.deg_fine, projected=True)
        dens_3d = self._make_dens_profile(self.r_fine, self.r_fine_err, self.deg_fine)

        temp_3d.save_view(os.path.join(test_out_path, "input_temp_3d_fine.png"))
        dens_3d.save_view(os.path.join(test_out_path, "input_dens_3d_fine.png"))

        expected_entropy = (temp_3d.values * (dens_3d.values ** (-2.0 / 3.0))).to("keV cm^2")
        expected_pressure = (temp_3d.values * dens_3d.values).to("keV / cm^3")

        for interp_flag in (False, True):
            hm = HydrostaticMass(
                temp_3d,
                dens_3d,
                interp_data=interp_flag,
                num_samples=self.num_samples,
            )
            self.assertTrue(np.allclose(hm.radii, self.r_fine))
            self.assertTrue(np.all(hm.values > Quantity(0.0, "Msun")))
            hm.save_view(os.path.join(test_out_path, f"hm_equal_interp_{interp_flag}.png"))

            for t_label, t_prof in (("3d", temp_3d), ("proj", temp_proj)):
                se = SpecificEntropy(
                    t_prof,
                    dens_3d,
                    interp_data=interp_flag,
                    num_samples=self.num_samples,
                )
                self.assertTrue(np.allclose(se.radii, self.r_fine))
                self.assertTrue(np.allclose(se.values, expected_entropy, rtol=2e-2))
                se.save_view(os.path.join(test_out_path, f"se_equal_{t_label}_interp_{interp_flag}.png"))

                tp = ThermalPressure(
                    t_prof,
                    dens_3d,
                    interp_data=interp_flag,
                    num_samples=self.num_samples,
                )
                self.assertTrue(np.allclose(tp.radii, self.r_fine))
                self.assertTrue(np.allclose(tp.values, expected_pressure, rtol=2e-2))
                tp.save_view(os.path.join(test_out_path, f"tp_equal_{t_label}_interp_{interp_flag}.png"))

    def test_data_driven_unequal_radii_modes(self) -> None:
        """Test data-driven profiles when density is finer than temperature and vice versa, with and without interp."""
        test_out_path = os.path.join(MISC_OUTPUT_TESTS, self.id())
        os.makedirs(test_out_path, exist_ok=True)

        temp_coarse = self._make_temp_profile(self.r_coarse, self.r_coarse_err, self.deg_coarse)
        dens_fine = self._make_dens_profile(self.r_fine, self.r_fine_err, self.deg_fine)

        temp_fine = self._make_temp_profile(self.r_fine, self.r_fine_err, self.deg_fine)
        dens_coarse = self._make_dens_profile(self.r_coarse, self.r_coarse_err, self.deg_coarse)

        for grid_label, t_prof, d_prof in (
            ("temp_coarse_dens_fine", temp_coarse, dens_fine),
            ("temp_fine_dens_coarse", temp_fine, dens_coarse),
        ):
            for interp_flag in (False, True):
                hm = HydrostaticMass(t_prof, d_prof, interp_data=interp_flag, num_samples=self.num_samples)
                se = SpecificEntropy(t_prof, d_prof, interp_data=interp_flag, num_samples=self.num_samples)
                tp = ThermalPressure(t_prof, d_prof, interp_data=interp_flag, num_samples=self.num_samples)

                self.assertTrue(np.allclose(hm.radii, self.r_fine))
                self.assertTrue(np.allclose(se.radii, self.r_fine))
                self.assertTrue(np.allclose(tp.radii, self.r_fine))
                self.assertTrue(np.all(hm.values > Quantity(0.0, "Msun")))
                self.assertTrue(np.all(se.values > Quantity(0.0, "keV cm^2")))
                self.assertTrue(np.all(tp.values > Quantity(0.0, "keV / cm^3")))

                hm.save_view(os.path.join(test_out_path, f"hm_{grid_label}_interp_{interp_flag}.png"))
                se.save_view(os.path.join(test_out_path, f"se_{grid_label}_interp_{interp_flag}.png"))
                tp.save_view(os.path.join(test_out_path, f"tp_{grid_label}_interp_{interp_flag}.png"))

        # Check that bin-matching (interp_data=False) vs linear interpolation (interp_data=True) produce distinct
        #  values at fine radii that do not coincide with coarse bin centres
        se_matched = SpecificEntropy(temp_coarse, dens_fine, interp_data=False, num_samples=self.num_samples)
        se_interp = SpecificEntropy(temp_coarse, dens_fine, interp_data=True, num_samples=self.num_samples)
        # Fine bin 0 (100 kpc) and bin 1 (160 kpc) share coarse temperature bin 0 ([70, 190) kpc, centred at 130 kpc)
        expected_matched_bin0 = (temp_coarse.values[0] * (dens_fine.values[0] ** (-2.0 / 3.0))).to("keV cm^2")
        self.assertTrue(np.isclose(se_matched.values[0], expected_matched_bin0, rtol=2e-2))
        self.assertFalse(np.allclose(se_matched.values, se_interp.values, rtol=1e-3))

    def test_data_driven_auto_radii_truncation(self) -> None:
        """Verify automatic radius extraction warns and truncates to the coarser profile's annulus bounds."""
        test_out_path = os.path.join(MISC_OUTPUT_TESTS, self.id())
        os.makedirs(test_out_path, exist_ok=True)

        temp_coarse = self._make_temp_profile(self.r_coarse, self.r_coarse_err, self.deg_coarse)
        dens_ext = self._make_dens_profile(self.r_extended, self.r_extended_err, self.deg_extended)

        for interp_flag in (False, True):
            for prof_cls in (HydrostaticMass, SpecificEntropy, ThermalPressure):
                with self.assertWarns(UserWarning):
                    prof = prof_cls(
                        temp_coarse,
                        dens_ext,
                        interp_data=interp_flag,
                        num_samples=self.num_samples,
                    )
                # The 460 kpc point exceeds the coarse outer bound (430 kpc) and must be truncated
                self.assertEqual(len(prof.radii), len(self.r_fine))
                self.assertTrue(np.allclose(prof.radii, self.r_fine))
                prof.save_view(os.path.join(test_out_path, f"{prof_cls.__name__}_truncated_interp_{interp_flag}.png"))

    def test_model_and_mixed_modes(self) -> None:
        """Test both-models and mixed model/data modes across HydrostaticMass, SpecificEntropy, and ThermalPressure."""
        test_out_path = os.path.join(MISC_OUTPUT_TESTS, self.id())
        os.makedirs(test_out_path, exist_ok=True)

        temp_prof = self._make_temp_profile(self.r_fine, self.r_fine_err, self.deg_fine, with_model=True)
        dens_prof = self._make_dens_profile(self.r_fine, self.r_fine_err, self.deg_fine, with_model=True)

        # Using custom evaluation radii when models are involved
        r_custom = Quantity([120.0, 200.0, 280.0, 360.0], "kpc")
        r_custom_err = Quantity(np.full(len(r_custom), 20.0), "kpc")
        deg_custom = rad_to_ang(r_custom, z=self.z)

        model_combos = [
            ("simple_vikhlinin_temp", "king", False, r_custom, r_custom_err, deg_custom),
            ("simple_vikhlinin_temp", None, True, r_custom, r_custom_err, deg_custom),
            ("simple_vikhlinin_temp", None, False, self.r_fine, self.r_fine_err, self.deg_fine),
            (None, "king", True, r_custom, r_custom_err, deg_custom),
            (None, "king", False, self.r_fine, self.r_fine_err, self.deg_fine),
        ]

        for t_mod, d_mod, interp_flag, rads, rads_err, deg_rads in model_combos:
            for prof_cls in (HydrostaticMass, SpecificEntropy, ThermalPressure):
                prof = prof_cls(
                    temp_prof,
                    dens_prof,
                    temperature_model=t_mod,
                    density_model=d_mod,
                    radii=rads,
                    radii_err=rads_err,
                    deg_radii=deg_rads,
                    interp_data=interp_flag,
                    num_samples=self.num_samples,
                )
                self.assertTrue(np.allclose(prof.radii, rads))
                self.assertEqual(len(prof.values), len(rads))
                self.assertTrue(np.all(prof.values.value > 0))
                prof.save_view(
                    os.path.join(
                        test_out_path, f"{prof_cls.__name__}_tmod_{t_mod}_dmod_{d_mod}_interp_{interp_flag}.png"
                    )
                )

        # Omitting radii or deg_radii when a model is specified must raise ValueError
        for prof_cls in (HydrostaticMass, SpecificEntropy, ThermalPressure):
            with self.assertRaises(ValueError):
                prof_cls(
                    temp_prof,
                    dens_prof,
                    temperature_model="simple_vikhlinin_temp",
                    density_model="king",
                    num_samples=self.num_samples,
                )
            with self.assertRaises(ValueError):
                prof_cls(
                    temp_prof,
                    dens_prof,
                    temperature_model="simple_vikhlinin_temp",
                    density_model="king",
                    radii=r_custom,
                    radii_err=r_custom_err,
                    num_samples=self.num_samples,
                )

    def test_point_evaluation_requested_radii_and_rad_check(self) -> None:
        """Test mass(), entropy(), and pressure() at requested non-node radii, caching, and rad_check bounds."""
        temp_prof = self._make_temp_profile(self.r_fine, self.r_fine_err, self.deg_fine, with_model=True)
        dens_prof = self._make_dens_profile(self.r_fine, self.r_fine_err, self.deg_fine, with_model=True)

        # Data-driven interpolated profiles on r_fine
        hm_data = HydrostaticMass(temp_prof, dens_prof, interp_data=True, num_samples=self.num_samples)
        se_data = SpecificEntropy(temp_prof, dens_prof, interp_data=True, num_samples=self.num_samples)
        tp_data = ThermalPressure(temp_prof, dens_prof, interp_data=True, num_samples=self.num_samples)

        # Evaluate at two distinct scalar radii (130 kpc and 310 kpc) not in r_fine and verify they differ
        r_inner = Quantity(130.0, "kpc")
        r_outer = Quantity(310.0, "kpc")
        r_pair = Quantity([130.0, 310.0], "kpc")

        m_in, m_in_dist = hm_data.mass(r_inner)
        m_out, _ = hm_data.mass(r_outer)
        self.assertEqual(m_in.shape, (3,))
        self.assertFalse(np.isclose(m_in[0], m_out[0]))
        # Scalar call without radius_err should be cached in _masses
        m_in_cached, m_in_dist_cached = hm_data.mass(r_inner)
        self.assertTrue(np.array_equal(m_in_dist, m_in_dist_cached))
        self.assertTrue(np.allclose(m_in, m_in_cached))

        # Passing a 1-element 1D array radius should also evaluate cleanly without UnboundLocalError on stor_key
        m_in_arr1, _ = hm_data.mass(Quantity([130.0], "kpc"))
        self.assertEqual(m_in_arr1.shape, (3,))
        self.assertTrue(np.allclose(m_in_arr1[0], m_in[0], rtol=2e-2))

        # Evaluating mass with radius_err should propagate radius uncertainty without colliding with scalar cache
        m_in_werr, m_in_werr_dist = hm_data.mass(r_inner, radius_err=Quantity(15.0, "kpc"))
        self.assertEqual(m_in_werr.shape, (3,))
        self.assertFalse(np.array_equal(m_in_dist, m_in_werr_dist))

        # Verify SpecificEntropy.entropy() and ThermalPressure.pressure() evaluate at the requested radius (PR #1597)
        k_in, k_in_dist = se_data.entropy(r_inner)
        k_out, _ = se_data.entropy(r_outer)
        k_pair, _ = se_data.entropy(r_pair)
        self.assertEqual(k_in.shape, (3,))
        self.assertEqual(k_pair.shape, (3, 2))
        self.assertFalse(np.isclose(k_in[0], k_out[0]))
        self.assertTrue(np.allclose(k_pair[0], Quantity([k_in[0], k_out[0]]), rtol=2e-2))
        # Scalar entropy call should be cached in _entropies
        _, k_in_dist_cached = se_data.entropy(r_inner)
        self.assertTrue(np.array_equal(k_in_dist, k_in_dist_cached))

        p_in, p_in_dist = tp_data.pressure(r_inner)
        p_out, _ = tp_data.pressure(r_outer)
        p_pair, _ = tp_data.pressure(r_pair)
        self.assertEqual(p_in.shape, (3,))
        self.assertEqual(p_pair.shape, (3, 2))
        self.assertFalse(np.isclose(p_in[0], p_out[0]))
        self.assertTrue(np.allclose(p_pair[0], Quantity([p_in[0], p_out[0]]), rtol=2e-2))
        # Scalar pressure call should be cached in _pressures
        _, p_in_dist_cached = tp_data.pressure(r_inner)
        self.assertTrue(np.array_equal(p_in_dist, p_in_dist_cached))

        # Out-of-bounds radius (> 430 kpc) must raise ValueError for data-driven profiles
        r_oob = Quantity(500.0, "kpc")
        with self.assertRaises(ValueError):
            hm_data.mass(r_oob)
        with self.assertRaises(ValueError):
            se_data.entropy(r_oob)
        with self.assertRaises(ValueError):
            tp_data.pressure(r_oob)

        # Model-driven profiles should emit a UserWarning (not raise ValueError) when extrapolating beyond bounds
        for prof_cls, eval_method in (
            (HydrostaticMass, "mass"),
            (SpecificEntropy, "entropy"),
            (ThermalPressure, "pressure"),
        ):
            mod_prof = prof_cls(
                temp_prof,
                dens_prof,
                temperature_model="simple_vikhlinin_temp",
                density_model="king",
                radii=self.r_fine,
                radii_err=self.r_fine_err,
                deg_radii=self.deg_fine,
                num_samples=self.num_samples,
            )
            with self.assertWarns(UserWarning):
                res, _ = getattr(mod_prof, eval_method)(r_oob)
            self.assertTrue(res[0].value > 0)

            # In mixed mode (temperature_model set, density_model=None), exceeding the data-driven density profile's
            #  outer boundary must raise ValueError rather than warning
            mixed_prof = prof_cls(
                temp_prof,
                dens_prof,
                temperature_model="simple_vikhlinin_temp",
                density_model=None,
                interp_data=False,
                radii=self.r_fine,
                radii_err=self.r_fine_err,
                deg_radii=self.deg_fine,
                num_samples=self.num_samples,
            )
            with self.assertRaises(ValueError):
                getattr(mixed_prof, eval_method)(r_oob)
            if isinstance(mixed_prof, HydrostaticMass):
                # Single-radius evaluation on a non-interpolated mixed-mode HydrostaticMass must raise ValueError
                with self.assertRaises(ValueError):
                    mixed_prof.mass(r_inner)

    def test_data_driven_custom_radii(self) -> None:
        """Test custom (radii, radii_err, deg_radii) and incomplete-subset guards in pure data-driven mode."""
        test_out_path = os.path.join(MISC_OUTPUT_TESTS, self.id())
        os.makedirs(test_out_path, exist_ok=True)

        temp_prof = self._make_temp_profile(self.r_fine, self.r_fine_err, self.deg_fine)
        dens_prof = self._make_dens_profile(self.r_fine, self.r_fine_err, self.deg_fine)

        # Custom radial grid (4 bins inside [70, 430] kpc) distinct from the 6-bin input profiles
        r_custom = Quantity([115.0, 195.0, 275.0, 355.0], "kpc")
        r_custom_err = Quantity(np.full(len(r_custom), 25.0), "kpc")
        deg_custom = rad_to_ang(r_custom, z=self.z)

        # Passing only a subset of (radii, radii_err, deg_radii) in data-driven mode must raise ValueError
        incomplete_subsets = [
            {"radii": r_custom, "radii_err": None, "deg_radii": None},
            {"radii": r_custom, "radii_err": r_custom_err, "deg_radii": None},
            {"radii": r_custom, "radii_err": None, "deg_radii": deg_custom},
            {"radii": None, "radii_err": r_custom_err, "deg_radii": deg_custom},
        ]
        for prof_cls in (HydrostaticMass, SpecificEntropy, ThermalPressure):
            for kwargs in incomplete_subsets:
                with self.assertRaises(ValueError):
                    prof_cls(
                        temp_prof,
                        dens_prof,
                        radii=kwargs["radii"],
                        radii_err=kwargs["radii_err"],
                        deg_radii=kwargs["deg_radii"],
                        num_samples=self.num_samples,
                    )

        # Passing all three custom radius arguments in data-driven mode (both interpolated and bin-matched) must
        #  preserve the caller-supplied grid rather than overwriting it with the input profile radii
        for interp_flag in (False, True):
            for prof_cls in (HydrostaticMass, SpecificEntropy, ThermalPressure):
                prof = prof_cls(
                    temp_prof,
                    dens_prof,
                    radii=r_custom,
                    radii_err=r_custom_err,
                    deg_radii=deg_custom,
                    interp_data=interp_flag,
                    num_samples=self.num_samples,
                )
                self.assertEqual(len(prof.radii), len(r_custom))
                self.assertTrue(np.allclose(prof.radii, r_custom))
                self.assertTrue(np.allclose(prof.radii_err, r_custom_err))
                self.assertTrue(np.allclose(prof.deg_radii, deg_custom))
                self.assertTrue(np.all(prof.values.value > 0))
                prof.save_view(
                    os.path.join(test_out_path, f"{prof_cls.__name__}_data_custom_radii_interp_{interp_flag}.png")
                )

    def test_mass_radius_err_and_non_interp_edge_cases(self) -> None:
        """Test HydrostaticMass.mass() radius_err branches and non-interpolated (interp_data=False) boundary checks."""
        temp_prof = self._make_temp_profile(self.r_fine, self.r_fine_err, self.deg_fine, with_model=True)
        dens_prof = self._make_dens_profile(self.r_fine, self.r_fine_err, self.deg_fine, with_model=True)

        hm_interp = HydrostaticMass(temp_prof, dens_prof, interp_data=True, num_samples=self.num_samples)
        hm_mod = HydrostaticMass(
            temp_prof,
            dens_prof,
            temperature_model="simple_vikhlinin_temp",
            density_model="king",
            radii=self.r_fine,
            radii_err=self.r_fine_err,
            deg_radii=self.deg_fine,
            num_samples=self.num_samples,
        )

        # 1. Scalar radius + scalar radius_err cache hit: a repeated call with identical (radius, radius_err) must
        #    retrieve the exact cached posterior distribution from self._masses[f"{radius.value}_{radius_err}"]
        r_scal = Quantity(180.0, "kpc")
        r_scal_err = Quantity(12.0, "kpc")
        m_werr_1, dist_werr_1 = hm_interp.mass(r_scal, radius_err=r_scal_err)
        m_werr_2, dist_werr_2 = hm_interp.mass(r_scal, radius_err=r_scal_err)
        self.assertIn(f"{r_scal.value}_{r_scal_err}", hm_interp._masses)
        self.assertTrue(np.array_equal(dist_werr_1, dist_werr_2))
        self.assertTrue(np.allclose(m_werr_1, m_werr_2))

        # 2. Array radius + array radius_err propagation in both interpolated data-driven mode (where sample_data_radii
        #    draws a 2D (M, N) calc_rad array and skips direct node sampling) and model-driven mode
        r_arr = Quantity([130.0, 250.0, 370.0], "kpc")
        r_arr_err = Quantity([10.0, 15.0, 20.0], "kpc")
        for hm_inst in (hm_interp, hm_mod):
            m_arr_noerr, _ = hm_inst.mass(r_arr)
            m_arr_werr, dist_arr_werr = hm_inst.mass(r_arr, radius_err=r_arr_err)
            self.assertEqual(m_arr_werr.shape, (3, len(r_arr)))
            self.assertEqual(dist_arr_werr.shape, (self.num_samples, len(r_arr)))
            # Medians should remain consistent within ~5% while uncertainties increase when radius_err is included
            self.assertTrue(np.allclose(m_arr_werr[0], m_arr_noerr[0], rtol=5e-2))

        # Also verify array radius matching self.radii with array radius_err on hm_interp takes the interpolated
        #  sample_data_radii path rather than the direct-sampling shortcut
        m_nodes_werr, dist_nodes_werr = hm_interp.mass(self.r_fine, radius_err=self.r_fine_err)
        self.assertEqual(m_nodes_werr.shape, (3, len(self.r_fine)))
        self.assertEqual(dist_nodes_werr.shape, (self.num_samples, len(self.r_fine)))

        # 3. Validation guards on radius_err shape and units in HydrostaticMass.mass()
        # Mismatched shapes (scalar radius with array radius_err, or different-length 1D arrays) must raise ValueError
        with self.assertRaises(ValueError):
            hm_interp.mass(r_scal, radius_err=r_arr_err)
        with self.assertRaises(ValueError):
            hm_interp.mass(r_arr, radius_err=Quantity([10.0, 15.0], "kpc"))
        # Incompatible radius_err units (e.g. keV instead of length) must raise UnitConversionError
        with self.assertRaises(UnitConversionError):
            hm_interp.mass(r_scal, radius_err=Quantity(5.0, "keV"))

        # 4. Non-interpolated (interp_data=False) point evaluation & boundary checks
        hm_nointerp = HydrostaticMass(temp_prof, dens_prof, interp_data=False, num_samples=self.num_samples)
        se_nointerp = SpecificEntropy(temp_prof, dens_prof, interp_data=False, num_samples=self.num_samples)
        tp_nointerp = ThermalPressure(temp_prof, dens_prof, interp_data=False, num_samples=self.num_samples)

        # Scalar evaluation at an off-node radius (115 kpc falls inside bin 0 [70, 130) kpc) and at the exact outer
        #  annulus boundary (430 kpc, matched via match[..., -1] |= radius == bnds[-1, 1])
        r_bin0 = Quantity(115.0, "kpc")
        r_outer_edge = self.r_fine[-1] + self.r_fine_err[-1]  # 430.0 kpc

        k_bin0, _ = se_nointerp.entropy(r_bin0)
        k_edge, _ = se_nointerp.entropy(r_outer_edge)
        self.assertTrue(np.isclose(k_bin0[0], se_nointerp.values[0], rtol=2e-2))
        self.assertTrue(np.isclose(k_edge[0], se_nointerp.values[-1], rtol=2e-2))

        p_bin0, _ = tp_nointerp.pressure(r_bin0)
        p_edge, _ = tp_nointerp.pressure(r_outer_edge)
        self.assertTrue(np.isclose(p_bin0[0], tp_nointerp.values[0], rtol=2e-2))
        self.assertTrue(np.isclose(p_edge[0], tp_nointerp.values[-1], rtol=2e-2))

        # Multi-element array evaluation on hm_nointerp including the exact outer boundary (430 kpc)
        r_multi_edge = Quantity([115.0, 250.0, 430.0], "kpc")
        m_multi_edge, _ = hm_nointerp.mass(r_multi_edge)
        self.assertEqual(m_multi_edge.shape, (3, 3))
        self.assertTrue(np.all(m_multi_edge[0] > Quantity(0.0, "Msun")))

        # Radius smaller than the innermost annulus bound (10 kpc < 70 kpc) passes rad_check (which only checks the
        #  outer bound) but matches zero annular bins (len(inds) != radius.size) and must raise ValueError
        r_below_inner = Quantity(10.0, "kpc")
        with self.assertRaises(ValueError):
            hm_nointerp.mass(Quantity([10.0, 115.0], "kpc"))
        with self.assertRaises(ValueError):
            se_nointerp.entropy(r_below_inner)
        with self.assertRaises(ValueError):
            tp_nointerp.pressure(r_below_inner)

    def test_unit_conversions_and_reverse_mixed_rad_check(self) -> None:
        """Test mass-density (solMass/Mpc^3) and Kelvin (K) conversion branches and reverse mixed-mode rad_check."""
        test_out_path = os.path.join(MISC_OUTPUT_TESTS, self.id())
        os.makedirs(test_out_path, exist_ok=True)

        # Reference profiles in standard units (keV and 1/cm^3)
        temp_kev = self._make_temp_profile(self.r_fine, self.r_fine_err, self.deg_fine)
        dens_num = self._make_dens_profile(self.r_fine, self.r_fine_err, self.deg_fine)

        # Construct a GasDensity3D profile in mass-density units (solMass / Mpc^3) representing the exact same gas
        #  density via rho = n * MEAN_MOL_WEIGHT * m_p
        mass_dens_vals = (dens_num.values * MEAN_MOL_WEIGHT * m_p).to("solMass / Mpc^3")
        mass_dens_errs = (dens_num.values_err * MEAN_MOL_WEIGHT * m_p).to("solMass / Mpc^3")
        dens_mass = GasDensity3D(
            self.r_fine,
            mass_dens_vals,
            self.centre,
            self.src_name,
            self.obs_id,
            self.inst,
            "onion",
            dens_num.generation_profile,
            radii_err=self.r_fine_err,
            values_err=mass_dens_errs,
            deg_radii=self.deg_fine,
        )

        # Construct a temperature profile copy whose stored values are in Kelvin (K) via T_K = T_keV / k_B so that
        #  the not temp.unit.is_equivalent("keV") conversion branch in entropy() and pressure() (and the already-in-K
        #  branch in mass()) is exercised
        temp_kelvin = self._make_temp_profile(self.r_fine, self.r_fine_err, self.deg_fine)
        temp_kelvin._values = (temp_kev.values / k_B).to("K")
        temp_kelvin._values_err = (temp_kev.values_err / k_B).to("K")

        hm_ref = HydrostaticMass(temp_kev, dens_num, interp_data=True, num_samples=self.num_samples)
        se_ref = SpecificEntropy(temp_kev, dens_num, interp_data=True, num_samples=self.num_samples)
        tp_ref = ThermalPressure(temp_kev, dens_num, interp_data=True, num_samples=self.num_samples)

        hm_conv = HydrostaticMass(temp_kelvin, dens_mass, interp_data=True, num_samples=self.num_samples)
        se_conv = SpecificEntropy(temp_kelvin, dens_mass, interp_data=True, num_samples=self.num_samples)
        tp_conv = ThermalPressure(temp_kelvin, dens_mass, interp_data=True, num_samples=self.num_samples)

        self.assertTrue(np.allclose(hm_conv.values, hm_ref.values, rtol=2e-2))
        self.assertTrue(np.allclose(se_conv.values, se_ref.values, rtol=2e-2))
        self.assertTrue(np.allclose(tp_conv.values, tp_ref.values, rtol=2e-2))

        hm_conv.save_view(os.path.join(test_out_path, "hm_kelvin_mass_dens.png"))
        se_conv.save_view(os.path.join(test_out_path, "se_kelvin_mass_dens.png"))
        tp_conv.save_view(os.path.join(test_out_path, "tp_kelvin_mass_dens.png"))

        # Reverse mixed-mode rad_check: temperature_model=None (data-driven) and density_model="king" (model-driven).
        #  Use a fine temperature profile covering [70, 490] kpc (r_extended) and a coarse density profile covering
        #  [70, 430] kpc (r_coarse) with a fitted King model.
        temp_ext = self._make_temp_profile(self.r_extended, self.r_extended_err, self.deg_extended, with_model=False)
        dens_coarse_mod = self._make_dens_profile(self.r_coarse, self.r_coarse_err, self.deg_coarse, with_model=True)

        for prof_cls, eval_method in (
            (HydrostaticMass, "mass"),
            (SpecificEntropy, "entropy"),
            (ThermalPressure, "pressure"),
        ):
            rev_mixed = prof_cls(
                temp_ext,
                dens_coarse_mod,
                temperature_model=None,
                density_model="king",
                interp_data=True,
                radii=self.r_coarse,
                radii_err=self.r_coarse_err,
                deg_radii=self.deg_coarse,
                num_samples=self.num_samples,
            )
            # 1. Evaluating at 460 kpc (exceeds dens_coarse_mod outer bound 430 kpc, which has a model, but is inside
            #    temp_ext outer bound 490 kpc) must emit a UserWarning and succeed
            with self.assertWarns(UserWarning):
                res_warn, _ = getattr(rev_mixed, eval_method)(Quantity(460.0, "kpc"))
            self.assertTrue(res_warn[0].value > 0)

            # 2. Evaluating at 520 kpc (exceeds temp_ext outer bound 490 kpc, which has no model) must raise ValueError
            with self.assertRaises(ValueError):
                getattr(rev_mixed, eval_method)(Quantity(520.0, "kpc"))

            # 3. Passing incompatible radius units (e.g. keV) to rad_check must raise UnitConversionError
            with self.assertRaises(UnitConversionError):
                rev_mixed.rad_check(Quantity(100.0, "keV"))

    def test_hydrostatic_mass_derived_methods_and_dist_views(self) -> None:
        """Test annular_mass, baryon_fraction, baryon_fraction_profile, overdensity_radius, and view_*_dist methods."""
        test_out_path = os.path.join(MISC_OUTPUT_TESTS, self.id())
        os.makedirs(test_out_path, exist_ok=True)

        temp_prof = self._make_temp_profile(self.r_fine, self.r_fine_err, self.deg_fine, with_model=True)
        dens_prof = self._make_dens_profile(self.r_fine, self.r_fine_err, self.deg_fine, with_model=True)
        # Set _tele and _save_path on dens_prof inside test_out_path because GasDensity3D.gas_mass() calls self.save()
        dens_prof._tele = "xmm"
        dens_prof._save_path = os.path.join(test_out_path, "dens_prof_gas_mass.xga")

        hm_mod = HydrostaticMass(
            temp_prof,
            dens_prof,
            temperature_model="simple_vikhlinin_temp",
            density_model="king",
            radii=self.r_fine,
            radii_err=self.r_fine_err,
            deg_radii=self.deg_fine,
            num_samples=self.num_samples,
        )
        hm_data = HydrostaticMass(temp_prof, dens_prof, interp_data=True, num_samples=self.num_samples)
        se_mod = SpecificEntropy(
            temp_prof,
            dens_prof,
            temperature_model="simple_vikhlinin_temp",
            density_model="king",
            radii=self.r_fine,
            radii_err=self.r_fine_err,
            deg_radii=self.deg_fine,
            num_samples=self.num_samples,
        )
        tp_mod = ThermalPressure(
            temp_prof,
            dens_prof,
            temperature_model="simple_vikhlinin_temp",
            density_model="king",
            radii=self.r_fine,
            radii_err=self.r_fine_err,
            deg_radii=self.deg_fine,
            num_samples=self.num_samples,
        )

        # ------------------- 1. HydrostaticMass.annular_mass -------------------
        # Scalar inner and outer radii: annular_mass returns outer_mass_dist - inner_mass_dist
        r_in_s = Quantity(120.0, "kpc")
        r_out_s = Quantity(320.0, "kpc")
        ann_dist_s = hm_mod.annular_mass(r_out_s, r_in_s)
        _, m_out_dist_s = hm_mod.mass(r_out_s)
        _, m_in_dist_s = hm_mod.mass(r_in_s)
        self.assertEqual(ann_dist_s.shape, (self.num_samples,))
        self.assertTrue(np.allclose(ann_dist_s, m_out_dist_s - m_in_dist_s))

        # 1D array inner and outer radii
        r_in_arr = Quantity([100.0, 200.0], "kpc")
        r_out_arr = Quantity([200.0, 350.0], "kpc")
        ann_dist_arr = hm_mod.annular_mass(r_out_arr, r_in_arr)
        self.assertEqual(ann_dist_arr.shape, (self.num_samples, 2))
        self.assertTrue(np.all(np.median(ann_dist_arr, axis=0) > Quantity(0.0, "Msun")))

        # Validation guards for annular_mass: scalar/non-scalar mismatch, 2D array, and mismatched 1D shapes
        with self.assertRaises(ValueError):
            hm_mod.annular_mass(r_out_s, r_in_arr)
        with self.assertRaises(ValueError):
            hm_mod.annular_mass(Quantity([[100.0, 200.0]], "kpc"), Quantity([[200.0, 300.0]], "kpc"))
        with self.assertRaises(ValueError):
            hm_mod.annular_mass(r_out_arr, Quantity([100.0, 150.0, 200.0], "kpc"))

        # ------------------- 2. HydrostaticMass.baryon_fraction & baryon_fraction_profile -------------------
        bf_res, bf_dist = hm_mod.baryon_fraction(r_out_s)
        self.assertEqual(bf_res.shape, (3,))
        self.assertEqual(bf_dist.shape, (self.num_samples,))
        self.assertTrue(0.0 < bf_res[0].value < 1.0)

        # Distribution-length truncation branches in baryon_fraction when len(hy_mass_dist) != len(gas_mass_dist):
        #  (a) len(hy_mass_dist) < len(gas_mass_dist): cache a 150-sample hydrostatic mass distribution in
        #      hm_mod._masses while gas_mass_dist has self.num_samples (200) entries
        r_trunc_lt = Quantity(275.0, "kpc")
        hm_mod._masses[str(r_trunc_lt)] = Quantity(np.full(self.num_samples - 50, 1e14), "Msun")
        _, bf_dist_shorter = hm_mod.baryon_fraction(r_trunc_lt)
        self.assertEqual(len(bf_dist_shorter), self.num_samples - 50)

        #  (b) len(hy_mass_dist) > len(gas_mass_dist): cache a 250-sample hydrostatic mass distribution in
        #      hm_mod._masses while gas_mass_dist has self.num_samples (200) entries
        r_trunc_gt = Quantity(290.0, "kpc")
        hm_mod._masses[str(r_trunc_gt)] = Quantity(np.full(self.num_samples + 50, 1e14), "Msun")
        _, bf_dist_longer = hm_mod.baryon_fraction(r_trunc_gt)
        self.assertEqual(len(bf_dist_longer), self.num_samples)

        # baryon_fraction_profile at default radii and at custom (radii, deg_radii)
        bf_prof_default = hm_mod.baryon_fraction_profile()
        self.assertIsInstance(bf_prof_default, BaryonFraction)
        self.assertTrue(np.allclose(bf_prof_default.radii, self.r_fine))
        bf_prof_default.save_view(os.path.join(test_out_path, "baryon_fraction_profile_default.png"))

        r_bf_cust = Quantity([140.0, 240.0, 340.0], "kpc")
        deg_bf_cust = rad_to_ang(r_bf_cust, z=self.z)
        bf_prof_cust = hm_mod.baryon_fraction_profile(radii=r_bf_cust, deg_radii=deg_bf_cust)
        self.assertTrue(np.allclose(bf_prof_cust.radii, r_bf_cust))
        bf_prof_cust.save_view(os.path.join(test_out_path, "baryon_fraction_profile_custom.png"))

        # Validation guards for baryon_fraction and baryon_fraction_profile
        # Non-scalar radius in baryon_fraction must raise ValueError
        with self.assertRaises(ValueError):
            hm_mod.baryon_fraction(r_out_arr)
        # Custom radii without deg_radii in baryon_fraction_profile must raise ValueError
        with self.assertRaises(ValueError):
            hm_mod.baryon_fraction_profile(radii=r_bf_cust)
        # Data-driven HydrostaticMass (density_model=None) calls GasDensity3D.gas_mass(model=None), raising
        #  NotImplementedError
        with self.assertRaises(NotImplementedError):
            hm_data.baryon_fraction(r_out_s)
        # Unfitted density model (fit_method=None) must raise XGAFitError
        self.assertIsNotNone(hm_mod._dens_model)
        if hm_mod._dens_model is not None:
            orig_fit_method = hm_mod._dens_model._fit_method
            try:
                hm_mod._dens_model._fit_method = None
                with self.assertRaises(XGAFitError):
                    hm_mod.baryon_fraction(r_out_s)
            finally:
                hm_mod._dens_model._fit_method = orig_fit_method

        # ------------------- 3. HydrostaticMass.overdensity_radius -------------------
        cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)
        # Determine the exact integer overdensity factor Delta corresponding to r = 250 kpc so the target radius is
        #  guaranteed to lie cleanly inside the [100, 420] kpc data range without extrapolating
        r_ref_od = Quantity(250.0, "kpc")
        m_ref_od, _ = hm_mod.mass(r_ref_od)
        rho_ref_od = m_ref_od[0] / (4.0 * np.pi * (r_ref_od**3) / 3.0)
        target_delta = int(round((rho_ref_od / cosmo.critical_density(self.z)).to("").value))

        # Two-stage search (init_step=100 kpc -> buffered 1 kpc refinement) within [100, 420] kpc
        r_od_two_stage = hm_mod.overdensity_radius(
            delta=target_delta,
            redshift=self.z,
            cosmo=cosmo,
            init_lo_rad=Quantity(100.0, "kpc"),
            init_hi_rad=Quantity(420.0, "kpc"),
            init_step=Quantity(100.0, "kpc"),
            out_unit=Unit("kpc"),
        )
        # Single-stage search (init_step=1 kpc) and string out_unit="Mpc"
        r_od_one_stage = hm_mod.overdensity_radius(
            delta=target_delta,
            redshift=self.z,
            cosmo=cosmo,
            init_lo_rad=Quantity(200.0, "kpc"),
            init_hi_rad=Quantity(300.0, "kpc"),
            init_step=Quantity(1.0, "kpc"),
            out_unit="Mpc",
        )
        self.assertTrue(r_od_two_stage.unit == Unit("kpc"))
        self.assertTrue(r_od_one_stage.unit == Unit("Mpc"))
        self.assertTrue(np.isclose(r_od_two_stage, r_ref_od, atol=Quantity(2.0, "kpc")))
        self.assertTrue(np.isclose(r_od_two_stage, r_od_one_stage.to("kpc"), atol=Quantity(2.0, "kpc")))

        # Validation guards for overdensity_radius
        with self.assertRaises(ValueError):
            hm_mod.overdensity_radius(0, self.z, cosmo)
        with self.assertRaises(ValueError):
            hm_mod.overdensity_radius(500.0, self.z, cosmo)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            hm_mod.overdensity_radius(2500, self.z, cosmo, out_unit=123)  # type: ignore[arg-type]
        with self.assertRaises(UnitConversionError):
            hm_mod.overdensity_radius(2500, self.z, cosmo, out_unit="deg")
        with self.assertRaises(ValueError):
            hm_mod.overdensity_radius(2500, 0.0, cosmo)
        # Brackets that do not enclose the target overdensity radius must raise ValueError
        with self.assertRaises(ValueError):
            hm_mod.overdensity_radius(
                2500,
                self.z,
                cosmo,
                init_lo_rad=Quantity(100.0, "kpc"),
                init_hi_rad=Quantity(140.0, "kpc"),
                init_step=Quantity(10.0, "kpc"),
            )

        # ------------------- 4. Distribution viewers (view_*_dist) -------------------
        with patch("matplotlib.pyplot.show") as mock_show:
            hm_mod.view_mass_dist(r_out_s)
            plt.close("all")
            hm_mod.view_baryon_fraction_dist(r_out_s)
            plt.close("all")
            se_mod.view_entropy_dist(r_out_s)
            plt.close("all")
            tp_mod.view_pressure_dist(r_out_s)
            plt.close("all")
            self.assertEqual(mock_show.call_count, 4)

        # Non-scalar radius passed to any view_*_dist method must raise ValueError
        with self.assertRaises(ValueError):
            hm_mod.view_mass_dist(r_out_arr)
        with self.assertRaises(ValueError):
            hm_mod.view_baryon_fraction_dist(r_out_arr)
        with self.assertRaises(ValueError):
            se_mod.view_entropy_dist(r_out_arr)
        with self.assertRaises(ValueError):
            tp_mod.view_pressure_dist(r_out_arr)
