#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 4/27/26, 12:33 PM. Copyright (c) The Contributors.

import os
import sys

if __name__ == "__main__":
    cwd = os.getcwd()

    # DAXA does not yet support environment variable configuration overrides, so we have to
    #  manually move the user's config file out of the way and replace it with the test one
    daxa_conf_dir = os.environ.get('XDG_CONFIG_HOME', os.path.join(os.path.expanduser('~'), '.config', 'daxa'))
    if not os.path.exists(daxa_conf_dir):
        os.makedirs(daxa_conf_dir)

    daxa_conf_path = os.path.join(daxa_conf_dir, 'daxa.cfg')
    daxa_backup_path = os.path.join(daxa_conf_dir, 'backup_daxa.cfg')
    # If there is already a config file we move it to a backup
    if os.path.exists(daxa_conf_path):
        import shutil
        shutil.move(daxa_conf_path, daxa_backup_path)

    # Now we copy the template daxa config into the correct place
    import shutil
    shutil.copy('test_data/test_template_daxa.cfg', daxa_conf_path)

    # We set the XGA_CONFIG_DIR environment variable so that XGA knows where to look for/write its config
    os.environ['XGA_CONFIG_DIR'] = os.path.join(cwd, 'test_data/config/')

    try:
        # Path logic to ensure we can import tests
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

        from astropy.units import Quantity
        from daxa.archive import Archive
        from daxa.mission import XMMPointed, eRASS1DE, Chandra
        from daxa.process.simple import full_process_xmm, full_process_erosita, full_process_chandra
        from daxa.process._backend_check import find_sas, find_esass, find_ciao
        from daxa.exceptions import SASNotFoundError, eSASSNotFoundError, CIAONotFoundError
        from tests.source_info import SRC_INFO, SUPP_SRC_INFO
        from astropy.coordinates import SkyCoord

        # Create SkyCoord instances for the test sources
        test_coords = SkyCoord(ra=[SRC_INFO['ra'], SUPP_SRC_INFO['ra']],
                               dec=[SRC_INFO['dec'], SUPP_SRC_INFO['dec']],
                               unit='deg')

        # Check backends
        try:
            find_sas()
            sas_avail = True
        except SASNotFoundError:
            sas_avail = False

        try:
            find_esass()
            esass_avail = True
        except eSASSNotFoundError:
            esass_avail = False

        try:
            find_ciao()
            ciao_avail = True
        except CIAONotFoundError:
            ciao_avail = False

        missions = []
        if sas_avail:
            xm = XMMPointed()
            xm.filter_on_positions(test_coords)
            missions.append(xm)

        if esass_avail:
            er = eRASS1DE()
            er.filter_on_positions(test_coords, search_distance=Quantity(3.6, 'deg'))
            missions.append(er)

        if ciao_avail:
            ch = Chandra(insts="ACIS")
            ch.filter_on_positions(test_coords)
            missions.append(ch)

        if len(missions) > 0:
            arch = Archive('xga_tests', missions)
            if esass_avail:
                full_process_erosita(arch)
            if sas_avail:
                full_process_xmm(arch)
            if ciao_avail:
                full_process_chandra(arch)

            # Now that DAXA has processed the data, we want to ensure that XGA has a fresh,
            #  accurate census for the tests to use. We use the reinitialise_xga function
            #  to make sure the correct config directory is picked up, then trigger a full
            #  rebuild of the census to ensure it matches the data just downloaded.
            import xga
            xga.reinitialise_xga(os.environ['XGA_CONFIG_DIR'])
            xga.rebuild_census(full_rebuild=True)
        else:
            raise ValueError("No mission backends (SAS, eSASS, or CIAO) are available.")

    finally:
        # We restore the user's daxa config file
        import shutil
        if os.path.exists(daxa_backup_path):
            shutil.move(daxa_backup_path, daxa_conf_path)
        else:
            # If there was no backup, we just remove the test config we put there
            os.remove(daxa_conf_path)
