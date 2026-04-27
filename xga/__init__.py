#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 4/25/26, 4:03 PM. Copyright (c) The Contributors.
from . import _version
__version__ = _version.get_versions()['version']

def __getattr__(name):
    """
    A module level __getattr__ which allows us to lazily load the XGA configuration and census from the utils
    module. This helps to avoid race conditions when importing XGA from other modules (like DAXA).

    :param str name: The name of the attribute to be returned.
    :return: The value of the attribute.
    """
    import importlib
    # We import the utils module to check which variables are marked as lazy
    utils = importlib.import_module('.utils', __package__)

    # We check if the requested name is in the set of lazy variables and NOT in the exclusion set
    if name in utils._LAZY_VARS and name not in utils._KEEP_PRIVATE_LAZY_VARS:
        return getattr(utils, name)
    elif name == 'sas':
        generate = importlib.import_module('.generate', __package__)
        return generate.sas
    elif name == 'generate':
        return importlib.import_module('.generate', __package__)
    elif name == 'utils':
        return utils
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Here we set up 'shims' to ensure that pre-multi-mission imports of SAS wrapper functions
#  still function as intended.
import sys
from types import ModuleType

# We use a custom dictionary-like object for sys.modules to lazily import the sas modules
class LazyModuleShim(ModuleType):
    def __init__(self, target_name):
        super().__init__(target_name)
        self.target_name = target_name
        self._module = None
        # We set this to None to avoid triggering a load when someone checks for __file__
        self.__file__ = None
        self.__path__ = []

    @property
    def _real_module(self):
        if self._module is None:
            # We import xga.generate here, which will then allow us to access the submodules
            import importlib
            generate = importlib.import_module('xga.generate')
            if self.target_name == 'xga.sas':
                self._module = generate.sas
            elif self.target_name == 'xga.sas.phot':
                self._module = generate.sas.phot
            elif self.target_name == 'xga.sas.misc':
                self._module = generate.sas.misc
            elif self.target_name == 'xga.sas.spec':
                self._module = generate.sas.spec
            elif self.target_name == 'xga.sas.lightcurve':
                self._module = generate.sas.lightcurve
            elif self.target_name == 'xga.sas.run':
                self._module = generate.sas.run
        return self._module

    def __getattr__(self, name):
        return getattr(self._real_module, name)

    def __dir__(self):
        return dir(self._real_module)

# We populate sys.modules with our shims
sys.modules['xga.sas'] = LazyModuleShim('xga.sas')
sys.modules['xga.sas.phot'] = LazyModuleShim('xga.sas.phot')
sys.modules['xga.sas.misc'] = LazyModuleShim('xga.sas.misc')
sys.modules['xga.sas.spec'] = LazyModuleShim('xga.sas.spec')
sys.modules['xga.sas.lightcurve'] = LazyModuleShim('xga.sas.lightcurve')
sys.modules['xga.sas.run'] = LazyModuleShim('xga.sas.run')
