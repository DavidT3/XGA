#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 05/03/2021, 09:17. Copyright (c) David J Turner


class BaseModel1D:
    """
    The superclass of XGA's 1D models, with base functionality implemented, including the numerical methods for
    calculating derivatives and abel transforms which can be overwritten by subclasses if analytical solutions
    are available. The BaseModel class shouldn't be instantiated by itself, as it won't do anything.
    """
    def __init__(self, x_unit, y_unit):
        pass

    def __call__(self, x):
        return self.model_func(x, *self._pars)

    def model_func(self, x, pars):
        pass

    def derivative(self, x):
        pass

    def inverse_abel(self, x):
        pass

    def integral(self):
        pass

    @property
    def model_pars(self):
        pass

    @model_pars.setter
    def model_pars(self, new_vals):
        pass

    @property
    def start_pars(self):
        pass

    @start_pars.setter
    def start_pars(self, new_vals):
        pass

    @property
    def par_priors(self):
        pass

    @par_priors.setter
    def par_priors(self, new_vals):
        pass