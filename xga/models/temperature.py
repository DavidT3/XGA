#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 23/01/2021, 17:04. Copyright (c) David J Turner

def vikhlinin_3d_temp():
    pass


# So that things like fitting functions can be written generally to support different models
TEMP_MODELS = {}
TEMP_MODELS_STARTS = {}
TEMP_MODELS_PRIORS = {}


# SB_MODELS = {"beta_profile": beta_profile, "double_beta_profile": double_beta_profile,
#              "simple_vikhlinin": simple_vikhlinin}
#
# # For curve_fit type fitters where a initial value is important
# SB_MODELS_STARTS = {"beta_profile": [1, 50, 1], "double_beta_profile": [1, 400, 1, 100, 0.5, 0.5],
#                     "simple_vikhlinin": [1, 100, 1, 300, 3, 0.1, 1]}
#
# SB_MODELS_PRIORS = {"beta_profile": [[0, 3], [0, 300], [0, 10]],
#                     "double_beta_profile": [[0, 1000], [0, 2000], [0, 1000], [0, 2000], [-100, 100], [0, 100]],
#                     "simple_vikhlinin": [[0, 1000], [0, 2000], [-100, 100], [0, 2000], [-100, 100],
#                                          [-100, 100], [0, 100]]}