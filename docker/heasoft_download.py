#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 19/11/2022, 20:00. Copyright (c) The Contributors

import shutil

import requests

with requests.get('https://heasarc.gsfc.nasa.gov/cgi-bin/Tools/tarit/tarit.pl?mode=download&arch=src&src_pc_linux_centos=Y&src_other_specify=&general=attitude&general=heasarc&general=heasptools&general=heatools&general=heagen&xanadu=xspec', stream=True) as streamo:
    with open('testo.tar.gz', 'wb') as writo:
        shutil.copyfileobj(streamo.raw, writo)

