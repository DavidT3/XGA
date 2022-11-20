#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 19/11/2022, 21:27. Copyright (c) The Contributors

import shutil

import requests

# This is a very cheesy script which uses the HEASoft source-code-assembly script to download a minimal
#  version of HEASoft with XSPEC + the base high energy tools. I could download the whole package using
#  wget and a normal URL in the dockerfile, but I'm really trying to minimise the size of the final image so
#  this is the solution I ended up with.
heasoft_url = 'https://heasarc.gsfc.nasa.gov/cgi-bin/Tools/tarit/tarit.pl?mode=download&arch=src&src_pc_linux_' \
              'ubuntu=Y&src_other_specify=&general=attitude&general=heasarc&general=heasptools&general=heatools&' \
              'general=heagen&xanadu=xspec'

#  Opens a get connection to that URL, and the data are streamed (so it starts to write the file before
#  everything has been downloaded into memory)
with requests.get(heasoft_url, stream=True) as streamo:
    # Gets the content disposition header, so we can try and ascertain the version of HEASoft we're downloading
    #  Unfortunately unlike the full HEASoft version we can't get a specific version from an FTP server, this
    #  script just seems to serve the most up-to-date version.
    cont_dis = streamo.headers['Content-Disposition']

    # This is me trying to make sure that the name follows the pattern I expect it to
    if 'filename="heasoft-' in cont_dis and 'src.tar.gz"' in cont_dis:
        version = cont_dis.split('heasoft-')[-1].split('src')[0]
    else:
        version = 'unknown'

    # Finally I save the version number in a file; I think HEASoft might write a version environment variable
    #  itself but it would be nice to have an independent version
    with open('heasoft_version.txt', 'w') as versiono:
        versiono.write(version)

    # The download is written to this compressed tar, once decompressed the folder name will have the version
    #  of HEASoft in it
    with open('heasoft.tar.gz', 'wb') as writo:
        shutil.copyfileobj(streamo.raw, writo)
