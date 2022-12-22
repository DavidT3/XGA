#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 19/11/2022, 19:56. Copyright (c) The Contributors

# Dockerfile for creating a complete functioning environment for the X-ray: Generate and Analyse (XGA) Python
#  module within a Docker container
#  Inspired by the official HEASOFT Dockerfile (https://heasarc.gsfc.nasa.gov/docs/software/heasoft/docker.html)

# NOTES FOR FUTURE DAVID
#  https://pythonspeed.com/articles/activate-conda-dockerfile/ - has info on activating Conda in Docker

#FROM ubuntu:20.04
FROM condaforge/mambaforge:22.9.0-1

# This ensures that the global environment variables which describe the platform that this image is being built
#  for are available in this local space
ARG TARGETPLATFORM
ARG TARGETOS
ARG TARGETARCH

# Make some checks to ensure that the target platform (both architecture and OS) are allowable for this image
RUN #if [ $TARGETOS != "linux" ]; then echo 'This image can only be built for a Linux target platform'; exit 1; \
#      elif [ $TARGETARCH != "arm64" ] && [ $TARGETARCH != "amd64" ];  \
#        then echo 'This image can only be build for amd64 or arm64 architectures'; exit 1; fi

#
ARG xga_version=0.3.0
ARG heasoft_version=6.31
ARG sas_version=20.0.0
# ARG lira_version=

ENV XGA_VERSION=${xga_version}
ENV HEASOFT_VERSION=${heasoft_version}
ENV SAS_VERSION=${sas_version}
# ENV LIRA_VERSION=${lira_version}
ENV SAS_PERL=/usr/bin/perl

LABEL version="${xga_version}" \
      description="X-ray: Generate and Analyse (XGA) v${version} https://pypi.org/project/xga/" \
      maintainer="David J Turner"

# Install base utilities + HEASoft prerequisites
# Last line creates a Conda environment for XGA (installing fitsio through Conda because otherwise ARM \
#  builds/possibly others will error) and then installs it. This will have to be updated once I finally
#  release XGA on Conda
# For context Mamba is a C++ rewrite of the Conda package manager, is much faster, and is a drop in replacement
#  for Conda
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get -y upgrade \
 && apt-get -y install \
	gcc \
	gfortran \
	g++ \
	libcurl4 \
	libcurl4-gnutls-dev \
	libncurses5-dev \
	libreadline6-dev \
	make \
	ncurses-dev \
	perl-modules \
	tcsh \
	vim \
	wget \
	xorg-dev \
    nano \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN groupadd xga && useradd -r -m -g xga xga \
 && mkdir -p /opt/heasoft \
 && mkdir -p /opt/sas \
 && chown -R xga:xga /opt/heasoft \
 && chown -R xga:xga /opt/sas \
 && chown -R xga:xga /opt/conda

# TODO REMOVE THIS IN THE END, JUST MAKES IT QUICKER THAN DOWNLOADING EVERY TIME I WANT TO BUILD WITHOUT CACHE
COPY heasoft.tar.gz /home/xga
COPY sas_20.0.0-Ubuntu20.04.tgz /home/xga
RUN chown -R xga:xga /home/xga/heasoft.tar.gz && \
 chown -R xga:xga /home/xga/sas_20.0.0-Ubuntu20.04.tgz

USER xga
WORKDIR /home/xga

#COPY heasoft_download.py /home/xga

# Retrieve the HEASoft source code, unpack, configure,
# make, install, clean up, and create symlinks....
RUN echo Downloading a minimal HEASoft installation...  \
# && python heasoft_download.py \
 && echo "Unpacking HEASoft..." \
 && mkdir heasoft \
 && mkdir sas \
 && tar -xzf heasoft.tar.gz -C heasoft --strip-components=1 \
 && tar -xf sas_20.0.0-Ubuntu20.04.tgz -C /opt/sas/ \
# && chown -R xga:xga /home/heasoft \
 && rm -f heasoft.tar.gz \
 && rm -f sas_20.0.0-Ubuntu20.04.tgz

RUN mamba create --name=xga_env python=3.8 fitsio=1.1.7

# Make RUN commands use the new environment:
SHELL ["mamba", "run", "-n", "xga_env", "/bin/bash", "--login", "-c"]

RUN cd ${HOME}/heasoft/BUILD_DIR/ \
 && pip install xga \
 && echo "Building HEASoft..." \
 && ./configure --prefix=/opt/heasoft 2>&1 | tee ${HOME}/configure.log \
 && make 2>&1 | tee ${HOME}/build.log \
 && make install 2>&1 | tee ${HOME}/install.log \
 && /bin/bash -c 'cd /opt/heasoft/; if compgen -G "x86_64*" > /dev/null; then for loop in x86_64*/*;  \
    do ln -sf $loop; done; else for loop in aarch64*/*; do ln -sf $loop; done; fi' \
 && cd \
 && gzip -9 ${HOME}/*.log \
 && cp -p ${HOME}/heasoft/Release_Notes* /opt/heasoft/ \
 && rm -rf ${HOME}/heasoft \
 && /bin/echo >> /home/xga/.bashrc \
 && /bin/echo '# Initialize HEASoft environment' >> /home/xga/.bashrc \
 && /bin/echo 'export HEADAS=/opt/heasoft' >> /home/xga/.bashrc \
 && /bin/echo '. $HEADAS/headas-init.sh' >> /home/xga/.bashrc

# Install SAS
RUN cd /opt/sas \
 && ./install.sh \
 && cd ${HOME} \
 && /bin/echo 'conda activate xga_env' >> /home/xga/.bashrc \
 && /bin/echo '# Initialize SAS' >> /home/xga/.bashrc \
 && /bin/echo 'export SAS_DIR=/opt/sas/xmmsas_*' >> /home/xga/.bashrc
# && /bin/echo 'source $SAS_DIR/setsas.sh' >> /home/xga/.bashrc
# && /bin/echo 'export SAS_CCFPATH=/home/xga/xmm_ccfs' >> /home/xga/.bashrc \

# && cp -p Xspec/BUILD_DIR/hmakerc /opt/heasoft/bin/ \
# && cp -p Xspec/BUILD_DIR/Makefile-std /opt/heasoft/bin/ \
##RUN cd /opt/heasoft/caldb \
##    && wget https://heasarc.gsfc.nasa.gov/FTP/caldb/software/tools/caldb.config \
##    && wget https://heasarc.gsfc.nasa.gov/FTP/caldb/software/tools/alias_config.fits
##
#ENV CC=/usr/bin/gcc \
#    CXX=/usr/bin/g++ \
#    FC=/usr/bin/gfortran \
#    PERL=/usr/bin/perl \
#    PERLLIB=/opt/heasoft/lib/perl \
#    PERL5LIB=/opt/heasoft/lib/perl \
##    PYTHON=/usr/bin/python \
##    PYTHONPATH=/opt/heasoft/lib/python:/opt/heasoft/lib \
#    PATH=/opt/heasoft/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
#    HEADAS=/opt/heasoft \
#    LHEASOFT=/opt/heasoft \
#    FTOOLS=/opt/heasoft \
#    LD_LIBRARY_PATH=/opt/heasoft/lib \
#    LHEAPERL=/usr/bin/perl \
#    PFCLOBBER=1 \
#    PFILES=/home/heasoft/pfiles;/opt/heasoft/syspfiles \
#    FTOOLSINPUT=stdin \
#    FTOOLSOUTPUT=stdout \
#    LHEA_DATA=/opt/heasoft/refdata \
#    LHEA_HELP=/opt/heasoft/help \
#    EXT=lnx \
#    PGPLOT_FONT=/opt/heasoft/lib/grfont.dat \
#    PGPLOT_RGB=/opt/heasoft/lib/rgb.txt \
#    PGPLOT_DIR=/opt/heasoft/lib \
#    POW_LIBRARY=/opt/heasoft/lib/pow \
#    XRDEFAULTS=/opt/heasoft/xrdefaults \
#    TCLRL_LIBDIR=/opt/heasoft/lib \
#    XANADU=/opt/heasoft \
#    XANBIN=/opt/heasoft \
#    CALDB=https://heasarc.gsfc.nasa.gov/FTP/caldb \
#    CALDBCONFIG=/opt/heasoft/caldb/caldb.config \
#    CALDBALIAS=/opt/heasoft/caldb/alias_config.fits

#CMD ["conda", "activate", "xga_env"]