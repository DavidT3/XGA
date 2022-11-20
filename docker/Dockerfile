#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 19/11/2022, 19:56. Copyright (c) The Contributors

# Dockerfile for creating a complete functioning environment for the X-ray: Generate and Analyse (XGA) Python
#  module within a Docker container
#  Inspired by the official HEASOFT Dockerfile (https://heasarc.gsfc.nasa.gov/docs/software/heasoft/docker.html)

#FROM ubuntu:20.04
FROM condaforge/mambaforge:22.9.0-1

# This ensures that the global environment variables which describe the platform that this image is being built
#  for are available in this local space
ARG TARGETPLATFORM
ARG TARGETOS
ARG TARGETARCH

RUN echo "I'm building for $TARGETPLATFORM"

# Make some checks to ensure that the target platform (both architecture and OS) are allowable for this image
RUN if [ $TARGETOS != "linux" ]; then echo 'This image can only be built for a Linux target platform'; exit 1; \
      elif [ $TARGETARCH != "arm64" ] && [ $TARGETARCH != "amd64" ];  \
        then echo 'This image can only be build for amd64 or arm64 architectures'; exit 1; fi

#
ARG xga_version=0.3.0
ARG heasoft_version=6.31
ARG sas_version=20.0.0
# ARG lira_version=

ENV XGA_VERSION=${xga_version}
ENV HEASOFT_VERSION=${heasoft_version}
ENV SAS_VERSION=${sas_version}
# ENV LIRA_VERSION=${lira_version}

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
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* \
 && mamba create --name=xga_env python=3.8 fitsio=1.1.7 && mamba init && . ~/.bashrc && mamba activate xga_env &&  \
    pip install xga==${xga_version}


#RUN groupadd heasoft && useradd -r -m -g heasoft heasoft \
# && mkdir -p /opt/heasoft/caldb \
# && chown -R heasoft:heasoft /opt/heasoft
#
#USER heasoft
#WORKDIR /home/heasoft
#
## Retrieve the HEASoft source code, unpack, configure,
## make, install, clean up, and create symlinks....
#ARG heasoft_tarfile_suffix=src
#RUN wget https://heasarc.gsfc.nasa.gov/FTP/software/lheasoft/lheasoft${HEASOFT_VERSION}/heasoft-${HEASOFT_VERSION}${heasoft_tarfile_suffix}.tar.gz \
# && echo "Unpacking heasoft-${HEASOFT_VERSION}${heasoft_tarfile_suffix}.tar.gz..." \
# && tar xzf heasoft-${HEASOFT_VERSION}${heasoft_tarfile_suffix}.tar.gz \
# && rm -f heasoft-${HEASOFT_VERSION}${heasoft_tarfile_suffix}.tar.gz \
# && cd ${HOME}/heasoft-${HEASOFT_VERSION}/BUILD_DIR/ \
# && ./configure --prefix=/opt/heasoft 2>&1 | tee ${HOME}/configure.log \
# && make 2>&1 | tee ${HOME}/build.log \
# && make install 2>&1 | tee ${HOME}/install.log \
# && /bin/bash -c 'cd /opt/heasoft/; for loop in x86_64*/*; do ln -sf $loop; done' \
# && cd ${HOME}/heasoft-${HEASOFT_VERSION} \
# && cp -p Xspec/BUILD_DIR/hmakerc /opt/heasoft/bin/ \
# && cp -p Xspec/BUILD_DIR/Makefile-std /opt/heasoft/bin/ \
# && cd \
# && gzip -9 ${HOME}/*.log \
# && cp -p ${HOME}/heasoft-${HEASOFT_VERSION}/Release_Notes* /opt/heasoft/ \
# && rm -rf ${HOME}/heasoft-${HEASOFT_VERSION}*
#
## Configure shells...
#RUN /bin/echo >> /home/heasoft/.bashrc \
# && /bin/echo '# Initialize HEASoft environment' >> /home/heasoft/.bashrc \
# && /bin/echo 'export HEADAS=/opt/heasoft' >> /home/heasoft/.bashrc \
# && /bin/echo '. $HEADAS/headas-init.sh' >> /home/heasoft/.bashrc \
# && /bin/echo >> /home/heasoft/.bashrc \
# && /bin/echo '# Initialize environment for CALDB' >> /home/heasoft/.bashrc \
# && /bin/echo 'export CALDB=https://heasarc.gsfc.nasa.gov/FTP/caldb' >> /home/heasoft/.bashrc \
# && /bin/echo 'export CALDBCONFIG=/opt/heasoft/caldb/caldb.config' >> /home/heasoft/.bashrc \
# && /bin/echo 'export CALDBALIAS=/opt/heasoft/caldb/alias_config.fits' >> /home/heasoft/.bashrc \
# && /bin/echo >> /home/heasoft/.profile \
# && /bin/echo '# Initialize HEASoft environment' >> /home/heasoft/.profile \
# && /bin/echo 'export HEADAS=/opt/heasoft' >> /home/heasoft/.profile \
# && /bin/echo '. $HEADAS/headas-init.sh' >> /home/heasoft/.profile \
# && /bin/echo >> /home/heasoft/.profile \
# && /bin/echo '# Initialize environment for CALDB' >> /home/heasoft/.profile \
# && /bin/echo 'export CALDB=https://heasarc.gsfc.nasa.gov/FTP/caldb' >> /home/heasoft/.profile \
# && /bin/echo 'export CALDBCONFIG=/opt/heasoft/caldb/caldb.config' >> /home/heasoft/.profile \
# && /bin/echo 'export CALDBALIAS=/opt/heasoft/caldb/alias_config.fits' >> /home/heasoft/.profile \
# && /bin/echo '# Initialize HEASoft environment' >> /home/heasoft/.cshrc \
# && /bin/echo 'setenv HEADAS /opt/heasoft' >> /home/heasoft/.cshrc \
# && /bin/echo 'source $HEADAS/headas-init.csh' >> /home/heasoft/.cshrc \
# && /bin/echo >> /home/heasoft/.cshrc \
# && /bin/echo '# Initialize environment for CALDB' >> /home/heasoft/.cshrc \
# && /bin/echo 'setenv CALDB https://heasarc.gsfc.nasa.gov/FTP/caldb' >> /home/heasoft/.cshrc \
# && /bin/echo 'setenv CALDBCONFIG /opt/heasoft/caldb/caldb.config' >> /home/heasoft/.cshrc \
# && /bin/echo 'setenv CALDBALIAS /opt/heasoft/caldb/alias_config.fits' >> /home/heasoft/.cshrc
#
#RUN cd /opt/heasoft/caldb \
#    && wget https://heasarc.gsfc.nasa.gov/FTP/caldb/software/tools/caldb.config \
#    && wget https://heasarc.gsfc.nasa.gov/FTP/caldb/software/tools/alias_config.fits
#
#ENV CC=/usr/bin/gcc \
#    CXX=/usr/bin/g++ \
#    FC=/usr/bin/gfortran \
#    PERL=/usr/bin/perl \
#    PERLLIB=/opt/heasoft/lib/perl \
#    PERL5LIB=/opt/heasoft/lib/perl \
#    PYTHON=/usr/bin/python \
#    PYTHONPATH=/opt/heasoft/lib/python:/opt/heasoft/lib \
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
#
#CMD [ "fversion" ]