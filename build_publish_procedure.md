# Publishing a new version of XGA

This very short guide is for my own benefit, so I'll remember how I set all of this up the next time I need to do this. The process is largely automated now, but there are a couple of steps to it that have to be done in the right way.

## The steps
1) Check that all dependencies in setup.py and requirements.txt are up to date and correct.
2) Check that no new files or directories need to be included in the MANAFEST.in
3) Checkout the master branch to vx.x.x, then tag it (using Pycharm put the commit as HEAD).
4) Push to remote (make sure to include tags) - this will trigger the build and publishing to test PyPI
5) STOP - Now check that the install from ```pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple xga``` (I have had issues with fitsio and regions v0.4 on PIP, on Apollo, but conda install works).
6) Now do a release on the GitHub website - this should trigger the build and publishing to the real PyPI index.

