Introduction to XGA
===================

XGA is a Python package to explore and analyse X-ray sources detected in XMM-Newton observations. When a source is
declared, XGA finds every observation it appears in, and all analysis is performed on all available data.
It provides an easy to use Python interface with XMM's Science Analysis System (SAS), enabling independent products to be generated in parallel on
local computers and HPCs. XGA also makes it easy to generate products for samples of objects.

XGA also provides a similar interface with the popular X-ray spectrum fitting tool XSPEC, and makes it extremely
simple to create an XSPEC script for whatever source you are analysing, run that script, and then read the results
back into Python. The XSPEC interface will also run multiple fits in parallel.

