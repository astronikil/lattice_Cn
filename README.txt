This is a set of python (v3.0) codes to generate mock ITD data for
pion and proton to test the idea of obtaining the Wilson coefficients
(of say the proton) from the lattice and use it in the analysis of
PDFs of other hadrons, say the pion.

The Jupyter notebook where analysis is done is analysis.ipynb

The other python routines are given in *.py files.

You have to change the following files if you want to change the
way mock data is generated; here are some relevant files that might
need to be changed:

1) function "power_covariance" in gen_itd.py models the error bars
and the covariance between the data. Change this to change the
model.

2) functions "higher_twist" and "lattice_artifact" in gen_itd.py
model the higher-twist and
   small-z lattice artifacts.

2) functions c_artificial_log0, c_artificial_log1, and c_artificial_log2
in wil_coeff.py, model the higher-loop corrections to NLO. These
functions have to be changed to include another model.

3) params.py contains some relevant constants and the value of
alpha_s used. Change this based on mu that you use.

4) ... changing anything else might break the code!!!
