Package find_projections
-----------------------------
Comprehensively searches for informative projections (2-d projections) in the data to find boxes which separate out homogeneous data points. 
Returns all projection boxes which match search criteria (support, purity).

For discrete output, the algorithm tries to find 2-d projection boxes which can separate out any class of data from the rest with high purity.
For numeric output, the algorithm tries to find 2-d projection boxes which can separate out data points  with low variance.

This package has been written in C++ with Python wrapper (Python 2.7).
Uses boost ver. 1.65.
Uses pthreads for multi-threading support.

Run the following command for building and installing package-

python setup.py install

In python,
import find_projections.

For creating a wheel distribution, run python setup.py bdist_wheel.

API documentation is included in Python_API_DOC.pdf and usage has been shown in test.py.

Parameter names and values -
-----------------------------
binsize - 10 (Should be positive integer denoting the min. number of data points for binning data at each leaf
num_threads - 1 (If >1, will run multi-threaded on Linux)
support - 100 (Min. no. of data points in each projection found)
purity - 0.9 (Purity of each projection found)
mode - 1 (For numeric output, tries to find high-output boxes with low variance)

Valid values for mode are -
0 : Tries to find low variance boxes
1 : Tries to find high mean boxes
2 : Tries to find low mean boxes
