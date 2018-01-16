Package find_projections
-----------------------------
Comprehensively searches for informative projections (2-d projections) in the data to find boxes which separate out homogeneous data points. 
Returns all projection boxes which match search criteria (support, purity).

For discrete output, the algorithm tries to find 2-d projection boxes in the raw feature space which can separate out any class of data from the rest with high purity.
For numeric output, the algorithm tries to find 2-d projection boxesin the raw feature space  which can separate out data points  with low variance.

This package has been written in C++ with Python wrapper (Python 3.6+).
Uses boost ver. 1.65.
Uses pthreads for multi-threading support.

Run the following command for building and installing package-

python setup.py install

In python,
import find_projections.

API documentation is included in Python_API_DOC.pdf and usage has been shown in test_projections.py.

Hyperparameter names and values -
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

Primitive Demo in Jupyter Notebook
-----------------------------
Opening demo.ipynb in Jupyter facilities running multiple sample experiments on seed datasets

demo.ipynb: Includes many types of experiments on various datasets

demo_classification.ipynb: Includes basic classification mode experiments on seed datasets

demo_scripts/ holds functions used in the demo notebooks and d3m_data/ holds seed datasets

load_d3m_data.py - Data for demo is loaded from raw seed datasets and then manipulated to remove missing values.  Each seed dataset is handled differently

mod.py - Functions facilitating use of projections in decision list models

rank.py - Functions for sorting output by purity or support used to identify most interesting projections

utils.py - Functions for manipulating raw datasets

viz.py - Functions for visualizing projection output as boxes in 2D space
