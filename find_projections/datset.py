import libfind_projections
import numpy as np

def validate_params(ds, binsize, support):
    if ds.isValid() is False:
        return False

    size = ds.getSize()

    if ( binsize <= 0 )  or ( binsize >= size ):
        return False

    if ( support >= size ) :
        return False

    return True

class Datset:

     """
     Create Datset instance with numpy 2-d array of floats.
     """
     def __init__(self, data):
         rows = data.shape[0]
         cols = data.shape[1]

         if data.dtype not in np.sctypes['float']:
             return None

         self.ds = libfind_projections.Datset(data)

     """
     Set output array for classification task
     """
     def setOutputForClassification(self, output):
         if ( np.issubdtype(output.dtype, np.float ) ) :
            self.ds.fill_datset_output_for_classification(output)
         else:
            raise Exception("Invalid classification data type")

     """
     Set output array for regression task
     """
     def setOutputForRegression(self, output):
         if ( np.issubdtype(output.dtype, np.float ) ) :
            self.ds.fill_datset_output_for_regression(output)
         else:
            raise Exception("Invalid regressionion data type")

     """
     Checks if Datset instance has been populated properly.
     """
     def isValid(self):
         return self.ds.is_valid()

     """
     Returns the number of data points
     """
     def getSize(self):
         return self.ds.get_size()
