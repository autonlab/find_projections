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

     """
     Returns the default class/true mean of the output column
     """
     def get_default_value(self):
         return self.ds.get_default_value()
    
     def get_real_ref(self, i, j):
         return self.ds.ds_real_ref(i, j)     
   
     def _point_within_line(self, row, att, start, end):
         value = self.get_real_ref(row, att)
         if value >= start and value <= end:
             return True
         return False

     def _point_lies_in_projection(self, row, att1, start1, end1, att2, start2, end2):
         a1 = self._point_within_line(row, att1, start1, end1)
         a2 = self._point_within_line(row, att2, start2, end2)
         final = a1 and a2
         return final

     def _helper(self, fmap, row):
         for pr in fmap:
             att1 = pr[0]
             att2 = pr[1]
             start1 = pr[2]
             start2 = pr[3]
             end1 = pr[4]
             end2 = pr[5]
             value = pr[6]

             valid = self._point_lies_in_projection(row, att1, start1, end1, att2, start2, end2)
             if valid is True:
                 return (value, True)
         return (0, False)
  
