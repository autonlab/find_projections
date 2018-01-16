import libfind_projections

class FeatureMap:
    """
    Container class containing projection boxes found from search operations
    """
    def __init__(self, fmap):
        self.fmap = fmap

    """
    Returns the total number of projection boxes in this container object
    """
    def get_num_projections(self):
        return self.fmap.get_num_projections()

    """
    Retrieve the i'th projection-box
    """
    def get_projection(self, i):
        if ( i <  0 ) :
            return None
        return self.fmap.get_projection(i)
