import numpy as np
def rank_by_purity( parray ):
	if parray is not None:
		return parray[parray[:,1].argsort()]
	else:
		return None
	
def rank_by_support( parray):
	if parray is not None:
		return parray[parray[:,2].argsort()]
	else:
		return None

def get_all_of_class( parray, classnum ):
	if parray is not None:
		output=np.asarray([ parray[i] for i in xrange(parray.shape[0]) if parray[i][0]==classnum])
		if not output.any():
			return None
		else:
			return output
	else:
		return None

def top_of_purity( parray, num ):
	if parray is not None and num<parray.shape[0]:
		ind = np.argpartition(parray[:,1],-num)[-num:]
		return parray[ind]
	elif parray is not None:
		ind = np.argpartition(parray[:,1],-parray.shape[0])[-parray.shape[0]:]
		return parray[ind]
	else:
		return None

def top_of_support(parray, num):
	if parray is not None and num<parray.shape[0]:
		ind = np.argpartition(parray[:,2],-num)[-num:]
		return parray[ind]
	elif parray is not None:
		ind = np.argpartition(parray[:,2],-parray.shape[0])[-parray.shape[0]:]
		return parray[ind]
	else:
		return None

	
