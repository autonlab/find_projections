import csv
import numpy as np

def csv2list(path,header=True,save_header=False):
	ds=[]
	with open(path,'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		for row in reader:
			ds.append(row)
	if save_header:
		return (ds[1:],ds[0])
	elif header:
		return ds[1:]
	else:
		return ds

def map2int(categorical_feature,dictionary=None):
	if not dictionary:
		key = list(set(categorical_feature))
		val = xrange(len(key))
		dictionary = {key[i]:val[i] for i in xrange(len(key))}
	int_feature=[]
	for i in categorical_feature:
		int_feature.append(dictionary[i])
	return (int_feature,dictionary)

def del_feats(ds,sorted_feat_nums_to_del):
	if len(ds.shape) is 1:
		dim=0
	else:
		dim=1

	for e in sorted_feat_nums_to_del:
		if e<0:
			e+=ds.shape[dim]
		ds=np.delete(ds,e,dim)
	return ds

def del_feats_w_misval_thresh(train,test,header,threshold=50):
	for i in xrange(train.shape[1]-1,0,-1):
		if(sum([1 for x in train[:,i] if not x]) > threshold):
			train=np.delete(train,i,1)
			test=np.delete(test,i,1)
			header=np.delete(header,i)
	return (train,test,header)

def del_rows_w_misval(ds,rowID=None,label=None):
	for i in xrange(ds.shape[0]-1,0,-1):
		for e in ds[i,:]:
			if not e:
				ds=np.delete(ds,i,0)
				if rowID is not None:
					rowID =np.delete(rowID,i)
				if label is not None:
					label=np.delete(label,i)
				break
	if label is None and rowID is None:
		return (ds)
	elif label is None:
		return (ds,rowID)
	elif rowID is None:
		return (ds,label)
	else:
		return (ds,rowID,label)

def discrete_projection_to_list( p ):
	return [ p.get_class(), p.get_pos()/float(p.get_total()), p.get_total(), p.get_att1(), p.get_att2(), p.get_att1_start(), p.get_att1_end(), p.get_att2_start(), p.get_att2_end()]

def numeric_projection_to_list( p ):
	return [ p.get_total(), p.get_att1(), p.get_att2(), p.get_att1_start(), p.get_att1_end(), p.get_att2_start(), p.get_att2_end() ]
