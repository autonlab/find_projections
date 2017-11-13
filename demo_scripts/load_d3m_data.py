from utils import *

def load_protein_dataset():
	#Load protein dataset
	(protein_train_data,protein_headers)=csv2list('d3m_data/protein_4550/data/trainData.csv',save_header=True)
	protein_test_data=csv2list('d3m_data/protein_4550/data/testData.csv')
	protein_train_targets=csv2list('d3m_data/protein_4550/data/trainTargets.csv')
	
	# Extract train targets
	protein_train_label=[protein_train_targets[i][1] for i in xrange(len(protein_train_targets))]
	protein_train_d3m_index=[protein_train_data[i][0] for i in xrange(len(protein_train_data))]
	protein_test_d3m_index=[protein_test_data[i][0] for i in xrange(len(protein_test_data))]
	
	# Convert from lists to numpy arrays
	protein_train_data = np.asarray(protein_train_data)
	protein_train_label = np.asarray(protein_train_label)
	protein_test_data = np.asarray(protein_test_data)
	protein_headers = np.asarray(protein_headers)
	
	# Map the text label to a categorical integer label
	(protein_train_label,protein_label_dict) = map2int(protein_train_label)
	
	#Last 3 features are pseudo labels...let's remove them
	#First feature is d3mIndex, then mouse_id...let's remove them for now
	protein_train_data=del_feats(protein_train_data,[-3,-2,-1,1,0])
	protein_test_data=del_feats(protein_test_data,[-3,-2,-1,1,0])
	protein_headers=del_feats(protein_headers,[-3,-2,-1,1,0])
	
	#Get rid of features with many missing values, then samples with any missing values
	(protein_train_data,protein_test_data,protein_headers) = del_feats_w_misval_thresh(protein_train_data,protein_test_data,protein_headers,50)
	(protein_train_data,protein_train_d3m_index,protein_train_label)=del_rows_w_misval(protein_train_data,protein_train_d3m_index,protein_train_label)
	(protein_test_data,protein_test_d3m_index)=del_rows_w_misval(protein_test_data,protein_test_d3m_index)   
	
	protein_train_data = protein_train_data.astype(float)
	protein_test_data = protein_test_data.astype(float)
	protein_train_label = protein_train_label.astype(int)
	protein_train_d3m_index = protein_train_d3m_index.astype(int)
	protein_test_d3m_index = protein_test_d3m_index.astype(int)

	return (protein_train_data,protein_train_label,protein_test_data,protein_train_d3m_index,protein_test_d3m_index,protein_label_dict,protein_headers)

def load_baseball_dataset():

	#Load baseball dataset
	(baseball_train_data,baseball_headers)=csv2list('d3m_data/baseball_185/data/trainData.csv',save_header=True)
	baseball_test_data=csv2list('d3m_data/baseball_185/data/testData.csv')
	baseball_train_targets=csv2list('d3m_data/baseball_185/data/trainTargets.csv')
	
	# Extract train targets
	baseball_train_label=[baseball_train_targets[i][1] for i in xrange(len(baseball_train_targets))]
	baseball_train_d3m_index=[baseball_train_data[i][0] for i in xrange(len(baseball_train_data))]
	baseball_test_d3m_index=[baseball_test_data[i][0] for i in xrange(len(baseball_test_data))]
	
	# Convert from lists to numpy arrays
	baseball_train_data = np.asarray(baseball_train_data)
	baseball_train_label = np.asarray(baseball_train_label)
	baseball_test_data = np.asarray(baseball_test_data)
	baseball_headers = np.asarray(baseball_headers)
	
	
	(baseball_train_data[:,-1],baseball_position_dict)=map2int(baseball_train_data[:,-1])
	
	(baseball_test_data[:,-1],baseball_position_dict) = map2int(baseball_test_data[:,-1],baseball_position_dict)
	baseball_label_dict={'0':0,'1':1,'2':1}	
	(baseball_train_label,baseball_label_dict)=map2int(baseball_train_label,baseball_label_dict)
	baseball_train_data=del_feats(baseball_train_data,[1,0])
	baseball_test_data=del_feats(baseball_test_data,[1,0])
	baseball_headers=del_feats(baseball_headers,[1,0])
	
	
	#Get rid of features with many missing values, then samples with any missing values
	(baseball_train_data,baseball_test_data,baseball_headers) = del_feats_w_misval_thresh(baseball_train_data,baseball_test_data,baseball_headers,50)
	(baseball_train_data,baseball_train_d3m_index,baseball_train_label)=del_rows_w_misval(baseball_train_data,baseball_train_d3m_index,baseball_train_label)
	(baseball_test_data,baseball_test_d3m_index)=del_rows_w_misval(baseball_test_data,baseball_test_d3m_index)   
	
	baseball_train_data = baseball_train_data.astype(float)
	baseball_test_data = baseball_test_data.astype(float)
	baseball_train_label = baseball_train_label.astype(int)
	baseball_train_d3m_index = baseball_train_d3m_index.astype(int)
	baseball_test_d3m_index = baseball_test_d3m_index.astype(int)

	return (baseball_train_data,baseball_train_label,baseball_test_data,baseball_train_d3m_index,baseball_test_d3m_index,baseball_position_dict,baseball_label_dict,baseball_headers)



def load_auto_dataset():
	#Load auto mpg dataset
	
	(auto_train_data,auto_headers)=csv2list('d3m_data/autompg_196/data/trainData.csv',save_header=True)
	auto_test_data=csv2list('d3m_data/autompg_196/data/testData.csv')
	auto_train_targets=csv2list('d3m_data/autompg_196/data/trainTargets.csv')
	
	auto_train_label=[auto_train_targets[i][1] for i in xrange(len(auto_train_targets))]
	auto_train_d3m_index=[auto_train_data[i][0] for i in xrange(len(auto_train_data))]
	auto_test_d3m_index=[auto_test_data[i][0] for i in xrange(len(auto_test_data))]
	
		# Convert from lists to numpy arrays
	auto_train_data = np.asarray(auto_train_data)
	auto_train_label = np.asarray(auto_train_label)
	auto_test_data = np.asarray(auto_test_data)
	auto_headers = np.asarray(auto_headers)
	
	auto_train_data=del_feats(auto_train_data,[0])
	auto_test_data=del_feats(auto_test_data,[0])
	auto_headers=del_feats(auto_headers,[0])
	
	#Get rid of features with many missing values, then samples with any missing values
	(auto_train_data,auto_test_data,auto_headers)=del_feats_w_misval_thresh(auto_train_data,auto_test_data,auto_headers,50)
	(auto_train_data,auto_train_d3m_index,auto_train_label)=del_rows_w_misval(auto_train_data,auto_train_d3m_index,auto_train_label)
	(auto_test_data,auto_test_d3m_index)=del_rows_w_misval(auto_test_data,auto_test_d3m_index)
	
	auto_train_data = auto_train_data.astype(float)
	auto_test_data = auto_test_data.astype(float)
	auto_train_label = auto_train_label.astype(float)
	auto_train_d3m_index = auto_train_d3m_index.astype(int)
	auto_test_d3m_index = auto_test_d3m_index.astype(int)

	return (auto_train_data,auto_train_label,auto_test_data,auto_train_d3m_index,auto_test_d3m_index,auto_headers)


def load_radon_dataset():
	#Load radon dataset
	
	radon_train_data=csv2list('d3m_data/radon_r26/data/trainData.csv')
	radon_test_data=csv2list('d3m_data/radon_r26/data/testData.csv')
	radon_train_targets=csv2list('d3m_data/radon_r26/data/trainTargets.csv')
	(radon_raw_data,radon_headers)=csv2list('d3m_data/radon_r26/data/raw_data/radon.csv',save_header=True)
	
	radon_train_label=[radon_train_targets[i][1] for i in xrange(len(radon_train_targets))]
	radon_train_d3m_index=[radon_train_data[i][0] for i in xrange(len(radon_train_data))]
	radon_test_d3m_index=[radon_test_data[i][0] for i in xrange(len(radon_test_data))]
	
	# Convert from lists to numpy arrays
	radon_train_data = np.asarray(radon_train_data)
	radon_train_label = np.asarray(radon_train_label).astype(float)
	radon_test_data = np.asarray(radon_test_data)
	radon_headers = np.asarray(radon_headers)
	radon_raw_data = np.asarray(radon_raw_data)
	
	train_samples = radon_train_data[:,2].astype(int)
	test_samples = radon_test_data[:,2].astype(int)
	raw_instances = radon_raw_data[:,0].astype(int)
	
	radon_raw_data=del_feats(radon_raw_data,[-4,10,2,1,0])
	radon_headers=del_feats(radon_headers,[-4,10,2,1,0])
	
	#Get rid of features with many missing values, then samples with any missing values
	(radon_raw_data,[],radon_headers)=del_feats_w_misval_thresh(radon_raw_data,[],radon_headers,50)
	
	(radon_raw_data[:,6],radon_basement_dict)=map2int(radon_raw_data[:,6])
	
    	
	train_ids=[]
	test_ids=[]
	for i in xrange(radon_raw_data.shape[0]):
    		if(raw_instances[i] in train_samples):
        		train_ids.append(i)
    		elif(raw_instances[i] in test_samples):
        		test_ids.append(i)
	
	radon_train_data = np.asarray([radon_raw_data[i] for i in train_ids]).astype(float)
	radon_test_data = np.asarray([radon_raw_data[i] for i in test_ids]).astype(float)
	radon_train_label_raw = radon_train_label.astype(float)
	radon_train_label = [radon_train_label_raw[i]-min(radon_train_label_raw) for i in xrange(len(radon_train_label_raw))]
	radon_train_d3m_index= np.asarray(radon_train_d3m_index).astype(int)
	radon_test_d3m_index= np.asarray(radon_test_d3m_index).astype(int)


	return (radon_train_data,radon_train_label,radon_test_data,radon_train_d3m_index,radon_test_d3m_index,radon_headers)



def load_spectrometer_dataset():

	#Load spectrometer dataset
	(spectro_train_data,spectro_headers)=csv2list('d3m_data/spectrometer_313/data/trainData.csv',save_header=True)
	spectro_test_data=csv2list('d3m_data/spectrometer_313/data/testData.csv')
	spectro_train_targets=csv2list('d3m_data/spectrometer_313/data/trainTargets.csv')
	
	spectro_train_label = [spectro_train_targets[i][1] for i in xrange(len(spectro_train_targets))]
	spectro_train_d3m_index = [spectro_train_data[i][0] for i in xrange(len(spectro_train_data))]
	spectro_test_d3m_index = [spectro_test_data[i][0] for i in xrange(len(spectro_test_data))]
	
	# Convert from lists to numpy arrays
	spectro_train_data = np.asarray(spectro_train_data)
	spectro_train_label = np.asarray(spectro_train_label)
	spectro_test_data = np.asarray(spectro_test_data)
	spectro_headers = np.asarray(spectro_headers)
	
	spectro_train_data=del_feats(spectro_train_data,[1,0])
	spectro_headers = del_feats(spectro_headers,[1,0])
	spectro_test_data = del_feats(spectro_test_data,[1,0])
	
	spectro_train_data = spectro_train_data.astype(float)
	spectro_test_data = spectro_test_data.astype(float)
	spectro_train_label = spectro_train_label.astype(int)/10
	spectro_train_d3m_index = np.asarray(spectro_train_d3m_index).astype(int)
	spectro_test_d3m_index = np.asarray(spectro_test_d3m_index).astype(int)

	return (spectro_train_data,spectro_train_label,spectro_test_data,spectro_train_d3m_index,spectro_test_d3m_index,spectro_headers)

def load_thyroid_dataset():

	#Load thyroid dataset
	(thyroid_train_data,thyroid_headers)=csv2list('d3m_data/thyroid_38/data/trainData.csv',save_header=True)
	thyroid_test_data=csv2list('d3m_data/thyroid_38/data/testData.csv')
	thyroid_train_targets=csv2list('d3m_data/thyroid_38/data/trainTargets.csv')
	
	thyroid_train_label = [thyroid_train_targets[i][1] for i in xrange(len(thyroid_train_targets))]
	thyroid_train_d3m_index = [thyroid_train_data[i][0] for i in xrange(len(thyroid_train_data))]
	thyroid_test_d3m_index = [thyroid_test_data[i][0] for i in xrange(len(thyroid_test_data))]
	
	
	# Convert from lists to numpy arrays
	thyroid_train_data = np.asarray(thyroid_train_data)
	thyroid_train_label = np.asarray(thyroid_train_label)
	thyroid_test_data = np.asarray(thyroid_test_data)
	thyroid_headers = np.asarray(thyroid_headers)
	
	thyroid_train_data=del_feats(thyroid_train_data,[0])
	thyroid_test_data=del_feats(thyroid_test_data,[0])
	thyroid_headers=del_feats(thyroid_headers,[0])
	
	(thyroid_train_data,thyroid_test_data,thyroid_headers)=del_feats_w_misval_thresh(thyroid_train_data,thyroid_test_data,thyroid_headers,1000)
	(thyroid_train_data,thyroid_train_d3m_index,thyroid_train_label)=del_rows_w_misval(thyroid_train_data,thyroid_train_d3m_index,thyroid_train_label)
	(thyroid_test_data,thyroid_test_d3m_index)=del_rows_w_misval(thyroid_test_data,thyroid_test_d3m_index)
	
	(thyroid_train_data[:,-1],thyroid_referral_dict)=map2int(thyroid_train_data[:,-1])
	(thyroid_test_data[:,-1],thyroid_referral_dict)=map2int(thyroid_test_data[:,-1],thyroid_referral_dict)
	
	(thyroid_train_data[:,1],thyroid_sex_dict)=map2int(thyroid_train_data[:,1])
	(thyroid_test_data[:,1],thyroid_sex_dict)=map2int(thyroid_test_data[:,1],thyroid_sex_dict)
	
	thyroid_truth_dict={'f':0,'t':1}
	for c in xrange(thyroid_train_data.shape[1]):
    		key = list(set(thyroid_train_data[:,c]))
    		if key==['t','f'] or key==['f','t'] or key==['f'] or key==['t']:
        		(thyroid_train_data[:,c],thyroid_truth_dict)=map2int(thyroid_train_data[:,c],thyroid_truth_dict)
        		(thyroid_test_data[:,c],thyroid_truth_dict)=map2int(thyroid_test_data[:,c],thyroid_truth_dict)
	
	(thyroid_train_label,thyroid_label_dict)=map2int(thyroid_train_label)
	(thyroid_train_data,thyroid_train_d3m_index,thyroid_train_label)=del_rows_w_misval(thyroid_train_data,thyroid_train_d3m_index,thyroid_train_label)
	(thyroid_test_data,thyroid_test_d3m_index)=del_rows_w_misval(thyroid_test_data,thyroid_test_d3m_index)
	
	thyroid_train_data = thyroid_train_data.astype(float)
	thyroid_test_data = thyroid_test_data.astype(float)
	thyroid_train_label = np.asarray(thyroid_train_label).astype(int)
	thyroid_train_d3m_index = np.asarray(thyroid_train_d3m_index).astype(int)
	thyroid_test_d3m_index = np.asarray(thyroid_test_d3m_index).astype(int)


	return (thyroid_train_data,thyroid_train_label,thyroid_test_data,thyroid_train_d3m_index,thyroid_test_d3m_index,thyroid_headers,thyroid_referral_dict,thyroid_sex_dict,thyroid_truth_dict,thyroid_label_dict)



def load_imseg_dataset():

	#Load imseg dataset
	(imseg_train_data,imseg_headers)=csv2list('d3m_data/o_imseg/data/trainData.csv',save_header=True)
	imseg_test_data=csv2list('d3m_data/o_imseg/data/testData.csv')
	imseg_train_targets=csv2list('d3m_data/o_imseg/data/trainTargets.csv')
	imseg_test_targets=csv2list('d3m_data/o_imseg/data/testTargets.csv')
	imseg_train_label = [imseg_train_targets[i][1] for i in xrange(len(imseg_train_targets))]
	imseg_train_d3m_index=[imseg_train_data[i][0] for i in xrange(len(imseg_train_data))]
	imseg_test_d3m_index =[imseg_test_data[i][0] for i in xrange(len(imseg_test_data))]
	imseg_test_label = [imseg_test_targets[i][1] for i in xrange(len(imseg_test_targets))]
	
	# Convert from lists to numpy arrays
	imseg_train_data = np.asarray(imseg_train_data)
	imseg_train_label = np.asarray(imseg_train_label)
	imseg_test_data = np.asarray(imseg_test_data)
	imseg_headers = np.asarray(imseg_headers)
	imseg_train_d3m_index = np.asarray(imseg_train_d3m_index)
	imseg_test_d3m_index = np.asarray(imseg_test_d3m_index)
	imseg_test_label = np.asarray(imseg_test_label)
	
	imseg_train_data=del_feats(imseg_train_data,[0])
	imseg_test_data=del_feats(imseg_test_data,[0])
	imseg_headers=del_feats(imseg_headers,[0])
	
	(imseg_train_label,imseg_label_dict)=map2int(imseg_train_label)
	(imseg_test_label,imseg_label_dict)=map2int(imseg_test_label,imseg_label_dict)
	
	imseg_train_data = imseg_train_data.astype(float)
	imseg_test_data = imseg_test_data.astype(float)
	imseg_train_label = np.asarray(imseg_train_label).astype(int)
	imseg_train_d3m_index = imseg_train_d3m_index.astype(int)
	imseg_test_d3m_index = imseg_test_d3m_index.astype(int)
	imseg_test_label = np.asarray(imseg_test_label).astype(int)

	return(imseg_train_data,imseg_train_label,imseg_test_data,imseg_test_label,imseg_train_d3m_index,imseg_test_d3m_index,imseg_headers,imseg_label_dict)
