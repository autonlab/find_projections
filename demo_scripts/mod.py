def make_dlist(projections):
    dlist = [[]]*len(projections)
    for p in xrange(len(projections)):
        dlist[p]=[projections[p].get_class(), projections[p].get_att1(), projections[p].get_att2(),\
                  projections[p].get_att1_start(), projections[p].get_att1_end(),\
                  projections[p].get_att2_start(), projections[p].get_att2_end()]
    return dlist
    
def print_dlist(dlist):
    for p in dlist:
        print 'class:',p[0],'att1:',p[1],'att2:',p[2],'xmin:',p[3],'xmax:',p[4],'ymin:',p[5],'ymax:',p[6]
        
def dlist_make_predictions(dlist,data):
    predictions=[-1]*data.shape[0]
    for i in xrange(data.shape[0]):
        d=data[i,:]
        for b in dlist:
            if d[b[1]]>=b[3] and d[b[1]]<=b[4] and d[b[2]]>=b[5] and d[b[2]]<=b[6]:
                predictions[i]=b[0]
    return predictions

def points_inside_box(pr,data,labels,d3mIndex):
	bounds = [pr.get_att1_start(),pr.get_att1_end(),\
              pr.get_att2_start(),pr.get_att2_end()]
	dims = [pr.get_att1(),pr.get_att2()]
	box=[i for i in dims]
	box.extend([j for j in bounds])
	points_inside = []
	labels_inside = []
	d3mIdx_inside = []
	for i in xrange(data.shape[0]):
		d=data[i,:]
		l=labels[i]
		idx=d3mIndex[i]
		if d[box[0]]>=box[2] and d[box[0]]<=box[3] and d[box[1]]>=box[4] and d[box[1]]<=box[5]:
			points_inside.append(d)
			labels_inside.append(l)
			d3mIdx_inside.append(idx)
	return d3mIdx_inside ,labels_inside

