import matplotlib
import matplotlib.pyplot as plt

def print_projection(pr):
        print pr.get_class(),pr.get_att1(),\
        pr.get_att2(),pr.get_total(),\
        pr.get_att1_start(),pr.get_att1_end(),\
        pr.get_att2_start(),pr.get_att2_end(),\
        pr.get_pos(),pr.get_neg(),\
        pr.get_pos()/float(pr.get_pos()+pr.get_neg())
        
def show_projection(pr,data,labels,headers):
    fig=plt.figure(figsize=(9,9),dpi=80,facecolor='w',edgecolor='k')
    bounds = [pr.get_att1_start(),pr.get_att1_end(),\
              pr.get_att2_start(),pr.get_att2_end()]
    dims = [pr.get_att1(),pr.get_att2()]
    
    x = data[:,dims[0]]
    y = data[:,dims[1]]
    l = labels
    
    fig = plt.figure()
    ax=fig.add_subplot(111)
    ax.add_patch(matplotlib.patches.Rectangle((bounds[0],bounds[2]),bounds[1]-bounds[0],bounds[3]-bounds[2],fill=False))
    matplotlib.pyplot.scatter(x,y,s=5,c=l,linewidths=.25)
    plt.xlabel(headers[dims[0]])
    plt.ylabel(headers[dims[1]])

def show_box( box, data,labels,headers):
	fig=plt.figure(figsize=(9,9),dpi=80,facecolor='w',edgecolor='k')
	bounds = box[5:]
	dims = box[3:5]
	dims = [int(i) for i in dims]
	x=data[:,dims[0]]
	y=data[:,dims[1]]
	l = labels

	fig=plt.figure()
	ax=fig.add_subplot(111)
	ax.add_patch(matplotlib.patches.Rectangle((bounds[0],bounds[2]),bounds[1]-bounds[0],bounds[3]-bounds[2],fill=False))
	matplotlib.pyplot.scatter(x,y,s=5,c=l,linewidths=.25)
	plt.xlabel(headers[dims[0]])
	plt.ylabel(headers[dims[1]])


def show_projection_rlabel(pr,data,labels,headers):
    bounds = [pr.get_att1_start(),pr.get_att1_end(),\
              pr.get_att2_start(),pr.get_att2_end()]
    dims = [pr.get_att1(),pr.get_att2()]
    
    x = data[:,dims[0]]
    y = data[:,dims[1]]
    l = labels
    
    fig = plt.figure()
    ax=fig.add_subplot(111)
    ax.add_patch(matplotlib.patches.Rectangle((bounds[0],bounds[2]),bounds[1]-bounds[0],bounds[3]-bounds[2],fill=False))
    matplotlib.pyplot.scatter(x,y,s=5,c=l,cmap='gnuplot2',linewidths=.25)
    plt.colorbar()
    plt.xlabel(headers[dims[0]])
    plt.ylabel(headers[dims[1]])
    
def show_projection_binary(pr,data,labels,headers):
    bounds = [pr.get_att1_start(),pr.get_att1_end(),\
              pr.get_att2_start(),pr.get_att2_end()]
    dims = [pr.get_att1(),pr.get_att2()]
    classnum = pr.get_class()
    
    x = data[:,dims[0]]
    y = data[:,dims[1]]
    l = labels
    
    fig = plt.figure()
    ax=fig.add_subplot(111)
    ax.add_patch(matplotlib.patches.Rectangle((bounds[0],bounds[2]),bounds[1]-bounds[0],bounds[3]-bounds[2],fill=False))
    use_colors={0: "r",1:"b"}
    matplotlib.pyplot.scatter(x,y,s=5,c=[use_colors[i] for i in l],linewidths=0)
    plt.xlabel(headers[dims[0]])
    plt.ylabel(headers[dims[1]])
