from os import listdir
from os.path import isfile, join
from scipy.spatial import distance
import numpy as np
import numpy.matlib
import scipy.linalg
import cv2
#mypath_train=Directory containing Training images 
#samples=samples for each subjects keep filename as subjectnn
#subjects= Number of classes or persons 
#sz=2 element array of resolution at which images are to be resized
#evect= Number of eigenvectors to be selected in PCA space(should be greater than number of subjects) 
#Return Values:
#		 DATABASE=original read database from mypath_train as vectorized appended matrix with size 512*512 of every column
# 		 P=Projection matrix of PCA
#  		 W=Projection matrix of LDA
#		 DATABASE_PCA=The database on PCA projected space(To compute distance and classify)
#		 DATABASE_LDA=The database on LDA projected space(To compute distance and classify)   
def train(mypath_train,samples,subjects,sz,evect):
	szr=sz[0];
	szc=sz[1];
	A=numpy.empty((szr*szc,0));
	sz=0;
	onlyfiles = [ f for f in listdir(mypath_train) if isfile(join(mypath_train,f)) ]
	onlyfiles = np.sort(onlyfiles)
	images = numpy.empty(len(onlyfiles), dtype=object)
	for n in range(0, len(onlyfiles)):
  	  images[n] = cv2.imread( join(mypath_train,onlyfiles[n]) )
  	  if images[n] is not None:
   	    small=cv2.resize(cv2.cvtColor(images[n],cv2.COLOR_BGR2GRAY),(szr,szc),interpolation=cv2.INTER_CUBIC);
   	    sm=np.double(numpy.array(small).ravel());
   	    A=numpy.c_[A,sm];
	DATABASE=A;
	mu=A.mean(1);
	A=((A.transpose())-(mu)).transpose()
	L=np.dot(np.double(A.transpose()),np.double(A))
	D,V=scipy.linalg.eig(L);
	D=np.real(D);
	V=np.real(V);
	sort_perm = D.argsort()[::-1]
	V = V[:,sort_perm]
	thr=0.9
	sm=sum(D)
	for j in xrange(0,D.size):
	  tmpsm=sum(D[0:j])
	  if(tmpsm/sm>=thr):
  	    break;
	evect=j+1;
	print(evect)
	V=V[:,0:evect]
	P=np.dot(np.double(A),V)
	a=np.dot(P.transpose(),A)
	Sw=0;
	mu_d=numpy.empty((a.shape[0],0));
	Sb=0;
	for i in xrange(0,a.shape[1],samples):
  	  mu_i=a[:,i:i+samples].mean(1);
  	  Sw=Sw+np.cov(a[:,i:i+samples]);  
  	  mu_d=np.c_[mu_d,mu_i];	
	Sb=np.cov(mu_d);
	D,W=scipy.linalg.eig(np.dot(scipy.linalg.inv(Sw),Sb));
	W=np.real(W);
	D=np.real(D);
	sort_perm = D.argsort()[::-1][:n]
	W = W[:,sort_perm]
	W=W[:,0:subjects-1]
	DATABASE_LDA=np.dot(W.transpose(),a).transpose();
        DATABASE_PCA=a.transpose();
	return DATABASE,P,W,DATABASE_PCA,DATABASE_LDA;

def classify_PCA_LDA(Image_mat,samples,subjects,sz,P,W,DATABASE_PCA,DATABASE_LDA,index,labels):
   	szr=sz[0];
	szc=sz[1];
	small=Image_mat;
	if len(Image_mat.shape) is not 2:
		small=cv2.resize(cv2.cvtColor(Image_mat,cv2.COLOR_BGR2GRAY),(szr,szc),interpolation=cv2.INTER_CUBIC);
	else:
		small=cv2.resize(Image_mat,(szr,szc),interpolation=cv2.INTER_CUBIC);
	sm=numpy.array(small).ravel();
   	sm=(sm)-(sm).mean();
   	Proj=np.dot(P.transpose(),np.double(sm));
  	Y=np.dot(W.transpose(),Proj);
  	dist=distance.cdist(DATABASE_LDA,np.array([Y]),'mahalanobis');
	idx=np.argmin(dist);
  	labels[index]=np.uint(np.ceil(idx/samples)+1)










