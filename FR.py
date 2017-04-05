# IMPORTING REQUIRED LIBRARIES FOR IMAGE AND VIDEO MANIPUALTION
import numpy as np
import cv2
import sys
import time
import PCA_LDA_func
import Track
import os
#fname1 = "PCA.txt"
#fname1 = "LDA.txt"
my_path='/media/sharvil/00521D65521D60A8/IET/SEM6/ML_AOBD/Final/Mugshots_renamed'
samples=11;
subjects=5;
sz=np.array([300,300]);
evect=28;
DATABASE,P,W,DATABASE_PCA,DATABASE_LDA=PCA_LDA_func.train(my_path,samples,subjects,sz,evect)
#cap = cv2.VideoCapture('File_000.mov')	#Netra and rishbh
#cap = cv2.VideoCapture('IMG_6485.MOV')	#All 5
#cap = cv2.VideoCapture('IMG_1085.MOV')	#Yesha Did Benchmark
cap = cv2.VideoCapture(0)		#Webcam
names=np.array(['Rishabh','Sharvil','Pooja','Netra','Yesha']);
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
flag =0
start  = time.time()
det_frame=[];
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 );
flag_c=0;
while(1):
	ret, frame = cap.read()  # READING FRAMES FROM CAM
    	if frame is None:
        	break;
    	#frame=cv2.transpose(frame) #For yesha didi's video
    	if (flag%30==0):
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        	faces = faceCascade.detectMultiScale(
            	gray,
            	scaleFactor=1.5,
            	minNeighbors=3,
	    	minSize=(60,60),
            	flags = cv2.CASCADE_SCALE_IMAGE
       		 )
		sub_arr=np.empty((1,0));
		tw=faces;
		rois=[];
		l=0;
		for (x,y,w,h) in faces:
	    		troi = frame[y:y+h, x:x+w]			
	    		tmp_roi = cv2.cvtColor(troi, cv2.COLOR_BGR2GRAY)   # CONEVRTING TO GRAYSCALE
	    		#cv2.imshow('win'+str(l),tmp_roi)
			l=l+1;
			rois.append(tmp_roi)
	    		idx=PCA_LDA_func.classify_PCA_LDA(tmp_roi,samples,subjects,sz,P,W,DATABASE_PCA,DATABASE_LDA)
	    		sub=np.uint(np.ceil(idx/samples)+1);
	    		sub_arr=np.c_[sub_arr,sub];
		det_frame=frame;
		flag_c=0;    
	if(flag%30!=0):
	    	j=0;
		f=0;
		#hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		
		if(flag_c==0):
			'''roi_hists=[];
			for (x,y,w,h) in faces:	    	      
				roi = det_frame[y:y+h, x:x+w]
				hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
				mask = cv2.inRange(hsv_roi, np.array((0.,10,60)), np.array((20,150,255.)))
				roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
				cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
				roi_hists.append(roi_hist)'''
			flag_c=1
		else:	
	    		for (x,y,w,h) in faces:
	       			'''roi = det_frame[y:y+h, x:x+w]
	      			hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
   	      			mask = cv2.inRange(hsv_roi, np.array((0.,0.18*255,0)), np.array((38,0.68*255,255.)))
   	      			roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
   	      			cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
	      			abc=tuple(map(tuple, tw))
	      			ret,tmp_tw=Track.Tracking(hsv,roi_hists[j],abc[j],term_crit);
	      			tw[j]=np.array(tmp_tw);
	      			j=j+1;
	      			box = cv2.boxPoints(ret)
	      			box = np.int0(box)
	      			a,b,c,d = cv2.boundingRect(box)'''
    	      			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
	      			cv2.putText(frame,str(names[np.uint8(sub_arr[0,f])-1]),(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
				f=f+1
    	flag = flag + 1
    	cv2.imshow('frame',frame)
    	if cv2.waitKey(1) & 0xFF == ord('f'):  # ENTERING ALPHABET FOR QUTTING / QUITTING MECHANISM         
		break

# When everything is done, release the capture
end = time.time()
seconds = end - start
print "Time taken : {0} seconds".format(seconds)
print flag
 
    # Calculate frames per second
fps  = (flag-1) / seconds;
print "Estimated frames per second : {0}".format(fps);

# RELEASE ALL THE RESOURCES AFTER COMPLETION
cap.release()
cv2.destroyAllWindows()
