# IMPORTING REQUIRED LIBRARIES FOR IMAGE AND VIDEO MANIPUALTION
import numpy as np
import cv2
import sys
import time
import PCA_LDA_paral
import Track
from threading import Thread
my_path='/media/sharvil/00521D65521D60A8/IET/SEM6/ML_AOBD/Final/Mugshots_renamed'
samples=11;
subjects=5;
sz=np.array([512,512]);
evect=28;
DATABASE,P,W,DATABASE_PCA,DATABASE_LDA=PCA_LDA_paral.train(my_path,samples,subjects,sz,evect)
#cap = cv2.VideoCapture('File_001.mov')
cap = cv2.VideoCapture('IMG_1085.MOV')
#cap = cv2.VideoCapture(0)
names=np.array(['Rishabh','Sharvil','Pooja','Netra','Yesha']);
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
flag =0
start  = time.time()
det_frame=[];
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 );
flag_c=0;
cnt=0;
fps2=0;
det=0;
rec=0;
while(1):
	start2  = time.time()
	ret, frame = cap.read()  # READING FRAMES FROM CAM
    	if frame is None:
        	break;
    	frame=cv2.transpose(frame)
	    	
	if (flag%30==0):
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        	faces = faceCascade.detectMultiScale(
            	gray,
            	scaleFactor=2,
            	minNeighbors=5,
	    	#minSize=(100,100),
            	flags = cv2.CASCADE_SCALE_IMAGE
       		 )
		sub_arr=np.zeros(len(faces));
		rois=[];
		l=0;
		det=det+1;
		for (x,y,w,h) in faces:
	    		troi = frame[y:y+h, x:x+w]	
	    		tmp_roi = cv2.cvtColor(troi, cv2.COLOR_BGR2GRAY)   # CONEVRTING TO GRAYSCALE
			rois.append(tmp_roi)
	    		PCA_LDA_paral.classify_PCA_LDA(tmp_roi,samples,subjects,sz,P,W,DATABASE_PCA,DATABASE_LDA,l,sub_arr)
			rec=rec+(np.uint(sub_arr[0])==5 or np.uint(sub_arr[1])==5);
			l=l+1;
		det_frame=frame;
		flag_c=0; 
	if(flag%30!=0):
	    	j=0;
		f=0;
		
		if(flag_c==0):
			flag_c=1
		else:	
	    		for (x,y,w,h) in faces:
    	      			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
	      			cv2.putText(frame,str(names[np.uint8(sub_arr[f])-1]),(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
				f=f+1
		
    	flag = flag + 1
	#print(1/(end2-start2))
	if(flag%31==0):	
		fps2=(flag-1)/(cnt)	
	cv2.putText(frame,str(fps2),(20,20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
	cv2.putText(frame,'Detected:'+str(det),(60,60),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
	cv2.putText(frame,'Correctly Recognized:'+str(rec),(60,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
    	cv2.imshow('frame',frame)
    	if cv2.waitKey(1) & 0xFF == ord('f'):  # ENTERING ALPHABET FOR QUTTING / QUITTING MECHANISM         
		break
	end2  = time.time()
	cnt=cnt+end2-start2
		
# When everything is done, release the capture
end = time.time()
seconds = end - start
print('Number of detected faces:'+str(det));
print('Number of recognized faces:'+str(rec));
print "Time taken : {0} seconds".format(seconds)
print flag 
    # Calculate frames per second
fps  = (flag-1) / seconds;
print "Estimated frames per second : {0}".format(fps);

# RELEASE ALL THE RESOURCES AFTER COMPLETION
cap.release()
cv2.destroyAllWindows()
