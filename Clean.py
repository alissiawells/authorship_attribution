# -*- coding:utf-8 -*-  
import os  
import random
import re

def clean():
    mergefiledir = os.getcwd()+'/'+'ebd_data'+'/'+'embedding'
    filenames = os.listdir(mergefiledir)  

    if os.path.isdir(mergefiledir):
	    if not os.path.isdir(os.path.join(os.getcwd(), 'clean_embedding')):
		    os.mkdir(os.path.join(os.getcwd(), 'clean_embedding'))

    for filename in filenames:
	    output_path = os.getcwd()+'/'+'clean_embedding'+'/'+filename
	    output = open( output_path,'w')  
	    filepath = mergefiledir+'/'+filename  
	    for line in open(filepath):  	
		    #print(line)
		    m = re.match('(.+).txt\$\$(.+)', line)
		    author = m.group(1)
		    output.write(author)
		    vectors = m.group(2)
		    vector_list = re.split(' ',vectors)
		    for i,v in enumerate(vector_list):
			    output.write(' '+str(i+1)+':'+v)
		    output.write('\n')				
		
	    output.close()  
