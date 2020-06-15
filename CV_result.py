from os import listdir
from os.path import isfile, join
import os
import csv
import numpy as np
import pandas as pd
DIR_SAVE = os.path.join(os.environ["HOME"], "Dropbox/moment_base_classfication/moment_DRO_results_full")
mypath = DIR_SAVE
data_dir = [dir for dir in listdir(mypath) if not isfile(join(mypath, dir))]
#print(data_dir)
#method = "fr"
method = "kl_new"
Output_file = join(DIR_SAVE,method+'_output.csv')
F= open(Output_file,'w')
for dir in data_dir:
    MAX = 0
    max_file = ''
    max_array =[]
    full_dir = join(DIR_SAVE,dir)
    for f in listdir(full_dir):
        if f.find(".csv") == -1 or f.find(method) == -1 or f == method + ".csv":
            continue
        df=pd.read_csv(join(full_dir, f), sep=',',header=None)
        mean_array = np.mean(df,axis=0)

        if mean_array[1]>MAX:
            max_file = f
            max_array =mean_array
            MAX = mean_array[1]
    F.write(dir)
    F.write(" ,"+max_file)
    F.writelines([", %0.2f" % item  for item in max_array])
    F.write("\n")
F.close()
#np.savetxt(Output_file, np.array(output_txt), fmt="%c", delimiter=",")
