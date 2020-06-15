#kl_new
from os import listdir
from os.path import isfile, join
import os

mypath = 'datasets'
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
print(files)
rho_list = [0.005,0.01,0.02,0.03,0.05,0.1,0.15,0.16,0.2,0.5]
cv = 5
repeat_time = 10;
for f in files:
    for rho1 in rho_list:
        for rho2 in rho_list:
            commandline = 'python tester.py --method "kl_new" --cv ' + \
            str(cv) + ' --repeat ' + str(repeat_time) + ' --dataset "./datasets/' + \
            f + '"  --rho ' +str(rho1)  + ' ' + str(rho2)
            print(commandline)
            os.system(commandline)
