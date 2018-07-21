import string
import os
import numpy as np

num = 0

for s1z in np.arange(-.96, .98, .02):
    count = 0
    num = 0
    s1z_ = float("{0:.3f}".format(s1z))
    #print(s1z_)

    str_ = "q_3.00_s1z_" + str(s1z_)
    str_ = str_.replace("-0", "-")

    if int(s1z*100) % 10 == 0 or (int(s1z*-100) + 1) % 10 == 0:
	str_ = str_ + "0"
	print("\n")
	
    for filename in os.listdir("."):
	#num = num + 1    
	if filename.endswith(".dat"):
	    num = num + 1
	    if filename.startswith(str_):
		count = count + 1

    print(str_ + ": " + str(count))
    #num = num + 1

print(num)
