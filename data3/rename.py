import os
import string


for filename in os.listdir("."):
    if filename.endswith(".dat"):
	#print(filename)
	str = filename.replace("-0", "-")
	os.rename(filename, str)






