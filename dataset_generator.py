import random as rm

N = 4
f = open("dataset.txt", "a")
for i in range(100):
    st1 = round(rm.uniform(-100, 100),N)
    st2 = round(rm.uniform(-100, 100),N)
    st3 = round(rm.uniform(-100, 100),N)
    string = str(st1) + " " + str(st2) + " " + str(st3) + "\n"
    
    f.write(string)

f.close()
