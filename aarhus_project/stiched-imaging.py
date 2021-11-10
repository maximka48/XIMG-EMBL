import tine
import atexit
import logging
import time
logging.basicConfig(format='%(asctime)s %(message)s')

LINKLIST = []

def all_detach():
   global LINKLIST
   for x in LINKLIST:
       #print "detaching ",x 
       tine.detach(x)
atexit.register(all_detach)

MDSTATUS = None
COLSTATUS = None

COLERROR = False

def cb_md_status(a,b,c):
    global MDSTATUS
    MDSTATUS =  c
tine.update("/P14/MD3/MD3_0","Status",cb_md_status,10) 

def cb_col_status(a,b,c):
    global COLSTATUS
    COLSTATUS = c
    print c
tine.update("/p14/collection/mx-standard","status",cb_col_status,10)

def wait_md_ready():
    time.sleep(1)
    global MDSTATUS
    while MDSTATUS != "Ready":
       time.sleep(0.1)

def wait_col_ready():
    time.sleep(1)
    global COLSTATUS
    global COLERROR
    while COLSTATUS != "ready":
       if COLSTATUS == "error":
          COLERROR = True
          return
       time.sleep(0.1)
    COLERROR = False

n = 85

wait_col_ready()


tine.set("/p14/collection/mx-standard","ff-pre",True)

tine.set("/p14/collection/mx-standard","directory","/mnt/beegfs/P14/2020/p3l-gleb1/20201129/RAW_DATA/imaging")

tine.set("/p14/collection/mx-standard","start-angle",2.21)
tine.set("/p14/collection/mx-standard","range",0.1)
tine.set("/p14/collection/mx-standard","exposure-time",0.01)

tine.set("/P14/pco-camera/pco-camera","write-data",True)

#for y in [ -5.25, -5.0, -4.75, -4.5, -4.25, -4.0, -3.75  ]:
# ^ this is full range of motor position in vertical direction
for y in [ -4.0, -3.75  ]:
    tine.set("P14/MD3/MD3_0","AlignmentYPosition",y)
    wait_md_ready()
    for z in (-2.194, -1.944, -1.694,-1.444,-1.194, -0.944, -0.694, -0.444, -0.194, 0.056, 0.306, 0.556, 0.806, 1.056, 1.306, 1.556, 1.806):
       # motor positions in hor directions
       time.sleep(0.1)
       tine.set("P14/MD3/MD3_0","AlignmentZPosition",z)
       wait_md_ready()
       print "got y ", tine.get("P14/MD3/MD3_0","AlignmentZPosition")
       n = n + 1
       COLERROR = True
       while COLERROR == True:
           tine.set("/p14/collection/mx-standard","ff-num-images",30)
           tine.set("/p14/collection/mx-standard","num-images",3600)
           tine.set("/p14/collection/mx-standard","ff-offset",[0,-5.95-y,0])
		# this line below corresponds to the filename
           tine.set("/p14/collection/mx-standard","template","try0_full_135mm_%s_"%n+'%5d.tiff')
		
           tine.set("/p14/collection/mx-standard","collect")
           time.sleep(1)
           wait_col_ready()
           print y, tine.get("P14/MD3/MD3_0","AlignmentYPosition"), z, tine.get("P14/MD3/MD3_0","AlignmentZPosition")


# the whole logic is
# 7x17 grid position (17 in hor, 7 in ver)
# 1) set the distance
# 2) set starting angle of rotation
# 3) set y coordinate (vertical)
# 4) set z coordinate (horizontal)
# 5) do flatfield, moved by 1 axis only (vertical, by 6mm from 5.95mm up)
# 6) acquire 3600 frames (rotation, tomo)
# 7) repeat 4-6 for next horizontal grid position 
# 8) repeat 7 for next verical grid position
# 9) repeat 1-8 for another distance (4 in total)

# Gleb's advice
# stitch files before doing anything -> reconstruct as normal
# matching can be done 1st by considering motor positions
# sample moves down, camera moving up
# check left-right
