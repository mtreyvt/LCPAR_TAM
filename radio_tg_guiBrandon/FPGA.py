#------------------------------------------------------------------------------
#'FPGAs.py'                                LCPART VT-ECE
#                                                      8Feb24
# Desc. of file here... 
#------------------------------------------------------------------------------
# Imports here
#------------------------------------------------------------------------------
# Desc. of function here 
#------------------------------------------------------------------------------
import serial
import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

FACTOR_SIG = 100000
FACTOR_GATE = 100

#-----------------CHANGE GATE PROPERTIES BELOW--------------
# 1-> Sqaure Gate
# 2-> Hamming Gate
# 3-> Hann Gate
# 4-> Blackman Gate
# 5-> Kaiser Window
GATE_LENGTH = 8
KAISER_BETA = 2.5
#-----------------------------------------------------------

#Setting buadrate and opening serial to Zybo Z7 
baudrate = 115200
ser = serial.Serial(port='/dev/ttyUSB0',baudrate = baudrate)

def fpga_Setup(GATE_TYPE, GATE_LENGTH):

	#Creating Gate Signals
	squareSig = np.arange(GATE_LENGTH)
	for i in range(GATE_LENGTH):
    		np.put(squareSig, i, 1, mode='raise')
	hammSig = np.hamming(GATE_LENGTH)
	hannSig = np.hanning(GATE_LENGTH)
	blackSig = np.blackman(GATE_LENGTH)
	kaiserSig = np.kaiser(GATE_LENGTH, KAISER_BETA)

	if GATE_TYPE == 2:
	  gateSig = hammSig * FACTOR_GATE
	elif GATE_TYPE == 3:
	  gateSig = hannSig * FACTOR_GATE
	elif GATE_TYPE == 4:
	  gateSig = blackSig * FACTOR_GATE
	elif GATE_TYPE == 5:
	  gateSig = kaiserSig * FACTOR_GATE
	else:
	  gateSig = squareSig * FACTOR_GATE
	
	return gateSig

def fpga_TG(gateSig,data):
    N = int(len(data))
    
    sendSig = data;
    #sendSig = np.arange(N)
    
    #Steping through each data and sending it to Zybo Z7
    for i in range(N):
    	#np.put(sendSig, i, int(FACTOR_SIG*float(data)), mode='raise')
    	ser.write(int(sendSig[i]).to_bytes(2, 'big'))

#Sending Gate Signal
    gateSig = np.pad(gateSig, (0, N-GATE_LENGTH), 'constant')
    for i in range(N):
    	ser.write(int(gateSig[i]).to_bytes(2, 'big'))
    	print(gateSig[i], int(gateSig[i]).to_bytes(2, 'big'))

#Receving data from Zybo Z7
    recSig = np.arange(N)
    for i in range(N):
        np.put(recSig, i, int(ser.readline()), mode='raise')
        recSig = recSig/(FACTOR_SIG*FACTOR_GATE)

    return recSig
#--------------------

gateSig = fpga_Setup(2,8)

data = np.ones(256)
print (data)

recSig = fpga_TG(gateSig,data)
print(recSig)

#Plotting gated and non gated signals
#xpoints = np.arange(N)+1

#plt.plot(xpoints, sendSig/FACTOR_SIG, label = "Not Gated") 
#plt.plot(xpoints, recSig, label = "Gated") 
#plt.legend() 
#plt.show()

------------------------------------------------------EoF
