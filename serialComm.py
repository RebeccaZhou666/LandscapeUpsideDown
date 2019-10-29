
# import serial

# arduino = serial.Serial('/dev/tty.usbmodem144301', 9600, timeout=.1)
# while True:
# 	data = arduino.readline()[:-2] #the last bit gets rid of the new-line chars
# 	if data:
# 		print (data)



import serial, time
arduino = serial.Serial('/dev/tty.usbmodem144301', 9600, timeout=.1)
time.sleep(1) #give the connection a second to settle
arduino.write('50'.encode('UTF-8'))
while True:
	data = arduino.readline()
	print (data)
	# if data:
	# 	b = bytes(data.rstrip('\n'), 'UTF-8')
	# 	print(b)#strip out the new lines for now
		# (better to do .read() in the long run for this reason