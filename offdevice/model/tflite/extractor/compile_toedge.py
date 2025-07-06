import os, subprocess, sys


compression_methods = ["minmax", "msb", "smart", "root"]
bits = [1,2,3,4,5,6,7,8]
model_type = "cnn"
dataset_name = "capgmyo"

for sub in range(1,11):
    for sess in range(2):
        for compression in compression_methods:
            for bit in bits:
                subject = "0" + str(sub) if sub < 10 else str(sub)
                session = str(sess+1)
                filename = "%s_%s_%s_%s_%s_%sbits_ondevice.tflite"%(dataset_name, model_type, subject, session, compression, bit)
                command = "edgetpu_compiler -m 13 %s"%(filename)
                subprocess.run(command, shell = True, executable="/bin/bash")
