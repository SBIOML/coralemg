import os, subprocess, sys


compression_methods = ["minmax", "msb", "smart", "root"]


for i in range(3):
    for j in range(2):
        for compression in compression_methods:
            subject = "00" + str(i) if i < 10 else "0" + str(i)
            session = "00" + str(j+1)
            filename = "emager_%s_%s_%s.tflite"%(subject, session, compression)
            command = "edgetpu_compiler -m 13 %s"%(filename)
            subprocess.run(command, shell = True, executable="/bin/bash")

