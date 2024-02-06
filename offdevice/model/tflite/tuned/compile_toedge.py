import os, subprocess, sys


compression_methods = ["minmax", "msb", "smart", "root"]


for sub in range(3):
    for sess in range(2):
        for compression in compression_methods:
            for tuning_index in range(5):
                fine_tuning_range = range(tuning_index*2, tuning_index*2+2)
                subject = "00" + str(sub) if sub < 10 else "0" + str(sub)
                session = "00" + str(sess+1)
                filename = "emager_%s_%s_%s_tuned_%s_%s.tflite"%(subject, session, compression, fine_tuning_range[0], fine_tuning_range[-1])
                command = "edgetpu_compiler -m 13 %s"%(filename)
                subprocess.run(command, shell = True, executable="/bin/bash")

