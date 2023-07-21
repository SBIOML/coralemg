"""
`intan_rhd2164`
===========================

This is a CircuitPython driver for the Intan RHD2164 Digital Electrophysiology Interface Chip 

* Author(s): Etienne Buteau

Implementation Notes
--------------------

**Hardware:**

* Intan RHD2164

**Software and Dependencies:**

* Adafruit CircuitPython firmware for the supported boards:
  https://circuitpython.org/downloads

* Adafruit's Bus Device library: https://github.com/adafruit/Adafruit_CircuitPython_BusDevice

"""

import math
import struct
import time

from adafruit_bus_device.spi_device import SPIDevice
from micropython import const


# Register and other constant values:
# Register 0 - ADC config.
_RHD2164_ADC_CONFIG_WRITE = const(0xC000)
_RHD2164_ADC_CONFIG_READ = const(0xF000)
# Register 1 - Supply sensor & ADC buffer bias current
_RHD2164_SUPPLY_SENSOR_WRITE = const(0xC003)
_RHD2164_SUPPLY_SENSOR_READ = const(0xF003)
# Register 2 - MUX bias current
_RHD2164_MUX_BIAS_WRITE = const(0xC00C)
_RHD2164_MUX_BIAS_READ = const(0xF00C)
# Register 3 - MUX Load, Temp sensor, Aux digital output
_RHD2164_MUX_LOAD_WRITE = const(0xC00F)
_RHD2164_MUX_LOAD_READ = const(0xF00F)
# Register 4 - ADC output format & DSP offset removal
_RHD2164_ADC_FORMAT_WRITE = const(0xC030)
_RHD2164_ADC_FORMAT_READ = const(0xF030)
# Register 5 - Impedance check control
_RHD2164_IMP_CTRL_WRITE = const(0xC033)
_RHD2164_IMP_CTRL_READ = const(0xF033)
# Register 6 - Impedance check DAC [unchanged]
_RHD2164_IMP_DAC_WRITE = const(0xC03C)
_RHD2164_IMP_DAC_READ = const(0xF03C)
# Register 7 - Impedance check amplifier select [unchanged]
_RHD2164_IMP_AMP_WRITE = const(0xC03F)
_RHD2164_IMP_AMP_READ = const(0xF03F)
# Register 8-13 - On-chip amplifier bandwidth select
_RHD2164_AMP_BAN_0_WRITE = const(0xC0C0) # 8
_RHD2164_AMP_BAN_1_WRITE = const(0xC0C3) # 9
_RHD2164_AMP_BAN_2_WRITE = const(0xC0CC) # 10
_RHD2164_AMP_BAN_3_WRITE = const(0xC0CF) # 11
_RHD2164_AMP_BAN_4_WRITE = const(0xC0F0) # 12
_RHD2164_AMP_BAN_5_WRITE = const(0xC0F3) # 13

_RHD2164_AMP_BAN_0_READ = const(0xF0C0) # 8
_RHD2164_AMP_BAN_1_READ = const(0xF0C3) # 9
_RHD2164_AMP_BAN_2_READ = const(0xF0CC) # 10
_RHD2164_AMP_BAN_3_READ = const(0xF0CF) # 11
_RHD2164_AMP_BAN_4_READ = const(0xF0F0) # 12
_RHD2164_AMP_BAN_5_READ = const(0xF0F3) # 13
# Register 14-21 - Individual amplifier power
_RHD2164_AMP_POW_0_WRITE = const(0xC0FC) # 14
_RHD2164_AMP_POW_1_WRITE = const(0xC0FF) # 15
_RHD2164_AMP_POW_2_WRITE = const(0xC300) # 16
_RHD2164_AMP_POW_3_WRITE = const(0xC303) # 17
_RHD2164_AMP_POW_4_WRITE = const(0xC30C) # 18
_RHD2164_AMP_POW_5_WRITE = const(0xC30F) # 19
_RHD2164_AMP_POW_6_WRITE = const(0xC330) # 20
_RHD2164_AMP_POW_7_WRITE = const(0xC333) # 21

_RHD2164_AMP_POW_0_READ = const(0xF0FC) # 14
_RHD2164_AMP_POW_1_READ = const(0xF0FF) # 15
_RHD2164_AMP_POW_2_READ = const(0xF300) # 16
_RHD2164_AMP_POW_3_READ = const(0xF303) # 17
_RHD2164_AMP_POW_4_READ = const(0xF30C) # 18
_RHD2164_AMP_POW_5_READ = const(0xF30F) # 19
_RHD2164_AMP_POW_6_READ = const(0xF330) # 20
_RHD2164_AMP_POW_7_READ = const(0xF333) # 21

# ADC Calibration
_RHD2164_ADC_CAL = const(0x3333)

def misosplit():
  pass

class RHD2164:
    """
    Driver for the Intan RHD2164 Digital Electrophysiology Interface Chip .

    :param ~busio.SPI spi: The SPI bus the RHD2164 is connected to.
    :param ~microcontroller.Pin cs: The pin used for the CS signal.
    """

    def __init__(self, spi, cs, baudrate):
        self.spi_device = SPIDevice(spi, cs, baudrate, polarity=0, phase=0)
        self._WRT_BUFFER = bytearray(4)
        self._RD_BUFFER = bytearray(4)

    def _read_register(self, address):
      self._fill_write_buffer(self._WRT_BUFFER, address, 0x0000)
      with self.spi_device as device :
        device.write_readinto(self._WRT_BUFFER, self._RD_BUFFER)
    
    def _write_register(self, address, value):
      self._fill_write_buffer(self._WRT_BUFFER, address, value)
      with self.spi_device as device:
        device.write(self._WRT_BUFFER)

    def _fill_write_buffer(self, buffer, address, value):
      buffer[0] = address >> 8 
      buffer[1] = address & 0x00FF
      buffer[2] = value >> 8
      buffer[3] = value & 0x00FF
    
    def config_intan(self):
      # Register 0 - ADC config.
      self._write_register(_RHD2164_ADC_CONFIG_WRITE, 0xF3FC)
      # Register 1 - Supply sensor & ADC buffer bias current
      self._write_register(_RHD2164_SUPPLY_SENSOR_WRITE, 0x0C00)
      # Register 2 - MUX bias current
      self._write_register(_RHD2164_MUX_BIAS_WRITE, 0x0CC0)
      # Register 3 - MUX Load, Temp sensor, Aux digital output
      self._write_register(_RHD2164_MUX_LOAD_WRITE, 0x000C)
      # Register 4 - ADC output format & DSP offset removal
      self._write_register(_RHD2164_ADC_FORMAT_WRITE, 0xF000)
      # Register 5 - Impedance check control
      self._write_register(_RHD2164_IMP_CTRL_WRITE, 0x0000)
      # Register 6 - Impedance check DAC
      self._write_register(_RHD2164_IMP_DAC_WRITE, 0x0000)
      # Register 7 - Impedance check amplifier select
      self._write_register(_RHD2164_IMP_AMP_WRITE, 0x0000)

      # Register 8-13 - On-chip amplifier bandwidth select
      self._write_register(_RHD2164_AMP_BAN_0_WRITE, 0x003C)
      self._write_register(_RHD2164_AMP_BAN_1_WRITE, 0x00C3)
      self._write_register(_RHD2164_AMP_BAN_2_WRITE, 0x000C)
      self._write_register(_RHD2164_AMP_BAN_3_WRITE, 0x00CF)
      self._write_register(_RHD2164_AMP_BAN_4_WRITE, 0x0F3C)
      self._write_register(_RHD2164_AMP_BAN_5_WRITE, 0x0000)

      # Register 14-21 - Individual amplifier power
      self._write_register(_RHD2164_AMP_POW_0_WRITE, 0xFFFF)
      self._write_register(_RHD2164_AMP_POW_1_WRITE, 0xFFFF)
      self._write_register(_RHD2164_AMP_POW_2_WRITE, 0xFFFF)
      self._write_register(_RHD2164_AMP_POW_3_WRITE, 0xFFFF)
      self._write_register(_RHD2164_AMP_POW_4_WRITE, 0xFFFF)
      self._write_register(_RHD2164_AMP_POW_5_WRITE, 0xFFFF)
      self._write_register(_RHD2164_AMP_POW_6_WRITE, 0xFFFF)
      self._write_register(_RHD2164_AMP_POW_7_WRITE, 0xFFFF)

      # Calibrate ADC
      time.sleep(0.0001)
      self._write_register(_RHD2164_ADC_CAL, 0x0000)
    
    def read_intan(self):
      # Register 0 - ADC config.
      self._read_register(_RHD2164_ADC_CONFIG_READ)
      # Register 1 - Supply sensor & ADC buffer bias current
      self._read_register(_RHD2164_SUPPLY_SENSOR_READ)
      # Register 2 - MUX bias current
      self._read_register(_RHD2164_MUX_BIAS_READ)
      # Register 3 - MUX Load, Temp sensor, Aux digital output
      self._read_register(_RHD2164_MUX_LOAD_READ)
      # Register 4 - ADC output format & DSP offset removal
      self._read_register(_RHD2164_ADC_FORMAT_READ)
      # Register 5 - Impedance check control
      self._read_register(_RHD2164_IMP_CTRL_READ)
      # Register 6 - Impedance check DAC
      self._read_register(_RHD2164_IMP_DAC_READ)
      # Register 7 - Impedance check amplifier select
      self._read_register(_RHD2164_IMP_AMP_READ)
      # Register 8-13 - On-chip amplifier bandwidth select
      self._read_register(_RHD2164_AMP_BAN_0_READ)
      self._read_register(_RHD2164_AMP_BAN_1_READ)
      self._read_register(_RHD2164_AMP_BAN_2_READ)
      self._read_register(_RHD2164_AMP_BAN_3_READ)
      self._read_register(_RHD2164_AMP_BAN_4_READ)
      self._read_register(_RHD2164_AMP_BAN_5_READ)

      # Register 14-21 - Individual amplifier power
      self._read_register(_RHD2164_AMP_POW_0_READ)
      self._read_register(_RHD2164_AMP_POW_1_READ)
      self._read_register(_RHD2164_AMP_POW_2_READ)
      self._read_register(_RHD2164_AMP_POW_3_READ)
      self._read_register(_RHD2164_AMP_POW_4_READ)
      self._read_register(_RHD2164_AMP_POW_5_READ)
      self._read_register(_RHD2164_AMP_POW_6_READ)
      self._read_register(_RHD2164_AMP_POW_7_READ)









#print(hex(_RHD2164_ADC_FORMAT_WRITE & 0x00FF))

#WRT_BUFFER = bytearray(4)
#print(len(WRT_BUFFER))