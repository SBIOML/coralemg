import board
import gpiod
import busio
import board
import lcd_driver as lcd_lib
import rhd2164 as rhd
import time
import serial
import collections
import numpy as np
import struct

ones_mask = np.ones(128, dtype=np.uint8)
mask = np.array([0, 2] + [0, 1] * 63, dtype=np.uint8)
channel_map = [10, 22, 12, 24, 13, 26, 7, 28, 1, 30, 59, 32, 53, 34, 48, 36] + \
            [62, 16, 14, 21, 11, 27, 5, 33, 63, 39, 57, 45, 51, 44, 50, 40] + \
            [8, 18, 15, 19, 9, 25, 3, 31, 61, 37, 55, 43, 49, 46, 52, 38] + \
            [6, 20, 4, 17, 2, 23, 0, 29, 60, 35, 58, 41, 56, 47, 54, 42]

def reorder(data, mask, match_result):
    '''
    Looks for mask/template matching in data array and reorders
    :param data: (numpy array) - 1D data input
    :param mask: (numpy array) - 1D mask to be matched
    :param match_result: (int) - Expected result of mask-data convolution matching
    :return: (numpy array) - Reordered data array
    '''
    number_of_packet = int(len(data)/128)
    #print("Number of packets : %s"%(number_of_packet))
    roll_data = np.empty((number_of_packet, 128), dtype=np.uint8)
    for i in range(number_of_packet):
        data_slice = data[i*128:(i+1)*128]
        if i == 0:
            data_lsb = np.bitwise_and(data_slice, ones_mask, dtype=np.uint8)
            mask_match = np.convolve(mask, np.append(data_lsb, data_lsb), 'valid')
            try:
                offset = np.where(mask_match == match_result)[0][0] - 1
            except IndexError:
                return None
        roll_data[i,:] = np.roll(data_slice, -offset)
    return roll_data

def initialize_lcd():
    i2c = busio.I2C(board.I2C1_SCL, board.I2C1_SDA)
    while i2c.try_lock():
        pass
    lcd = lcd_lib.I2C_LCD(i2c, lcd_lib.DEFAULT_I2C_ADDR, 2, 16)

    lcd.backlight_on()
    lcd.display_on()
    lcd.clear()

    return lcd

def initialize_button():
    chip = gpiod.chip("0", gpiod.chip.OPEN_BY_NUMBER)
    button = chip.get_line(22)
    config = gpiod.line_request()
    config.request_type = gpiod.line_request.DIRECTION_INPUT
    button.request(config)
    #button.get_value()
    return button

def initialize_led():
    chip = gpiod.chip("0", gpiod.chip.OPEN_BY_NUMBER)
    red_led = chip.get_line(45)
    green_led = chip.get_line(13)
    config = gpiod.line_request()
    config.request_type = gpiod.line_request.DIRECTION_OUTPUT
    red_led.request(config)
    green_led.request(config)

    return red_led, green_led

def main():
    ser = serial.Serial('/dev/ttyS0', 1500000, timeout=1)
    ser.close()
    ser.open()
    ser.reset_input_buffer()
    for i in range(1000):
        bytes_available = ser.inWaiting()
        bytesToRead = bytes_available - (bytes_available % 128)
        if bytesToRead > 0:
            curr_time = time.time()
            raw_data_packet = ser.read(bytesToRead)
            data_packet = np.array(list(raw_data_packet), dtype=np.uint8)
            reordered_packet = reorder(data_packet, mask, 63)
            processed_packets = []
            fmt = '>64h'
            if reordered_packet is not None:
                for packet in reordered_packet:
                    samples = np.array(struct.unpack(fmt, packet), dtype=np.int16)
                    processed_packets.append(samples[channel_map])
            print(time.time()- curr_time)
    ser.close()

    # sensor = HDSensor('/dev/ttyS0', 1500000)
    # firstGo = True
    # data = sensor.live_read(firstTime=firstGo)  # sensor.sample()
    # firstGo = False
    # i = 0
    # while(i < 10):
    #     curr_time = time.time()
    #     sensor.live_read(firstTime=firstGo)
    #     print(time.time()-curr_time)
    #     i += 1
    # SPIDevice = "/dev/spidev0.0"
    # rhd0 = rhd.RHD2164_DRIVER(SPIDevice,rhd.BAUDRATE, debug=False)
    # #rhd0.calibrate_intan()

    # rhd0.read_I_of_intant()
    # rhd0.read_I_of_intant()
    # rhd0.read_I_of_intant()

    # with Profile() as profile:
    #     print(f"{rhd0.read_all_adc_intant() = }")
    #     (
    #         Stats(profile)
    #         .strip_dirs()
    #         .sort_stats(SortKey.CALLS)
    #         .print_stats()
    #     )


if __name__ == "__main__":
    main()
