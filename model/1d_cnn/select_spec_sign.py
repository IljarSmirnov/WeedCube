import glob
import re
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import csv
import pandas as pd
from skimage.color import rgb2gray

directory_path = "G:\Мой диск\hsi data\canola"
# Create a list of .npy files in the directory
npy_files = glob.glob(directory_path + '/*.npy')

# Function to extract numbers from the file name
def extract_number(f):
    s = re.findall("\d+",f)
    return (int(s[0]) if s else -1,f)

# Sort the list of .npy files in numerical order
npy_files = sorted(npy_files, key=extract_number)

nm_mapping = {397.66: 0.0, 400.28: 1.0, 402.9: 2.0, 405.52: 3.0, 408.13: 4.0, 410.75: 5.0, 413.37: 6.0, 416.0: 7.0, 418.62: 8.0, 421.24: 9.0, 423.86: 10.0, 426.49: 11.0, 429.12: 12.0, 431.74: 13.0, 434.37: 14.0, 437.0: 15.0, 439.63: 16.0, 442.26: 17.0, 444.89: 18.0, 447.52: 19.0, 450.16: 20.0, 452.79: 21.0, 455.43: 22.0, 458.06: 23.0, 460.7: 24.0, 463.34: 25.0, 465.98: 26.0, 468.62: 27.0, 471.26: 28.0, 473.9: 29.0, 476.54: 30.0, 479.18: 31.0, 481.83: 32.0, 484.47: 33.0, 487.12: 34.0, 489.77: 35.0, 492.42: 36.0, 495.07: 37.0, 497.72: 38.0, 500.37: 39.0, 503.02: 40.0, 505.67: 41.0, 508.32: 42.0, 510.98: 43.0, 513.63: 44.0, 516.29: 45.0, 518.95: 46.0, 521.61: 47.0, 524.27: 48.0, 526.93: 49.0, 529.59: 50.0, 532.25: 51.0, 534.91: 52.0, 537.57: 53.0, 540.24: 54.0, 542.91: 55.0, 545.57: 56.0, 548.24: 57.0, 550.91: 58.0, 553.58: 59.0, 556.25: 60.0, 558.92: 61.0, 561.59: 62.0, 564.26: 63.0, 566.94: 64.0, 569.61: 65.0, 572.29: 66.0, 574.96: 67.0, 577.64: 68.0, 580.32: 69.0, 583.0: 70.0, 585.68: 71.0, 588.36: 72.0, 591.04: 73.0, 593.73: 74.0, 596.41: 75.0, 599.1: 76.0, 601.78: 77.0, 604.47: 78.0, 607.16: 79.0, 609.85: 80.0, 612.53: 81.0, 615.23: 82.0, 617.92: 83.0, 620.61: 84.0, 623.3: 85.0, 626.0: 86.0, 628.69: 87.0, 631.39: 88.0, 634.08: 89.0, 636.78: 90.0, 639.48: 91.0, 642.18: 92.0, 644.88: 93.0, 647.58: 94.0, 650.29: 95.0, 652.99: 96.0, 655.69: 97.0, 658.4: 98.0, 661.1: 99.0, 663.81: 100.0, 666.52: 101.0, 669.23: 102.0, 671.94: 103.0, 674.65: 104.0, 677.36: 105.0, 680.07: 106.0, 682.79: 107.0, 685.5: 108.0, 688.22: 109.0, 690.93: 110.0, 693.65: 111.0, 696.37: 112.0, 699.09: 113.0, 701.81: 114.0, 704.53: 115.0, 707.25: 116.0, 709.97: 117.0, 712.7: 118.0, 715.42: 119.0, 718.15: 120.0, 720.87: 121.0, 723.6: 122.0, 726.33: 123.0, 729.06: 124.0, 731.79: 125.0, 734.52: 126.0, 737.25: 127.0, 739.98: 128.0, 742.72: 129.0, 745.45: 130.0, 748.19: 131.0, 750.93: 132.0, 753.66: 133.0, 756.4: 134.0, 759.14: 135.0, 761.88: 136.0, 764.62: 137.0, 767.36: 138.0, 770.11: 139.0, 772.85: 140.0, 775.6: 141.0, 778.34: 142.0, 781.09: 143.0, 783.84: 144.0, 786.58: 145.0, 789.33: 146.0, 792.08: 147.0, 794.84: 148.0, 797.59: 149.0, 800.34: 150.0, 803.1: 151.0, 805.85: 152.0, 808.61: 153.0, 811.36: 154.0, 814.12: 155.0, 816.88: 156.0, 819.64: 157.0, 822.4: 158.0, 825.16: 159.0, 827.92: 160.0, 830.69: 161.0, 833.45: 162.0, 836.22: 163.0, 838.98: 164.0, 841.75: 165.0, 844.52: 166.0, 847.29: 167.0, 850.06: 168.0, 852.83: 169.0, 855.6: 170.0, 858.37: 171.0, 861.14: 172.0, 863.92: 173.0, 866.69: 174.0, 869.47: 175.0, 872.25: 176.0, 875.03: 177.0, 877.8: 178.0, 880.58: 179.0, 883.37: 180.0, 886.15: 181.0, 888.93: 182.0, 891.71: 183.0, 894.5: 184.0, 897.28: 185.0, 900.07: 186.0, 902.86: 187.0, 905.64: 188.0, 908.43: 189.0, 911.22: 190.0, 914.02: 191.0, 916.81: 192.0, 919.6: 193.0, 922.39: 194.0, 925.19: 195.0, 927.98: 196.0, 930.78: 197.0, 933.58: 198.0, 936.38: 199.0, 939.18: 200.0, 941.98: 201.0, 944.78: 202.0, 947.58: 203.0, 950.38: 204.0, 953.19: 205.0, 955.99: 206.0, 958.8: 207.0, 961.6: 208.0, 964.41: 209.0, 967.22: 210.0, 970.03: 211.0, 972.84: 212.0, 975.65: 213.0, 978.46: 214.0, 981.27: 215.0, 984.09: 216.0, 986.9: 217.0, 989.72: 218.0, 992.54: 219.0, 995.35: 220.0, 998.17: 221.0, 1000.99: 222.0, 1003.81: 223.0}

def click_and_save_point(event, x, y, flags, param):
    # grab references to the global variables
    global points, image

    # if the left mouse button was clicked, record the point
    # (x, y) coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        points.append(point)

        # draw a circle at the point of interest
        cv2.circle(image, point, 4, (0, 255, 0), -1)
        cv2.imshow("image", image)

exit_loop = False

for file_path in npy_files:
    
    if exit_loop:
        break
    
    # load the image, clone it, and setup the mouse callback function
    points = []

    data = np.load(file_path)
    image = rgb2gray(data[:,:,[50,80,150]])
    image = ((image - image.min()) * (255 / (image.max() - image.min()))).astype(np.uint8)

    # Convert color from BGR to RGB

    clone = image.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_save_point)

    # keep looping until the 'q' key is pressed
    while True:
        # display the image and wait for a keypress
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF

        # if the 'r' key is pressed, reset the points
        if key == ord("r"):
            image = clone.copy()
            points = []

        # if the 'c' key is pressed, continue to the next image
        elif key == ord("c"):
            break
        
        # press esc to stop the program
        elif key == 27:
            exit_loop = True
            break

    # close all open windows
    cv2.destroyAllWindows()

    # write points to file
    # use os.path to get the base name of the file and replace the extension with .csv
    output_file_path = os.path.join(os.path.dirname(file_path), 'ground' + os.path.basename(file_path))  #point_
    output_file_path = os.path.splitext(output_file_path)[0] + '.csv'
    with open(output_file_path, 'w', newline='') as f:
        writer = csv.writer(f) #delimiter=';' для exel
        writer.writerow(nm_mapping.keys())
        for point in points:
            # Extract the spectral signature from the 3D data array at the selected point
            spectral_signature = data[point[1], point[0], :]

            writer.writerow(spectral_signature)
