import cv2
import easygui
import numpy as np
import imageio
import sys
import matploblib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image

def upload():
    ImagePath = easygui.fileopenbox()
    cartoon(ImagePath)

def cartoon(ImagePath):
    original_image = cv2.imread(ImagePath)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    if original_image is None:
        print("Cannot find any image. Please choose an appropriate file.")
        sys.exit()
        
    resized_1 = cv2.resize(original_image, (960, 540))

    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    resized_2 = cv2.resize(grayscale_image, (960, 540))
    
    smooth_grayscale = cv2.medianBlur(grayscale_image, 5)
    resized_3 = cv2.resize(smooth_grayscale, (960, 540))
    
    get_edge = cv2.adaptiveThreshold(smooth_grayscale, 255, 
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 9, 9)
        
    resized_4 = cv2.resize(get_edge, (960, 540))
    
