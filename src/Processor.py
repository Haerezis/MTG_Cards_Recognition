
import numpy as np
import cv2

class Processor :
    #img is a opencv image (numpy matrix)
    def __init__(self, img) :
        self.input_img = img.copy()
        self.output = None
        self.process()

    def process(self) :
        pass
