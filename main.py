import cv2
import GLCM
import numpy as np

if __name__ == '__main__':
    im = cv2.imread('1.png')
    im_gray = np.array(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY))

    glcm = GLCM.GLCM()
    k = glcm.calGLCM(im_gray)

    print('k is: ' + str(k))

    cv2.waitKey(0)