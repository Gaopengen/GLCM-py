import math
import numpy as np


class GLCMFeature:
    def __init__(self):
        self.ASM = 0.0
        self.CON = 0.0
        self.IDM = 0.0
        self.ENT = 0.0
        self.COR = 0.0
        self.k = 0.0

class GLCM:

    def __init__(self):
        self.MY_GRAYLEVEL = 256
        self.mat_hori = np.zeros((self.MY_GRAYLEVEL, self.MY_GRAYLEVEL), dtype=float)
        self.mat_ver = np.zeros((self.MY_GRAYLEVEL, self.MY_GRAYLEVEL), dtype=float)
        self.mat_ang45 = np.zeros((self.MY_GRAYLEVEL, self.MY_GRAYLEVEL), dtype=float)
        self.mat_ang135 = np.zeros((self.MY_GRAYLEVEL, self.MY_GRAYLEVEL), dtype=float)


    def calGLCM(self, mat_in):

        self._getHorizonGLCM(mat_in)
        self._getVerticalGLCM(mat_in)
        self._getGLCM45(mat_in)
        self._getGLCM135(mat_in)

        f_h = self.calFeature(self.mat_hori)
        f_v = self.calFeature(self.mat_ver)
        f_45 = self.calFeature(self.mat_ang45)
        f_135 = self.calFeature(self.mat_ang135)

        k = (f_h.k + f_v.k + f_45.k + f_135.k)/4.0
        return k

    def calFeature(self, mat_in):
        feature = GLCMFeature()
        ux = 0.0
        uy = 0.0

        for i in range(0, self.MY_GRAYLEVEL):
            for j in range(0, self.MY_GRAYLEVEL):
                feature.ASM += pow(mat_in[i,j], 2)
                feature.CON += pow(i-j, 2) * mat_in[i,j]
                feature.IDM += mat_in[i,j] / (1+pow(i-j, 2))
                ux += i * mat_in[i,j]
                uy += j * mat_in[i,j]

        for i in range(0, self.MY_GRAYLEVEL):
            for j in range(0, self.MY_GRAYLEVEL):
                if(mat_in[i,j] > 0):
                    feature.ENT += -mat_in[i,j] * math.log(mat_in[i,j], 2)

        sigma_x = 0.0
        sigma_y = 0.0

        for i in range(0, self.MY_GRAYLEVEL):
            for j in range(0, self.MY_GRAYLEVEL):
                sigma_x += pow(i-ux, 2)*mat_in[i,j]
                sigma_y += pow(i-uy, 2)*mat_in[i,j]
                feature.COR += i * j * mat_in[i,j]

        if feature.COR-ux*uy > 0:
            feature.COR = (feature.COR - ux*uy)/(sigma_x * sigma_y)
        else:
            feature.COR = 0.0

        feature.k = feature.CON + feature.ENT - feature.IDM - feature.COR - feature.ASM
        return feature


    def _getHorizonGLCM(self, src_mat):

        row = src_mat.shape[0]
        col = src_mat.shape[1]

        for i in range(0, row):
            for j in range(0, col):
                vi = src_mat[i,j]
                if j + 1 < col-1:
                    vj = src_mat[i, j+1]
                    self.mat_hori[vi, vj] += 1.0
                if j - 1 >= 0:
                    vj = src_mat[i, j-1]
                    self.mat_hori[vi, vj] += 1.0

        self.mat_hori = self._normalization(self.mat_hori)

    def _getVerticalGLCM(self, src_mat):
        row = src_mat.shape[0]
        col = src_mat.shape[1]

        for i in range(0, row):
            for j in range(0, col):
                vi = src_mat[i,j]
                if i + 1 < row-1:
                    vj = src_mat[i+1, j]
                    self.mat_ver[vi, vj] += 1.0
                if i - 1 >= 0:
                    vj = src_mat[i-1, j]
                    self.mat_ver[vi, vj] += 1.0

        self.mat_ver = self._normalization(self.mat_ver)


    def _getGLCM45(self, src_mat):
        row = src_mat.shape[0]
        col = src_mat.shape[1]

        for i in range(0, row):
            for j in range(0, col):
                vi = src_mat[i, j]
                if i - 1 >= 0 and j + 1 < col-1:
                    vj = src_mat[i-1, j+1]
                    self.mat_ang45[vi, vj] += 1.0
                if i + 1 < row-1 and j - 1 >= 0:
                    vj = src_mat[i+1, j-1]
                    self.mat_ang45[vi, vj] += 1.0
        self.mat_ang45 = self._normalization(self.mat_ang45)


    def _getGLCM135(self, src_mat):
        row = src_mat.shape[0]
        col = src_mat.shape[1]

        for i in range(0, row):
            for j in range(0, col):
                vi = src_mat[i, j]
                if i + 1 <= row-1 and j + 1 < col-1:
                    vj = src_mat[i+1, j+1]
                    self.mat_ang135[vi, vj] += 1.0
                if i - 1 >= 0 and j - 1 >= 0:
                    vj = src_mat[i-1, j-1]
                    self.mat_ang135[vi, vj] += 1.0
        self.mat_ang135 = self._normalization(self.mat_ang135)


    def _normalization(self, mat):

        row = mat.shape[0]
        col = mat.shape[1]
        total = 0.0
        for i in range(0, row):
            for j in range(0, col):
                total += mat[i,j]

        for i in range(0, row):
            for j in range(0, col):
                mat[i,j] = mat[i,j] / total

        return mat


