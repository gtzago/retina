# -*- coding: utf-8 -*-

from math import exp, inf, pi, sqrt, isnan
import numpy
import random
import sys
import time
import math

from scipy import ndimage
from skimage.feature.texture import greycomatrix
from skimage.measure import label
from skimage.measure._regionprops import regionprops
from sklearn.externals import joblib
import skimage.exposure

import cv2
import matplotlib.pyplot as plt
import numpy as np


class MicroaneurysmHemorrhageDetector:

    def __init__(self, retina_image):
        self.retina_image = retina_image

    def make_list(self, image, coord):
        '''Função para montar uma lista a partir das coordenadas dos pixels - Criada para caracteristicas_candidatos
        Parâmetros:
        image: imagem / coord: lista de coordenadas dos pixels do objeto'''

        lista = []
        for pixel in range(len(coord)):
            # Valor do pixel da coordenada do pixel
            lista.append(image[coord[pixel][0], coord[pixel][1]])

        return lista

    def aspect_ratio(self, image):
        '''Função para calcular a razão do maior comprimento em determinado eixo e
        o maior comprimento do outro eixo - Criada para caracteristicas_candidatos
        Parâmetros:
        image: imagem'''

        vertical = max(np.count_nonzero(image, axis=0))  # Maior comprimento da lesão na vertical
        horizontal = max(np.count_nonzero(image, axis=1))  # Maior comprimento da lesão na horizontal

        if vertical > horizontal:
            div = vertical / horizontal
        else:
            div = horizontal / vertical

        return div

    def sum_std_gaussian(self, imagem, coords):
        '''Função para calcular a média da gaussiana dos candidatos e seu desvio padrão
        Parâmetros:
        list_obj_org_gray: lista de pixels candidatos da imagem em tons de cinza /
        sigma: valor de sigma para realizar a gaussiana / order: ordem da gaussiana'''

        lista_pixels = self.make_list(imagem, coords)
        media = sum(lista_pixels) / len(lista_pixels)
        desvp = numpy.std(lista_pixels, ddof=1)  # Desvio padrão
        if math.isnan(desvp): # Se o desvio padrão for NaN
            desvp = 0

        return media, desvp

    def caracteristicas_candidatos(self, I_bin, I_green, I_red, I_blue, I_hue, I_sc, I_bg, I_org_gray):
        '''Função de extração de características dos candidatos à lesões
        Parâmetros:
        image: imagem binarizada / I_green: canal verde imagem / I_sc: Imagem com
        correção de sombra / I_bg: Imagem background / I_org_gray: Imagem original
        em tons de cinza'''

        features = []
        linhas, colunas = I_org_gray.shape

        image_labels, num_labels = label(I_bin, connectivity=2, return_num=True)  # Retorna a imagem com labels e a quant. de labels
        prop_image = regionprops(image_labels)  # Propriedade da imagem com labels

        I_pp = I_sc.astype(numpy.float64)
        # Gaussiana de ordem zero
        filtro_gaussiana_s1_o0 = ndimage.filters.gaussian_filter(I_pp, sigma=1, order=0)
        filtro_gaussiana_s2_o0 = ndimage.filters.gaussian_filter(I_pp, sigma=2, order=0)
        filtro_gaussiana_s4_o0 = ndimage.filters.gaussian_filter(I_pp, sigma=4, order=0)
        filtro_gaussiana_s8_o0 = ndimage.filters.gaussian_filter(I_pp, sigma=8, order=0)
        # Gaussiana de primeira ordem no eixo x
        filtro_gaussiana_s1_o1_x = ndimage.filters.gaussian_filter(I_pp, sigma=1, order=[1, 0])
        filtro_gaussiana_s2_o1_x = ndimage.filters.gaussian_filter(I_pp, sigma=2, order=[1, 0])
        filtro_gaussiana_s4_o1_x = ndimage.filters.gaussian_filter(I_pp, sigma=4, order=[1, 0])
        filtro_gaussiana_s8_o1_x = ndimage.filters.gaussian_filter(I_pp, sigma=8, order=[1, 0])
        # Gaussiana de primeira ordem no eixo y
        filtro_gaussiana_s1_o1_y = ndimage.filters.gaussian_filter(I_pp, sigma=1, order=[0, 1])
        filtro_gaussiana_s2_o1_y = ndimage.filters.gaussian_filter(I_pp, sigma=2, order=[0, 1])
        filtro_gaussiana_s4_o1_y = ndimage.filters.gaussian_filter(I_pp, sigma=4, order=[0, 1])
        filtro_gaussiana_s8_o1_y = ndimage.filters.gaussian_filter(I_pp, sigma=8, order=[0, 1])
        # Gaussiana de segunda ordem no eixo x
        filtro_gaussiana_s1_o2_xx = ndimage.filters.gaussian_filter(I_pp, sigma=1, order=[2, 0])
        filtro_gaussiana_s2_o2_xx = ndimage.filters.gaussian_filter(I_pp, sigma=2, order=[2, 0])
        filtro_gaussiana_s4_o2_xx = ndimage.filters.gaussian_filter(I_pp, sigma=4, order=[2, 0])
        filtro_gaussiana_s8_o2_xx = ndimage.filters.gaussian_filter(I_pp, sigma=8, order=[2, 0])
        # Gaussiana de segunda ordem no eixo y
        filtro_gaussiana_s1_o2_yy = ndimage.filters.gaussian_filter(I_pp, sigma=1, order=[0, 2])
        filtro_gaussiana_s2_o2_yy = ndimage.filters.gaussian_filter(I_pp, sigma=2, order=[0, 2])
        filtro_gaussiana_s4_o2_yy = ndimage.filters.gaussian_filter(I_pp, sigma=4, order=[0, 2])
        filtro_gaussiana_s8_o2_yy = ndimage.filters.gaussian_filter(I_pp, sigma=8, order=[0, 2])
        # Gaussiana de primeira ordem no eixo x e no eixo y
        filtro_gaussiana_s1_o2_xy = ndimage.filters.gaussian_filter(I_pp, sigma=1, order=[1, 1])
        filtro_gaussiana_s2_o2_xy = ndimage.filters.gaussian_filter(I_pp, sigma=2, order=[1, 1])
        filtro_gaussiana_s4_o2_xy = ndimage.filters.gaussian_filter(I_pp, sigma=4, order=[1, 1])
        filtro_gaussiana_s8_o2_xy = ndimage.filters.gaussian_filter(I_pp, sigma=8, order=[1, 1])

        for num_label in range(num_labels):  # Iteração de todos os labels
            area = prop_image[num_label].area  # Area do objeto de label [label]+1 # '1'
            perimeter = prop_image[num_label].perimeter  # Perímetro do objeto de label [label]+1 # '2'
            aspect_ratio = self.aspect_ratio(prop_image[num_label].image)  # Aspect ratio - Proporção do tamanho do candidato # '3'
            circ = (perimeter**2) / (4 * pi * area)  # Circularidade (circularity) # '4'
            sum_green_img = sum(self.make_list(I_green, prop_image[num_label].coords))  # Soma dos pixels do canal verde de determinado label # '5'
            mean_obj_green = sum_green_img / area  # Intensidade média dos pixel na imagem verde # '7'
            sum_shade_correction_img = sum(self.make_list(I_sc, prop_image[num_label].coords))  # Soma dos pixels da imagem com correção de sombra de determinado label # '6'
            mean_obj_shade_correction = sum_shade_correction_img / area  # Intensidade média dos pixel da imagem com correção de sombra # '8'
            sum_mean_img = sum(self.make_list(I_bg, prop_image[num_label].coords))  # Soma dos pixels da imagem após filtro média de determinado label
            mean_obj_mean = sum_mean_img / area  # Intensidade média dos pixel na imagem após filtro média
            list_obj_mean = self.make_list(I_bg, prop_image[num_label].coords)  # Lista dos valores de pixels da imagem após filtro de média
            arr = numpy.array(list_obj_mean)
            std_deviation = numpy.std(arr, ddof=1)  # Desvio padrão
            if std_deviation == 0 or math.isnan(std_deviation):
                normalized_green = 0
                normalized_shade_correction = 0
                normalized_mean_green = 0
                normalized_mean_shade_correction = 0
            else:
                normalized_green = (1 / std_deviation) * (sum_green_img - mean_obj_mean)  # A intensidade normalizada do canal verde # '9'
                normalized_shade_correction = (sum_shade_correction_img / std_deviation)  # A intensidade normalizada da imagem com correção de sombra # '10'
                normalized_mean_green = (1 / std_deviation) * (mean_obj_green - mean_obj_mean)  # A intensidade média normalizada do canal verde # '11'
                normalized_mean_shade_correction = (mean_obj_shade_correction / std_deviation)  # A intensidade média normalizada da imagem com correção de sombra # '12'
            i_seed = min(self.make_list(I_org_gray, prop_image[num_label].coords))  # valor do pixel mais escuro da area de determinado label em relação a imagem original - Ref. Frame # '13' 
            # '14' Compactness
            c_lin, c_col = prop_image[num_label].centroid  # Centro da lesão
            min_lin, min_col, max_lin, max_col = prop_image[num_label].bbox  # Limites da lesão
            d1 = c_lin - min_lin
            d2 = max_lin - c_lin
            d3 = c_col - min_col
            d4 = max_col - c_col
            soma_d = d1 + d2 + d3 + d4
            media_d = soma_d / 4
            v = sqrt((soma_d - media_d) / 4)  # '14'

            # '15'
            raio = round((prop_image[num_label].equivalent_diameter + 3) / 2)
            lim_min_lin = min_lin - raio
            if lim_min_lin < 0:
                lim_min_lin = 0

            lim_max_lin = max_lin + raio + 1  # Se adiciona mais 1 devido a forma como o python pega as listas
            if lim_max_lin > linhas:
                lim_max_lin = linhas

            lim_min_col = min_col - raio
            if lim_min_col < 0:
                lim_min_col = 0

            lim_max_col = max_col + raio + 1
            if lim_max_col > colunas:
                lim_max_col = colunas

            lesao_amp_verm = I_red[lim_min_lin:lim_max_lin, lim_min_col:lim_max_col]
            lin_lesao_amp_verm, col_lesao_amp_verm = lesao_amp_verm.shape
            area_amp = lin_lesao_amp_verm * col_lesao_amp_verm  # Área da lesão ampliada
            soma_verm_total = sum(sum(lesao_amp_verm))  # Soma dos pixels com aumento do diametro do circulo
            soma_verm_lesao = sum(self.make_list(I_red, prop_image[num_label].coords))  # Soma dos pixels referente a lesao do canal vermelho
            soma_verm_sem_lesao = soma_verm_total - soma_verm_lesao
            rt_verm = (soma_verm_lesao / area) - (soma_verm_sem_lesao / (area_amp - area))  # '15'

            # '16'
            lesao_amp_verde = I_green[lim_min_lin:lim_max_lin, lim_min_col:lim_max_col]
            soma_verde_total = sum(sum(lesao_amp_verde))
            soma_verde_sem_lesao = soma_verde_total - sum_green_img
            rt_verde = (mean_obj_green) - (soma_verde_sem_lesao / (area_amp - area))  # '16'

            # '17'
            lesao_amp_azul = I_blue[lim_min_lin:lim_max_lin, lim_min_col:lim_max_col]
            soma_azul_total = sum(sum(lesao_amp_azul))  # Soma dos pixels com aumento do diametro do circulo
            soma_azul_lesao = sum(self.make_list(I_blue, prop_image[num_label].coords))  # Soma dos pixels  referente a lesao do canal azul
            soma_azul_sem_lesao = soma_azul_total - soma_azul_lesao
            rt_azul = (soma_azul_lesao / area) - (soma_azul_sem_lesao / (area_amp - area))  # '17'

            # '18'
            lesao_amp_hue = I_hue[lim_min_lin:lim_max_lin, lim_min_col:lim_max_col]
            soma_hue_total = sum(sum(lesao_amp_hue))  # Soma dos pixels com aumento do diametro do circulo
            soma_hue_lesao = sum(self.make_list(I_hue, prop_image[num_label].coords))  # Soma dos pixels  referente a lesao de hue
            soma_hue_sem_lesao = soma_hue_total - soma_hue_lesao
            rt_hue = (soma_hue_lesao / area) - (soma_hue_sem_lesao / (area_amp - area))  # '18'

            # Gaussiana de ordem zero
            media_s1_o0, devsp_s1_o0 = self.sum_std_gaussian(filtro_gaussiana_s1_o0, prop_image[num_label].coords)  # Função que retorna a media e o desvio padrão da lesão candidata. Os parâmetros de entrada são: lista, sigma e ordem
            media_s2_o0, devsp_s2_o0 = self.sum_std_gaussian(filtro_gaussiana_s2_o0, prop_image[num_label].coords)  # '19'
            media_s4_o0, devsp_s4_o0 = self.sum_std_gaussian(filtro_gaussiana_s4_o0, prop_image[num_label].coords)
            media_s8_o0, devsp_s8_o0 = self.sum_std_gaussian(filtro_gaussiana_s8_o0, prop_image[num_label].coords)
            # Gaussiana de primeira ordem no eixo x
            media_s1_o1_x, devsp_s1_o1_x = self.sum_std_gaussian(filtro_gaussiana_s1_o1_x, prop_image[num_label].coords)
            media_s2_o1_x, devsp_s2_o1_x = self.sum_std_gaussian(filtro_gaussiana_s2_o1_x, prop_image[num_label].coords)
            media_s4_o1_x, devsp_s4_o1_x = self.sum_std_gaussian(filtro_gaussiana_s4_o1_x, prop_image[num_label].coords)
            media_s8_o1_x, devsp_s8_o1_x = self.sum_std_gaussian(filtro_gaussiana_s4_o1_x, prop_image[num_label].coords)
            # Gaussiana de primeira ordem no eixo y
            media_s1_o1_y, devsp_s1_o1_y = self.sum_std_gaussian(filtro_gaussiana_s1_o1_y, prop_image[num_label].coords)
            media_s2_o1_y, devsp_s2_o1_y = self.sum_std_gaussian(filtro_gaussiana_s2_o1_y, prop_image[num_label].coords)
            media_s4_o1_y, devsp_s4_o1_y = self.sum_std_gaussian(filtro_gaussiana_s4_o1_y, prop_image[num_label].coords)
            media_s8_o1_y, devsp_s8_o1_y = self.sum_std_gaussian(filtro_gaussiana_s4_o1_y, prop_image[num_label].coords)
            # Gaussiana de segunda ordem no eixo x
            media_s1_o2_xx, devsp_s1_o2_xx = self.sum_std_gaussian(filtro_gaussiana_s1_o2_xx, prop_image[num_label].coords)
            media_s2_o2_xx, devsp_s2_o2_xx = self.sum_std_gaussian(filtro_gaussiana_s2_o2_xx, prop_image[num_label].coords)
            media_s4_o2_xx, devsp_s4_o2_xx = self.sum_std_gaussian(filtro_gaussiana_s4_o2_xx, prop_image[num_label].coords)
            media_s8_o2_xx, devsp_s8_o2_xx = self.sum_std_gaussian(filtro_gaussiana_s8_o2_xx, prop_image[num_label].coords)
            # Gaussiana de segunda ordem no eixo y
            media_s1_o2_yy, devsp_s1_o2_yy = self.sum_std_gaussian(filtro_gaussiana_s1_o2_yy, prop_image[num_label].coords)
            media_s2_o2_yy, devsp_s2_o2_yy = self.sum_std_gaussian(filtro_gaussiana_s2_o2_yy, prop_image[num_label].coords)
            media_s4_o2_yy, devsp_s4_o2_yy = self.sum_std_gaussian(filtro_gaussiana_s4_o2_yy, prop_image[num_label].coords)
            media_s8_o2_yy, devsp_s8_o2_yy = self.sum_std_gaussian(filtro_gaussiana_s8_o2_yy, prop_image[num_label].coords)
            # Gaussiana de primeira ordem no eixo x e no eixo y
            media_s1_o2_xy, devsp_s1_o2_xy = self.sum_std_gaussian(filtro_gaussiana_s1_o2_xy, prop_image[num_label].coords)
            media_s2_o2_xy, devsp_s2_o2_xy = self.sum_std_gaussian(filtro_gaussiana_s2_o2_xy, prop_image[num_label].coords)
            media_s4_o2_xy, devsp_s4_o2_xy = self.sum_std_gaussian(filtro_gaussiana_s4_o2_xy, prop_image[num_label].coords)
            media_s8_o2_xy, devsp_s8_o2_xy = self.sum_std_gaussian(filtro_gaussiana_s8_o2_xy, prop_image[num_label].coords)


            caracter = np.array([area, perimeter, aspect_ratio, circ, sum_green_img, mean_obj_green, sum_shade_correction_img,
                                 mean_obj_shade_correction, normalized_green, normalized_shade_correction, normalized_mean_green,
                                 normalized_mean_shade_correction, i_seed, v, rt_verm, rt_verde, rt_azul, rt_hue, media_s1_o0,
                                 devsp_s1_o0, media_s2_o0, devsp_s2_o0, media_s4_o0, devsp_s4_o0, media_s8_o0, devsp_s8_o0,
                                 media_s1_o1_x, devsp_s1_o1_x, media_s2_o1_x, devsp_s2_o1_x, media_s4_o1_x, devsp_s4_o1_x,
                                 media_s8_o1_x, devsp_s8_o1_x, media_s1_o1_y, devsp_s1_o1_y, media_s2_o1_y, devsp_s2_o1_y,
                                 media_s4_o1_y, devsp_s4_o1_y, media_s8_o1_y, devsp_s8_o1_y, media_s1_o2_xx, devsp_s1_o2_xx,
                                 media_s2_o2_xx, devsp_s2_o2_xx, media_s4_o2_xx, devsp_s4_o2_xx, media_s8_o2_xx, devsp_s8_o2_xx,
                                 media_s1_o2_yy, devsp_s1_o2_yy, media_s2_o2_yy, devsp_s2_o2_yy, media_s4_o2_yy, devsp_s4_o2_yy,
                                 media_s8_o2_yy, devsp_s8_o2_yy, media_s1_o2_xy, devsp_s1_o2_xy, media_s2_o2_xy, devsp_s2_o2_xy,
                                 media_s4_o2_xy, devsp_s4_o2_xy, media_s8_o2_xy, devsp_s8_o2_xy])

            features.append(caracter)

        return features

    def img_resize2(self, image):
        '''Redimensiona a imagem para 640 x 480 (ou um fator proporcional) '''

        img_resize = cv2.resize(image, (640, 480), interpolation=cv2.INTER_CUBIC)

        return img_resize

    def make_dict(self, image, coord):
        '''Função para montar uma dicionario a partir das coordenadas dos pixels - Criada para object_detection
        image: imagem / coord: lista de coordenadas dos pixels do objeto'''

        dicionario = {(image[coord[pixel][0], coord[pixel][1]]): [
            coord[pixel][0], coord[pixel][1]] for pixel in range(len(coord))}
        return dicionario

    def grow_back_patologies(self, I_bin, I_org_gray, I_bg):
        '''Função que encontra o pixel semente para a realização da operação
        de crescimento de região. Já retorna a imagem com a operação de crescimento de região
        Parâmetros: I_bin: Imagem binária / I_org_gray: Imagem original em tons de cinza
        I_bg: imagem background'''

        image_labels, num_labels = label(I_bin, connectivity=2, return_num=True)  # Retorna a imagem com labels e a quant. de labels
        prop_image = regionprops(image_labels)  # Propriedade da imagem com labels
        lin, col = I_bin.shape
        img_cres_reg = np.zeros((lin, col), dtype=numpy.uint8)

        for num in range(num_labels):  # Iteração de todos os labels / labels[0] também contabiliza o label 0 por isso a necessidade em subtrair 1
            dicionario = self.make_dict(I_org_gray, prop_image[num].coords)  # Faz uma dicionario para pegar o valor do menor pixel da imagem original
            i_seed = min(self.make_list(I_org_gray, prop_image[num].coords))  # valor do pixel mais escuro da area de determinado label em relação a imagem original
            pixel = dicionario[min(dicionario)]  # Posição do pixel mais escuro
            i_bg = I_bg[pixel[0], pixel[1]]  # Mesma posição de i_seed porem em I_bg
            t = round(0.5 * (i_bg - i_seed))  # Threshold para a região em crescimento
            t = abs(t)

            reg_grow = self.crescimento_regiao(I_org_gray, pixel, t, prop_image[num].area)
            reg_grow = reg_grow.astype(np.uint8)
            img_cres_reg = np.logical_or(img_cres_reg, reg_grow, dtype=np.uint8)

        return img_cres_reg

    def crescimento_regiao(self, img, seed, threshold, pix_area, conn=4):
        """
        Função que realiza o crescimento de região.
        Parâmetros: img: Imagem original em tons de cinza / seed: Posição do pixel semente
        threshold: limiar de parada / pix_area: área do objeto antes do crescimento '''
        """

        dims = img.shape[:2]

        if conn == 4:
            orient = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # 4 vizinhos
        elif conn == 8:
            orient = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0),
                      (-1, -1), (0, -1), (1, -1)]  # 8 vizinhos

        reg = np.zeros(dims)

        # Parâmetros
        mean_reg = float(img[seed[0], seed[1]])  # Toma como média o pixel semente
        size = 1

        contour = []  # [ [[x1, y1], val1],..., [[xn, yn], valn] ]
        contour_val = []
        dist = 0

        cur_pix = [seed[0], seed[1]]

        # Crescer
        while(dist <= threshold and size < pix_area * 2):
            # Adição de pixels
            for j in range(len(orient)):
                # Seleciona novo candidato
                temp_pix = [cur_pix[0] + orient[j]
                            [0], cur_pix[1] + orient[j][1]]

                # Verificando se ele pertence à imagem
                # Retorna booleano
                is_in_img = dims[0] > temp_pix[0] > 0 and dims[1] > temp_pix[1] > 0
                # O candidato é tomado se ainda não estiver selecionado antes
                if (is_in_img and (reg[temp_pix[0], temp_pix[1]] == 0)):
                    contour.append(temp_pix)
                    contour_val.append(img[temp_pix[0], temp_pix[1]])
                    reg[temp_pix[0], temp_pix[1]] = 255
            # Adicionando o pixel mais próximo do contorno nele
            dist = abs(int(numpy.mean(contour_val)) - mean_reg)

            dist_list = [abs(i - mean_reg) for i in contour_val]
            dist = min(dist_list)    # Selecionando a distância mínima
            index = dist_list.index(min(dist_list))  # mean distance index
            size += 1  # Atualizando tamanho da região
            reg[cur_pix[0], cur_pix[1]] = 255

            # Atualizando semente
            cur_pix = contour[index]

            # Removendo pixel da vizinhança
            del contour[index]
            del contour_val[index]

        return reg

    def image_preprocessing(self, image):
        ''' Pré-processamento da imagem '''

        img_resize = self.img_resize2(image)
        I_org = img_resize  # Imagem original
        I_org_gray = cv2.cvtColor(I_org, cv2.COLOR_RGB2GRAY)  # Imagem original em tons de cinza
        I_hsv = cv2.cvtColor(I_org, cv2.COLOR_RGB2HSV)
        I_org = I_org.astype(numpy.int16)
        I_hue = I_hsv[:, :, 0]  # Hue (matiz) da imagem hsv
        I_blue = I_org[:, :, 0]  # Canal azul da imagem
        I_green = I_org[:, :, 1]  # Canal verde da imagem
        I_red = I_org[:, :, 2]  # Canal vermelho da imagem
        I_bg = ndimage.median_filter(I_green, 25)  # Imagem background
        I_sc = I_green - I_bg  # Shade correction - Correção de sombra
        I_sc[I_sc > 0] = 0  # Remoção de patologias brilhantes

        return I_org, I_org_gray, I_hsv, I_hue, I_blue, I_green, I_red, I_bg, I_sc

    def gaussianas(self, image):
        ''' Pré-processa a imagem e realiza as operações de gaussiana sobre ela. Utilizada
        para processar as imagens de treinamento do k-NN'''

        # I_pp = self.image_preprocessing(image)
        I_pp = image.astype(numpy.float64)
        # Gaussianas de primeira ordem - Eixo x
        img_gaussian_1o_1_x = ndimage.filters.gaussian_filter(I_pp, sigma=1, order=[1, 0])
        img_gaussian_1o_2_x = ndimage.filters.gaussian_filter(I_pp, sigma=2, order=[1, 0])
        img_gaussian_1o_4_x = ndimage.filters.gaussian_filter(I_pp, sigma=4, order=[1, 0])
        img_gaussian_1o_8_x = ndimage.filters.gaussian_filter(I_pp, sigma=8, order=[1, 0])
        img_gaussian_1o_16_x = ndimage.filters.gaussian_filter(I_pp, sigma=16, order=[1, 0])
        # Gaussianas de primeira ordem - Eixo y
        img_gaussian_1o_1_y = ndimage.filters.gaussian_filter(I_pp, sigma=1, order=[0, 1])
        img_gaussian_1o_2_y = ndimage.filters.gaussian_filter(I_pp, sigma=2, order=[0, 1])
        img_gaussian_1o_4_y = ndimage.filters.gaussian_filter(I_pp, sigma=4, order=[0, 1])
        img_gaussian_1o_8_y = ndimage.filters.gaussian_filter(I_pp, sigma=8, order=[0, 1])
        img_gaussian_1o_16_y = ndimage.filters.gaussian_filter(I_pp, sigma=16, order=[0, 1])
        # Gaussianas de segunda ordem - Eixo x
        img_gaussian_2o_1_x = ndimage.filters.gaussian_filter(I_pp, sigma=1, order=[2, 0])
        img_gaussian_2o_2_x = ndimage.filters.gaussian_filter(I_pp, sigma=2, order=[2, 0])
        img_gaussian_2o_4_x = ndimage.filters.gaussian_filter(I_pp, sigma=4, order=[2, 0])
        img_gaussian_2o_8_x = ndimage.filters.gaussian_filter(I_pp, sigma=8, order=[2, 0])
        img_gaussian_2o_16_x = ndimage.filters.gaussian_filter(I_pp, sigma=16, order=[2, 0])
        # Gaussianas de segunda ordem - Eixo y
        img_gaussian_2o_1_y = ndimage.filters.gaussian_filter(I_pp, sigma=1, order=[0, 2])
        img_gaussian_2o_2_y = ndimage.filters.gaussian_filter(I_pp, sigma=2, order=[0, 2])
        img_gaussian_2o_4_y = ndimage.filters.gaussian_filter(I_pp, sigma=4, order=[0, 2])
        img_gaussian_2o_8_y = ndimage.filters.gaussian_filter(I_pp, sigma=8, order=[0, 2])
        img_gaussian_2o_16_y = ndimage.filters.gaussian_filter(I_pp, sigma=16, order=[0, 2])

        prop_img = np.dstack((I_pp, img_gaussian_1o_1_x, img_gaussian_1o_2_x, img_gaussian_1o_4_x,
                              img_gaussian_1o_8_x, img_gaussian_1o_16_x, img_gaussian_1o_1_y,
                              img_gaussian_1o_2_y, img_gaussian_1o_4_y, img_gaussian_1o_8_y,
                              img_gaussian_1o_16_y, img_gaussian_2o_1_x, img_gaussian_2o_2_x,
                              img_gaussian_2o_4_x, img_gaussian_2o_8_x, img_gaussian_2o_16_x,
                              img_gaussian_2o_1_y, img_gaussian_2o_2_y, img_gaussian_2o_4_y,
                              img_gaussian_2o_8_y, img_gaussian_2o_16_y))

        return prop_img

    def binarizar_img(self, image, prob_p, are):
        '''Função destinada a bizarizar a imagem, retornando somente lesões
        e pequenos ou pedaços de vasos sanguíneos'''
 
        vt_carac = [] # Lista para as caracteristicas retiradas da imagem
 
        lin_prop_img, col_prop_img, d = image.shape
 
        for linha in range(lin_prop_img):
            for coluna in range(col_prop_img):
                vt_carac.append(image[linha][coluna])
 
        vt_carac = np.array(vt_carac)
 
        k_NN = joblib.load('treinamento_kNN.pkl')
 
        prob_pixels = k_NN.predict_proba(vt_carac)  # Extrai a probabilidade do pixel ser 0 ou 1
 
        list_aux = []  # Lista auxiliar
        prob_pixels_1 = []  # Lista para armazenar a probabilidade do pixel ser 1
 
        list_aux = [e[1] for e in prob_pixels]
        for prob in list_aux:
            if prob >= prob_p:
                prob_pixels_1.append(255)
            else:
                prob_pixels_1.append(0)
 
        prob_pixels_1 = np.array(prob_pixels_1)
        img_bin = np.reshape(prob_pixels_1, (lin_prop_img, col_prop_img))
        img_bin = img_bin.astype(numpy.uint8)
 
        # Rotina para eliminar elementos com área maior que 300 pixels
        image_labels, num_labels = label(img_bin, connectivity=2, return_num=True)
        prop_image = regionprops(image_labels)
        for num_label in range(num_labels):
            if prop_image[num_label].area > are:
                for coord in prop_image[num_label].coords:
                    img_bin[coord[0], coord[1]] = 0
 
        return img_bin

    def morfologia_matematica(self, I_sc, ele):
        '''Função destinada a bizarizar a imagem, retornando somente lesões
        e pequenos ou pedaços de vasos sanguíneos'''

        n = ele   # 9 pixels
        comprimento = 1  # Comprimento 1
        filtro = np.zeros((n, n), np.uint8)
        filtro[(n // 2) - (comprimento // 2):(n // 2) + (comprimento // 2) + 1] = 1  # Elemento estruturante
        elementos_estruturantes = self.matched_filter_bank(filtro)  # 12 elementos estruturantes rotacionados a 15 graus
        aberturas = [cv2.morphologyEx(I_sc, cv2.MORPH_OPEN, k) for k in elementos_estruturantes]  # Doze operações de abertura na imagem I_sc
        max_aberturas = np.amax(aberturas, 0)  # Valor máximo de todas as aberturas / vasculature map
        I_lesion = I_sc - max_aberturas
        I_match = cv2.GaussianBlur(I_lesion, ksize=(11, 11), sigmaX=1, sigmaY=1)
        qtde_pixels, pixels = skimage.exposure.histogram(I_match, nbins=256)  # Retorna qtde de pixels e o respectivo pixel
        if not len(pixels) == 1:
            qtde_pixels = qtde_pixels[1:]  # Retina a qtde de pixels do pixel 0 pois não é de interesse
            pixels = pixels[1:]  # Retina o pixel 0 pois não é de interesse
        ret, I_bin = cv2.threshold(I_match, pixels[np.argmax(qtde_pixels)], 255, cv2.THRESH_BINARY)  # pixels[np.argmax(qtde_pixels)] retorna o pixel que mais aparece * De acordo com o artigo
        I_bin = cv2.convertScaleAbs(I_bin)

        return I_bin
    
    
