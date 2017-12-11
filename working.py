from math import exp, pi, sqrt
from matplotlib.pyplot import axis
from numpy import uint8, dtype
import numpy
import requests
from statistics import mode
import time
import random

from scipy import ndimage
import skimage.exposure
import skimage.filters
from skimage.measure import regionprops, perimeter, label
from sklearn import datasets
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier

import cv2
from eyecare.feature_detection.base import MicroaneurysmHemorrhageDetector
from eyecare.image.base import RetinaImage
import matplotlib.pyplot as plt
import numpy as np

from eyecare.segmenter.base import BloodVesselSegmenter
from matplotlib.colors import ListedColormap
from datetime import datetime


BASE_URL = 'http://0.0.0.0:8000'
RETINA_DB_URL = BASE_URL + '/retina_db/'
RESULTS_URL = BASE_URL + '/results/'
USER = 'admin'
PASS = 'admin'



# Buscanto todas imagens do banco de dados DRIVE
r = requests.get(
    RETINA_DB_URL + 'retinas/?origin_database=DRIVE', auth=(USER, PASS))


# Pegando todas as imagens do DRIVE
ret_array = [[(d['img_url'], d['id']) for d in r.json()['results']]]
while r.json()['next'] is not None:
    r = requests.get(r.json()['next'], auth=(USER, PASS))
    ret_array.append([(d['img_url'], d['id']) for d in r.json()['results']])

flat_ret_array = [item for sublist in ret_array for item in sublist]  # Tornando ret_array flat

vetor_caracteristicas = []  # lista para as carasteristicas da imagem processada
vetor_classes = []  # lista que representa as classes da imagem: 0 ou 1

for img_retina in range(len(flat_ret_array)):  # range(range(len(flat_ret_array)))
    retina = RetinaImage(BASE_URL + flat_ret_array[img_retina][0])  # Imagem da retina
    v = requests.get(RETINA_DB_URL + 'manualbloodvessels/?retina=%d' % (flat_ret_array[img_retina][1]), auth=(USER, PASS))
    vasos_array = v.json()['results']
    vasos = RetinaImage(BASE_URL + vasos_array[0]['img_url'])  # Imagem dos vasos sanguineos da retina
    algoritmo = MicroaneurysmHemorrhageDetector(retina.img)

    I_org, I_org_gray, I_hsv, I_hue, I_blue, I_green, I_red, I_bg, I_sc = algoritmo.image_preprocessing(retina.img)  # Preprocessamento da img de retina
    retina_gauss = algoritmo.gaussianas(I_sc)
    resize_vasos = algoritmo.img_resize2(vasos.img)  # Mudança de tamanho de imagem dos vasos
    vasos1 = resize_vasos[:, :, 1]
    limite = skimage.filters.threshold_otsu(vasos1)
    vasos1[vasos1 > limite] = 255
    vasos1[vasos1 <= limite] = 0

    vt_carac_aux_0 = []  # Vetor aux de características para classe 0
    vt_classes_aux_0 = []  # Vetor aux de classe para classe 0
    vt_carac_aux_255 = []  # Vetor aux de características para classe 255
    vt_classes_aux_255 = []  # Vetor aux de classe para classe 0
    amostras = 25

    while len(vt_carac_aux_0) < amostras or len(vt_carac_aux_255) < amostras:  # Seleciona quant. amostras de pixels de cada classe
        linha_aleatoria = np.random.randint(low=40, high=750)  # Número de linha aleatório / Valores escolhidos com base na análise da imagem para pegar somente a retina
        coluna_aleatoria = np.random.randint(low=10, high=726)  # Número de coluna aleatório
        if I_sc[linha_aleatoria][coluna_aleatoria] == 0:
            if not len(vt_carac_aux_0) == amostras:
                vt_carac_aux_0.append(retina_gauss[linha_aleatoria][coluna_aleatoria])
                vt_classes_aux_0.append(vasos1[linha_aleatoria][coluna_aleatoria])
        else:
            if not len(vt_carac_aux_255) == amostras:
                vt_carac_aux_255.append(retina_gauss[linha_aleatoria][coluna_aleatoria])
                vt_classes_aux_255.append(vasos1[linha_aleatoria][coluna_aleatoria])

    vetor_caracteristicas.extend(vt_carac_aux_0)
    vetor_caracteristicas.extend(vt_carac_aux_255)
    vetor_classes.extend(vt_classes_aux_0)
    vetor_classes.extend(vt_classes_aux_255)


vetor_caracteristicas = np.array(vetor_caracteristicas)
vetor_classes = np.array(vetor_classes)


# Rotina de pegar as imagens e extrair características das hemorragias
r = requests.get(
    RETINA_DB_URL + 'retinas/?origin_database=DIARETDB1', auth=(USER, PASS))

# Pegando todas as imagens do DIARETDB1
ret_array = [[(d['img_url'], d['id']) for d in r.json()['results']]]
while r.json()['next'] is not None:
    r = requests.get(r.json()['next'], auth=(USER, PASS))
    ret_array.append([(d['img_url'], d['id']) for d in r.json()['results']])

flat_ret_array = [item for sublist in ret_array for item in sublist]  # Tornando ret_array flat


retina_id_treinamento = [37717, 37727, 37710, 37715, 37676, 37672, 37731, 37738, 37747, 37681, 37709, 37707, 37702, 37685, 37725, 37724, 37674, 37708, 37693, 37697, 37732, 37686, 37740, 37664, 37729, 37706, 37682, 37688]


elemento_estruturante = [7, 9, 11]
probabilidade_vaso = [0.35, 0.40, 0.45]
area_vaso = [275, 300]


for ele in elemento_estruturante:
    for prob in probabilidade_vaso:
        for are in area_vaso:

            print('Ele estrut: %d, Prob vaso: %f, Area vaso: %d' % (ele, prob, are))

            qtde_img_com_hemo = 0  # Variáveis auxiliares para contagem de imagens
            qtde_img_sem_hemo = 0
            vetor_caracteristicas = []  # lista para as carasteristicas da imagem processada
            vetor_classes = []  # lista que representa as classes da imagem: 0 ou 1

            num_lesoes = 0
            num_nao_lesoes = 0

            vt_carac_aux_0 = []  # Vetor aux de características para classe 0 - Não lesão
            vt_classes_aux_0 = []  # Vetor aux de classe para classe 0 - Não lesão
            vt_carac_aux_1 = []  # Vetor aux de características para classe 1 - Lesão
            vt_classes_aux_1 = []  # Vetor aux de classe para classe 1 - Lesão
            novo_vt_carac_aux_0 = []
            novo_vt_carac_aux_1 = []

            # Rotina para treinamento do k-NN identificando lesões e não lesões
            for img_retina in range(len(flat_ret_array)):
                if flat_ret_array[img_retina][1] in retina_id_treinamento:  # Verificar se a imagem esta na lista de treinamento
                    retina = RetinaImage(BASE_URL + flat_ret_array[img_retina][0])  # Imagem da retina
                    # Marcações de hemorragia
                    h = requests.get(RETINA_DB_URL + 'manualhemorrhages/?retina=%d' % (flat_ret_array[img_retina][1]), auth=(USER, PASS))
                    hemorragia_array = h.json()['results']
                    hemorragia = RetinaImage(BASE_URL + hemorragia_array[0]['img_url'])  # Imagem com marcações de hemorragia da retina
                    # Marcações de pequenos pontos vermelhos
                    rsd = requests.get(RETINA_DB_URL + 'manualredsmalldots/?retina=%d' % (flat_ret_array[img_retina][1]), auth=(USER, PASS))
                    rsd_array = rsd.json()['results']
                    red_small_dot = RetinaImage(BASE_URL + rsd_array[0]['img_url'])  # Imagem com marcações de hemorragia da retina
                    algoritmo = MicroaneurysmHemorrhageDetector(retina.img)
            
                    hemorragia_resize = algoritmo.img_resize2(hemorragia.img)
                    I_hemorragia_gray = cv2.cvtColor(hemorragia_resize, cv2.COLOR_RGB2GRAY)
            
                    red_small_dot_resize = algoritmo.img_resize2(red_small_dot.img)
                    red_small_dot_gray = cv2.cvtColor(red_small_dot_resize, cv2.COLOR_RGB2GRAY)
            
                    # Pré-processamento
                    I_org, I_org_gray, I_hsv, I_hue, I_blue, I_green, I_red, I_bg, I_sc = algoritmo.image_preprocessing(retina.img)
            
                    retina_gauss = algoritmo.gaussianas(I_sc)  # Gaussianas da imagem
            
                    # Segmentador de vasos sanguíneos
                    segmentador_vasos = BloodVesselSegmenter(retina)
                    segmentador_vasos.find_blood_vessels()
                    vasos = segmentador_vasos.predicted_blood_vessels
                    vasos = vasos.astype(numpy.uint8)
                    vasos_inv = np.logical_not(vasos)
                    vasos_inv = vasos_inv.astype(numpy.uint8)
                    linha, coluna = I_org_gray.shape
                    vasos_reshape = cv2.resize(vasos_inv, (coluna, linha), interpolation=cv2.INTER_CUBIC)
                    kernel = np.ones((3, 3))
                    erosao = cv2.erode(vasos_reshape, kernel, iterations=1)
            
                    # Extração dos candidatos baseado em morfologia matemática
                    
                    I_bin = algoritmo.morfologia_matematica(I_sc, ele)
                    lesoes_mm = np.logical_and(erosao, I_bin, dtype=np.uint8)
                    lesoes_mm = lesoes_mm.astype(numpy.uint8)
                    retina_cres_reg_mm = algoritmo.grow_back_patologies(lesoes_mm, I_org_gray, I_bg)
            
                    # Extração dos candidatos baseado em classificação de pixels
             
                    # Binarizando a imagem da retina
                    retina_binarizada = algoritmo.binarizar_img(retina_gauss, prob, are)  # Retorna pequenos vasos e lesões
                    lesoes_cp = np.logical_and(erosao, retina_binarizada, dtype=np.uint8)
                    lesoes_cp = lesoes_cp.astype(numpy.uint8)
                    # Crescimento de região da retina binarizada
                    retina_cres_reg_cp = algoritmo.grow_back_patologies(lesoes_cp, I_org_gray, I_bg)  # O que retornar não será lesões neste caso.
             
                    # União das duas extrações
                    retina_cres_reg = np.logical_or(retina_cres_reg_mm, retina_cres_reg_cp, dtype=np.uint8)
                    retina_cres_reg = retina_cres_reg.astype(numpy.uint8)
            
                    # Setando marcações
                    I_hemorragia_gray[I_hemorragia_gray >= 192] = 255  # Sendo a probabilidade de ser lesão >= 75% recebe o valor 255
                    I_hemorragia_gray[I_hemorragia_gray < 192] = 0
            
                    red_small_dot_gray[red_small_dot_gray >= 192] = 255  # Sendo a probabilidade de ser lesão >= 75% recebe o valor 255
                    red_small_dot_gray[red_small_dot_gray < 192] = 0
            
                    marcacoes = np.logical_or(I_hemorragia_gray, red_small_dot_gray, dtype=np.uint8)
                    marcacoes = marcacoes.astype(numpy.uint8)
            
                    kernel = np.ones((2, 2))
                    dilatacao = cv2.dilate(marcacoes, kernel, iterations=1)
            
                    lesoes = np.logical_and(dilatacao, retina_cres_reg, dtype=np.uint8)  # Encontrando candidatos que estão dentro do círculo
                    lesoes = lesoes.astype(numpy.uint8)
                    lesoes[lesoes == 0] = 0
                    lesoes[lesoes == 1] = 255
                    
                    nao_lesao = retina_cres_reg - lesoes  # Pega somente o que não é considerado lesão
                    nao_lesao = nao_lesao.astype(numpy.uint8)
                    nao_lesao[nao_lesao == 0] = 0
                    nao_lesao[nao_lesao == 1] = 255
            
                                
                    if numpy.any(lesoes):

                        carac_cand_lesao = algoritmo.caracteristicas_candidatos(lesoes, I_green, I_red, I_blue, I_hue, I_sc, I_bg, I_org_gray)
            
                        num_lesoes = num_lesoes + len(carac_cand_lesao)
            
                        vt_carac_aux_1.append(carac_cand_lesao)
            
                    if numpy.any(nao_lesao):
            
                        carac_cand_lesao = algoritmo.caracteristicas_candidatos(nao_lesao, I_green, I_red, I_blue, I_hue, I_sc, I_bg, I_org_gray)
            
                        num_nao_lesoes = num_nao_lesoes + len(carac_cand_lesao)
                        vt_carac_aux_0.append(carac_cand_lesao)
                    
            
            flat_vt_carac_aux_0 = [item for sublist in vt_carac_aux_0 for item in sublist]
            flat_vt_carac_aux_1 = [item for sublist in vt_carac_aux_1 for item in sublist]
            
            
            if (num_nao_lesoes > num_lesoes): # Número de não lesões maior que o número de lesões? 
                # Sim, então vetor de não lesões deve ser do mesmo tamanho que vetor de lesões 
                novo_vt_carac_aux_0 = flat_vt_carac_aux_0[:num_lesoes]
            
                vt_classes_aux_0 = [0] * num_lesoes
                vt_classes_aux_1 = [1] * num_lesoes
            
                vetor_caracteristicas = novo_vt_carac_aux_0 + flat_vt_carac_aux_1
                vetor_classes = vt_classes_aux_0 + vt_classes_aux_1
            else:
                novo_vt_carac_aux_1 = flat_vt_carac_aux_1[:num_nao_lesoes]
            
                vt_classes_aux_0 = [0] * num_nao_lesoes
                vt_classes_aux_1 = [1] * num_nao_lesoes
            
                vetor_caracteristicas = flat_vt_carac_aux_0 + novo_vt_carac_aux_1
                vetor_classes = vt_classes_aux_0 + vt_classes_aux_1
            
            vetor_caracteristicas = np.array(vetor_caracteristicas)
            vetor_classes = np.array(vetor_classes)
            
            k_NN_lesao = KNeighborsClassifier(n_neighbors=11)  # Instancia o classificador


            k_NN_lesao.fit(vetor_caracteristicas, vetor_classes)  # Treina classificador
            
            # Rotina para identificar lesões e não lesões
            
            total_candidatos = 0
            total_verdadeiro_positivo = 0
            total_verdadeiro_negativo = 0
            total_falso_positivo = 0
            total_falso_negativo = 0
            total_vp_img50 = 0
            total_vn_img50 = 0
            total_fp_img50 = 0
            total_fn_img50 = 0
            
            total_vp_img65 = 0
            total_vn_img65 = 0
            total_fp_img65 = 0
            total_fn_img65 = 0
            
            total_vp_img75 = 0
            total_vn_img75 = 0
            total_fp_img75 = 0
            total_fn_img75 = 0
            
            total_vp_img85 = 0
            total_vn_img85 = 0
            total_fp_img85 = 0
            total_fn_img85 = 0
            
            total_vp_img90 = 0
            total_vn_img90 = 0
            total_fp_img90 = 0
            total_fn_img90 = 0
            
            imgs_saudaveis = 0
            imgs_doentes = 0
            
            for img_retina in range(len(flat_ret_array)):
                if not flat_ret_array[img_retina][1] in retina_id_treinamento: # Verificar se a retina já foi utilizada no treinamento
                    retina = RetinaImage(BASE_URL + flat_ret_array[img_retina][0])  # Imagem da retina
                    # Marcações de hemorragia
                    h = requests.get(RETINA_DB_URL + 'manualhemorrhages/?retina=%d' % (flat_ret_array[img_retina][1]), auth=(USER, PASS))
                    hemorragia_array = h.json()['results']
                    hemorragia = RetinaImage(BASE_URL + hemorragia_array[0]['img_url'])  # Imagem com marcações de hemorragia da retina
                    # Marcações de pequenos pontos vermelhos
                    rsd = requests.get(RETINA_DB_URL + 'manualredsmalldots/?retina=%d' % (flat_ret_array[img_retina][1]), auth=(USER, PASS))
                    rsd_array = rsd.json()['results']
                    red_small_dot = RetinaImage(BASE_URL + rsd_array[0]['img_url'])  # Imagem com marcações de hemorragia da retina
            
                    algoritmo = MicroaneurysmHemorrhageDetector(retina.img)
                    
                    hemorragia_resize = algoritmo.img_resize2(hemorragia.img)
                    I_hemorragia_gray = cv2.cvtColor(hemorragia_resize, cv2.COLOR_RGB2GRAY)
            
                    red_small_dot_resize = algoritmo.img_resize2(red_small_dot.img)
                    red_small_dot_gray = cv2.cvtColor(red_small_dot_resize, cv2.COLOR_RGB2GRAY)
                            
            
                    I_org, I_org_gray, I_hsv, I_hue, I_blue, I_green, I_red, I_bg, I_sc = algoritmo.image_preprocessing(retina.img)
            
                    retina_gauss = algoritmo.gaussianas(I_sc) # Gaussianas da imagem
            
                    # Segmentador de vasos sanguíneos
                    segmentador_vasos = BloodVesselSegmenter(retina)
                    segmentador_vasos.find_blood_vessels()
                    vasos = segmentador_vasos.predicted_blood_vessels
                    vasos = vasos.astype(numpy.uint8)
                    vasos_inv = np.logical_not(vasos)
                    vasos_inv = vasos_inv.astype(numpy.uint8)
                    linha, coluna = I_org_gray.shape
                    vasos_reshape = cv2.resize(vasos_inv, (coluna, linha), interpolation=cv2.INTER_CUBIC)
                    kernel = np.ones((3, 3))
                    erosao = cv2.erode(vasos_reshape, kernel, iterations=1)
            
                    # Extração dos candidatos baseado em morfologia matemática
             
                    
                    I_bin = algoritmo.morfologia_matematica(I_sc, ele)
                    lesoes_mm = np.logical_and(erosao, I_bin, dtype=np.uint8)
                    lesoes_mm = lesoes_mm.astype(numpy.uint8)
                    retina_cres_reg_mm = algoritmo.grow_back_patologies(lesoes_mm, I_org_gray, I_bg)
             
                    # Extração dos candidatos baseado em classificação de pixels
              
                    # Binarizando a imagem da retina
                    retina_binarizada = algoritmo.binarizar_img(retina_gauss, prob, are)  # Retorna pequenos vasos e lesões
                    lesoes_cp = np.logical_and(erosao, retina_binarizada, dtype=np.uint8)
                    lesoes_cp = lesoes_cp.astype(numpy.uint8)
                    # Crescimento de região da retina binarizada
                    retina_cres_reg_cp = algoritmo.grow_back_patologies(lesoes_cp, I_org_gray, I_bg)  # O que retornar não será lesões neste caso.
             
                    # União das duas extrações
                    retina_cres_reg = np.logical_or(retina_cres_reg_mm, retina_cres_reg_cp, dtype=np.uint8)
                    retina_cres_reg = retina_cres_reg.astype(numpy.uint8)
            
            #         plt.figure(), plt.imshow(retina_cres_reg), plt.gray(), plt.title('Reg cres resultado')
            #         plt.show()

                    # Extração de característica dos candidatos
                    carac_cand_lesao = algoritmo.caracteristicas_candidatos(retina_cres_reg, I_green, I_red, I_blue, I_hue, I_sc, I_bg, I_org_gray)
                    carac_cand_lesao = np.array(carac_cand_lesao)
            
            #         ini = time.time()
            #         print('Identificando lesões')
                    # classificacao = k_NN_lesao.predict(carac_cand_lesao) # Retorna lista de 0's e 1's
            
                    prob_class = k_NN_lesao.predict_proba(carac_cand_lesao) # Retorna a probabilidade de cada candidato na classificação
            #         fim = time.time()
            #         print('Tempo da identificação: %d' % (fim - ini))
            
                    prob_lesao = [e[1] for e in prob_class] # Retorna a probabilidade do candidato ser lesão

                    # Setando marcações            
                    I_hemorragia_gray[I_hemorragia_gray >= 192] = 255  # Sendo a probabilidade de ser lesão >= 75% recebe o valor 255
                    I_hemorragia_gray[I_hemorragia_gray < 192] = 0
            
                    red_small_dot_gray[red_small_dot_gray >= 192] = 255  # Sendo a probabilidade de ser lesão >= 75% recebe o valor 255
                    red_small_dot_gray[red_small_dot_gray < 192] = 0
            
                    marcacoes = np.logical_or(I_hemorragia_gray, red_small_dot_gray, dtype=np.uint8)
                    marcacoes = marcacoes.astype(numpy.uint8)
            
                    kernel = np.ones((2, 2))
                    dilatacao = cv2.dilate(marcacoes, kernel, iterations=1)
            
                    # Estatística das imagens
            
                    # 0.5
                    if numpy.any(dilatacao): # A imagem possui lesões ?
                        # Sim
                        imgs_doentes = imgs_doentes + 1
                        if (max(prob_lesao) >= 0.50): # Algum candidato possui uma probabilidade de ser lesão maior que 50%?
                            # Sim
                            total_vp_img50 = total_vp_img50 + 1 # Verdairos positivos
                        else:
                            # Não 
                            total_fn_img50 = total_fn_img50 + 1 # Falsos negativos
                    else:
                        # Não
                        imgs_saudaveis = imgs_saudaveis + 1
                        if (max(prob_lesao) >= 0.50): # Algum candidato possui uma probabilidade de ser lesão maior que 50%?
                            # Sim
                            total_fp_img50 = total_fp_img50 + 1 # Falsos positivos
                        else:
                            # Não
                            total_vn_img50 = total_vn_img50 + 1 # Verdadeiros negativos
                    # 0.65       
                    if numpy.any(dilatacao): # A imagem possui lesões ?
                        # Sim
                        if (max(prob_lesao) >= 0.65): # Algum candidato possui uma probabilidade de ser lesão maior que 65%?
                            # Sim
                            total_vp_img65 = total_vp_img65 + 1 # Verdairos positivos
                        else:
                            # Não 
                            total_fn_img65 = total_fn_img65 + 1 # Falsos negativos
                    else:
                        # Não
                        if (max(prob_lesao) >= 0.65): # Algum candidato possui uma probabilidade de ser lesão maior que 65%?
                            # Sim
                            total_fp_img65 = total_fp_img65 + 1 # Falsos positivos
                        else:
                            # Não
                            total_vn_img65 = total_vn_img65 + 1 # Verdadeiros negativos

                    # 0.75
                    if numpy.any(dilatacao): # A imagem possui lesões ?
                        # Sim
                        if (max(prob_lesao) >= 0.75): # Algum candidato possui uma probabilidade de ser lesão maior que 75%?
                            # Sim
                            total_vp_img75 = total_vp_img75 + 1 # Verdairos positivos
                        else:
                            # Não 
                            total_fn_img75 = total_fn_img75 + 1 # Falsos negativos
                    else:
                        # Não
                        if (max(prob_lesao) >= 0.75): # Algum candidato possui uma probabilidade de ser lesão maior que 75%?
                            # Sim
                            total_fp_img75 = total_fp_img75 + 1 # Falsos positivos
                        else:
                            # Não
                            total_vn_img75 = total_vn_img75 + 1 # Verdadeiros negativos
                    # 0.85
                    if numpy.any(dilatacao): # A imagem possui lesões ?
                        # Sim
                        if (max(prob_lesao) >= 0.85): # Algum candidato possui uma probabilidade de ser lesão maior que 85%?
                            # Sim
                            total_vp_img85 = total_vp_img85 + 1 # Verdairos positivos
                        else:
                            # Não 
                            total_fn_img85 = total_fn_img85 + 1 # Falsos negativos
                    else:
                        # Não
                        if (max(prob_lesao) >= 0.85): # Algum candidato possui uma probabilidade de ser lesão maior que 85%?
                            # Sim
                            total_fp_img85 = total_fp_img85 + 1 # Falsos positivos
                        else:
                            # Não
                            total_vn_img85 = total_vn_img85 + 1 # Verdadeiros negativos

                    # 0.90
                    if numpy.any(dilatacao): # A imagem possui lesões ?
                        # Sim
                        if (max(prob_lesao) >= 0.90): # Algum candidato possui uma probabilidade de ser lesão maior que 90%?
                            # Sim
                            total_vp_img90 = total_vp_img90 + 1 # Verdairos positivos
                        else:
                            # Não 
                            total_fn_img90 = total_fn_img90 + 1 # Falsos negativos
                    else:
                        # Não
                        if (max(prob_lesao) >= 0.90): # Algum candidato possui uma probabilidade de ser lesão maior que 90%?
                            # Sim
                            total_fp_img90 = total_fp_img90 + 1 # Falsos positivos
                        else:
                            # Não
                            total_vn_img90 = total_vn_img90 + 1 # Verdadeiros negativos
            
                    # print(prob_class)
            
                    image_labels, num_candidatos = label(retina_cres_reg, connectivity=2, return_num=True)
                    total_candidatos = total_candidatos + num_candidatos
                    
            
            sensibilidade_img50 = total_vp_img50 / (total_vp_img50 + total_fn_img50)
            especificidade_img50 = total_vn_img50 / (total_vn_img50 + total_fp_img50)
            
            sensibilidade_img65 = total_vp_img65 / (total_vp_img65 + total_fn_img65)
            especificidade_img65 = total_vn_img65 / (total_vn_img65 + total_fp_img65)
            
            sensibilidade_img75 = total_vp_img75 / (total_vp_img75 + total_fn_img75)
            especificidade_img75 = total_vn_img75 / (total_vn_img75 + total_fp_img75)
            
            sensibilidade_img85 = total_vp_img85 / (total_vp_img85 + total_fn_img85)
            especificidade_img85 = total_vn_img85 / (total_vn_img85 + total_fp_img85)
            
            sensibilidade_img90 = total_vp_img90 / (total_vp_img90 + total_fn_img90)
            especificidade_img90 = total_vn_img90 / (total_vn_img90 + total_fp_img90)
            
            print('Estatística imagens')
            print('Verdadeiro positivo 50 = %d' % total_vp_img50)
            print('Verdadeiro negativo 50 = %d' % total_vn_img50)
            print('Falso positivo 50 = %d' % total_fp_img50)
            print('Falso negativo 50 = %d' % total_fn_img50)
            print('Sensibilidade 50 = %f' % sensibilidade_img50)
            print('Especificidade 50 = %f' % especificidade_img50)
            
            print('Verdadeiro positivo 65 = %d' % total_vp_img65)
            print('Verdadeiro negativo 65 = %d' % total_vn_img65)
            print('Falso positivo 65 = %d' % total_fp_img65)
            print('Falso negativo 65 = %d' % total_fn_img65)
            print('Sensibilidade 65 = %f' % sensibilidade_img65)
            print('Especificidade 65 = %f' % especificidade_img65)
            
            print('Verdadeiro positivo 75 = %d' % total_vp_img75)
            print('Verdadeiro negativo 75 = %d' % total_vn_img75)
            print('Falso positivo 75 = %d' % total_fp_img75)
            print('Falso negativo 75 = %d' % total_fn_img75)
            print('Sensibilidade 75 = %f' % sensibilidade_img75)
            print('Especificidade 75 = %f' % especificidade_img75)
            
            print('Verdadeiro positivo 85 = %d' % total_vp_img85)
            print('Verdadeiro negativo 85 = %d' % total_vn_img85)
            print('Falso positivo 85 = %d' % total_fp_img85)
            print('Falso negativo 85 = %d' % total_fn_img85)
            print('Sensibilidade 85 = %f' % sensibilidade_img85)
            print('Especificidade 85 = %f' % especificidade_img85)
            
            print('Verdadeiro positivo 90 = %d' % total_vp_img90)
            print('Verdadeiro negativo 90 = %d' % total_vn_img90)
            print('Falso positivo 90 = %d' % total_fp_img90)
            print('Falso negativo 90 = %d' % total_fn_img90)
            print('Sensibilidade 90 = %f' % sensibilidade_img90)
            print('Especificidade 90 = %f' % especificidade_img90)
            
            print('Candidatos encontrados = %d' % total_candidatos)
            
            print('Imagens doentes = %d' % imgs_doentes)
            print('Imagens saudaveis = %d' % imgs_saudaveis)
            
            print('Fim')
            
