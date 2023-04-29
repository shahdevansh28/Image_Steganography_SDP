from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from django.shortcuts import render, redirect
from .forms import *
from os import read
import cv2
import numpy as np
import math
import pywt
import matplotlib.pyplot as plt
import matplotlib.image as img
import types
# from google.colab.patches import cv2_imshow
from math import log

lem = 0

# Create your views here.


def messageToBinary(message):
    if type(message) == str:
        return ''.join([format(ord(i), "08b")for i in message])
    elif type(message) == bytes or type(message) == np.ndarray:
        return [format(i, "08b")for i in message]
    elif type(message) == int or type(message) == np.uint8:
        return format(message, "08b")
    else:
        raise TypeError("Input type not supported")


def dwt_messageToBinary(message):
    if type(message) == str:
        return ''.join([format(ord(i), "08b")for i in message])
    elif type(message) == bytes or type(message) == np.ndarray:
        return [format(i, "08b")for i in message]
    elif type(message) == int or type(message) == np.int8:
        return format(message, "08b")
    else:
        raise TypeError("Input type not supported")


def nearestperfectsquare(limit):
    sqrt = math.sqrt(limit)
    closest = [math.floor(sqrt)**2, math.ceil(sqrt)**2]

    if ((abs(limit-math.floor(sqrt)**2)) < (abs(limit-math.ceil(sqrt)**2)-1)):
        return math.floor(sqrt)**2
    else:
        return math.ceil(sqrt)**2


def findm(n, d, m):
    if (d < (n*n)+n-2**m):
        # first subrange
        return m+1
    else:
        return m


def readmbytes(message, m, data_index):
    sm = []
    if (data_index+m < len(message)):
        for i in range(data_index, data_index+m):
            sm.append(message[i])
    else:
        for i in range(data_index, len(message)):
            sm.append(message[i])
    return sm


def checkcondition(d, n, m):
    if (d < (n*n)+n-2**m):
        return 1
        # first subrange
    else:
        # second subrange
        return 0


def split(list_a, chunk_size):

    for i in range(0, len(list_a), chunk_size):
        yield list_a[i:i + chunk_size]

# LSB
def lsb_hidedata(image, secret_message):
    n_bytes = image.shape[0]*image.shape[1]*3 // 8

    if len(secret_message) > n_bytes:
        raise ValueError("insufficient bytes,need bigger image or less data")

    secret_message += "#####"
    data_index = 0
    binary_secret_message = messageToBinary(secret_message)
    data_len = len(binary_secret_message)

    for values in image:
        for pixel in values:
            r, g, b = messageToBinary(pixel)

            if data_index < data_len:
                pixel[0] = int(r[:-1]+binary_secret_message[data_index], 2)
                data_index += 1
            if data_index < data_len:
                pixel[1] = int(g[:-1]+binary_secret_message[data_index], 2)
                data_index += 1
            if data_index < data_len:
                pixel[2] = int(b[:-1]+binary_secret_message[data_index], 2)
                data_index += 1

            if data_index >= data_len:
                break
    return image


def lsb_showData(image):
    binary_data = ""
    for values in image:
        for pixel in values:
            r, g, b = messageToBinary(pixel)
            binary_data += r[-1]
            binary_data += g[-1]
            binary_data += b[-1]
    all_bytes = [binary_data[i:i+8] for i in range(0, len(binary_data), 8)]

    decoded_data = ""
    for byte in all_bytes:
        decoded_data += chr(int(byte, 2))
        if decoded_data[-5:] == "#####":
            break
    return decoded_data[:-5]

# DWT
def dwt_hidedata(image, secret_message):
    n_bytes = image.shape[0]*image.shape[1] // 8
    if len(secret_message) > n_bytes:
        raise ValueError("insufficient bytes,need bigger image or less data")

    secret_message += "#####"
    data_index = 0
    binary_secret_message = dwt_messageToBinary(secret_message)
    data_len = len(binary_secret_message)

    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            p = dwt_messageToBinary(image[i][j])
            if data_index < data_len:
                image[i][j] = int(p[:-1]+binary_secret_message[data_index], 2)
                data_index += 1
            if data_index >= data_len:
                break
    return image

# EMD
# Put your functions related to the algorithm


# PVD
def pvd_hidedata(image, secret_message):
    # print("image give to ... is ",image)
    n_bytes = image.shape[0]*image.shape[1]*3

    if len(secret_message) > n_bytes:
        raise ValueError("insufficient bytes,need bigger image or less data")

    distance1 = 0
    count = 0
    data_index = 0
    secret_message = messageToBinary(secret_message)
    data_len = len(secret_message)
    difference = []
    print("Encoding : ")
    for i in range(0, image.shape[0]-1):
        if (data_index >= data_len):
            break
        for j in range(0, image.shape[1]):
            if (data_index >= data_len):
                break
            for x in range(0, 3):
                if (data_index >= data_len):
                    break
            # print(abs(i-j))
                if (image[i][j][x] >= image[i+1][j][x]):
                    distance = (abs(image[i][j][x]-image[i+1][j][x]))
                else:
                    distance = (abs(image[i+1][j][x]-image[i][j][x]))

                n = nearestperfectsquare(distance)
                n = int(math.sqrt(n))
                if (i == 0 and j == 0 and x == 0):
                    print((image[i][j][x], image[i+1][j][x], n))
                if (distance < 2):
                    m_bytes_of_data = readmbytes(secret_message, 1, data_index)
                    if (secret_message[data_index] == '0'):
                        distance1 = 0
                    else:
                        distance1 = 1
                    # print(i,j,x,data_index,m_bytes_of_data)
                    data_index = data_index+1
                elif (n == 2):
                    m = 2
                    m_bytes_of_data = readmbytes(secret_message, m, data_index)
                    ranges = []
                    for k in range(2, 6):
                        ranges.append(k)
                    for k in range(0, len(ranges)):
                        s = format(ranges[k], "08b")
                        s1 = []
                        for m1 in range(8-m, 8):
                            s1.append(s[m1])
                        if (s1 == m_bytes_of_data):
                            distance1 = ranges[k]
                            # print(i,j,x,data_index,distance1,m_bytes_of_data)
                            data_index = data_index+m
                            break
                else:
                    n = nearestperfectsquare(distance)
                    n = int(math.sqrt(n))
                    if distance < 240:
                        if (n > 0 and data_index < data_len):
                            m = (math.floor(math.log2(2*n)))

                            if (1):

                                m = m+1
                                m_bytes_of_data = readmbytes(
                                    secret_message, m, data_index)
                                ranges = []
                                for k in range(n*n-n, n*n+n-2**(m-1)):
                                    ranges.append(k)
                                if (i == 0 and j == 0 and x == 1):
                                    print(ranges, m_bytes_of_data)
                                for k in range(0, len(ranges)):
                                    s = format(ranges[k], "08b")
                                    s1 = []
                                    for m1 in range(8-m, 8):
                                        s1.append(s[m1])

                                    if (s1 == m_bytes_of_data):
                                        distance1 = ranges[k]
                                        data_index += (m)
                                        # print("1",s1,m_bytes_of_data,distance1,i,j,x)
                                        break

                            if (distance1 == 0):
                                m = m-1
                                ranges = []
                                for k in range(n*n+n-2**m, n*n+n):
                                    ranges.append(k)

                                m_bytes_of_data.pop()
                                if (i == 0 and j == 0 and x == 1):
                                    print(ranges, m_bytes_of_data)
                                for k in range(0, len(ranges)):
                                    s = format(ranges[k], "08b")
                                    s1 = []
                                    for m1 in range(8-m, 8):
                                        s1.append(s[m1])
                                    if (s1 == m_bytes_of_data):
                                        distance1 = ranges[k]
                                        data_index += (m)
                                        # print("2",s1,m_bytes_of_data,distance1,i,j,x)
                                        break
                    else:
                        m = 4
                        m_bytes_of_data = readmbytes(
                            secret_message, m, data_index)
                        # print(i,j,x,data_index,m_bytes_of_data)
                        data_index = data_index+m
                        byte1 = ''.join(str(e) for e in m_bytes_of_data)
                        distance1 = 240+int(byte1, 2)
                dif = abs(distance1-distance)

                if (distance1 <= distance and image[i][j][x] < image[i+1][j][x]):
                    image[i][j][x] = image[i][j][x]+(math.ceil(dif/2))
                    image[i+1][j][x] = image[i+1][j][x]-(math.floor(dif/2))
                    distance1 = 0

                elif (distance1 <= distance and image[i][j][x] >= image[i+1][j][x]):
                    image[i][j][x] = image[i][j][x]-(math.ceil(dif/2))
                    image[i+1][j][x] = image[i+1][j][x]+(math.floor(dif/2))
                    distance1 = 0

                elif (distance1 > distance and image[i][j][x] < image[i+1][j][x]):
                    image[i][j][x] = image[i][j][x]-(math.ceil(dif/2))
                    image[i+1][j][x] = image[i+1][j][x]+(math.floor(dif/2))
                    distance1 = 0

                else:
                    if (image[i][j][x] == 255):
                        # special case pixel value can't be more than 255
                        image[i][j][x] = image[i][j][x]-1
                    else:
                        image[i][j][x] = image[i][j][x]+(math.ceil(dif/2))
                    image[i+1][j][x] = image[i+1][j][x]-(math.floor(dif/2))
                    distance1 = 0

    return image


def pvd_decode_data(image, length):
    print("Decoding : ")
    distance1 = 0
    decoded_data = []
    count = 0
    data_index = length
    difference = []
    for i in range(0, image.shape[0]-1):
        if (data_index <= 0):
            break
        for j in range(0, image.shape[1]):
            if (data_index <= 0):
                break
            for x in range(0, 3):
                if (image[i][j][x] >= image[i+1][j][x]):
                    distance1 = (abs(image[i][j][x]-image[i+1][j][x]))
                    # print(distance1)
                else:
                    distance1 = (abs(image[i+1][j][x]-image[i][j][x]))
                    # print(image[i][j][x]-image[i+1][j][x])
                n = nearestperfectsquare(distance1)
                if (n == 0):
                    n = 1
                n = int(math.sqrt(n))
                if distance1 < 240:
                    if (n > 0 and data_index > 0):
                        if (n == 2):
                            m = 2
                            binary_dist = messageToBinary(distance1)

                            dec = []
                            for k in range(8-m, 8):
                                decoded_data.append(binary_dist[k])
                                dec.append(binary_dist[k])
                            data_index = data_index-m
                            # print(i,j,x,distance1,m,dec)
                        else:
                            m = (math.floor(math.log2(2*n)))
                            if (checkcondition(distance1, n, m)):
                                # first subrange
                                m = m+1
                                binary_dist = messageToBinary(distance1)
                                dec = []
                                for k in range(8-m, 8):
                                    decoded_data.append(binary_dist[k])
                                    dec.append(binary_dist[k])
                                data_index = data_index-m
                                # print(i,j,x,distance1,m,dec)

                            else:
                                # second subrange

                                binary_dist = messageToBinary(distance1)
                                dec = []
                                for k in range(8-m, 8):
                                    decoded_data.append(binary_dist[k])
                                    dec.append(binary_dist[k])
                                data_index = data_index-m
                                # print(i,j,x,distance1,m,dec)

                else:
                    if (data_index > 0):
                        binary_dist = messageToBinary(distance1)
                        # print(i,j,x,distance1,4)
                        for k in range(4, 8):
                            decoded_data.append(binary_dist[k])
                        data_index = data_index-4

    decoded = decoded_data

    decoded_data_in_string = ""
    decoded_data_in_bin = list(split(decoded, 8))
    print("Decoded data in binary : ", decoded_data_in_bin)
    for byte in decoded_data_in_bin:
        byte1 = ''.join(str(e) for e in byte)
        decoded_data_in_string += chr(int(byte1, 2))

    return decoded_data_in_string


def home(request):
    return render(request, 'Home.html')


def image_view(request):

    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES, request.POST)
        if form.is_valid():
            img = request.FILES.get('Img')
            message = request.POST.get('message')
            choice = request.POST.get('choices')
            print(img.name, message, choice)
            form.save()
            binary_data = messageToBinary(message)
            lem = len(binary_data)
            request.session['dlen'] = lem

            data = {}
            data['image'] = "media/images/"+img.name
            data['msg'] = message

            # LSB
            if choice == 'lsb':
                image = cv2.imread(data['image'])
                data['msg'] += " Using LSB"
                encoded_image = lsb_hidedata(image, data['msg'])
                cv2.imwrite("media/images/lsb_hidden.png", encoded_image)
                data['image'] = "media/images/"+"lsb_hidden.png"
                print("Lsb")

            # DWT
            elif choice == 'dwt':
                print("Dwt")
                image = cv2.imread(data['image'], 0)
                coeffs = pywt.dwt2(image, 'haar')
                cA, (cH, cV, cD) = coeffs
                cH_round = np.round(cH)
                cH_round = cH_round.astype('int8')
                data['msg'] += " Using DWT"
                dwt_hidedata(cH_round, data['msg'])

                coeffs_updated = cA, (cH_round, cV, cD)
                encoded_image = pywt.idwt2(coeffs_updated, 'haar')

                # image = cv2.imread(data['image'])
                cv2.imwrite("media/images/dwt_hidden.png", encoded_image)
                data['image'] = "media/images/"+"dwt_hidden.png"
            
            # PVD
            elif choice == 'pvd':
                print("Pvd")
                print(data['image'])
                image = cv2.imread(data['image'])
                print("image given to ... is", image)
                encoded_image = pvd_hidedata(image, data['msg'])
                cv2.imwrite("media/images/pvd_hidden.png", encoded_image)
                data['image'] = "media/images/"+"pvd_hidden.png"
            # EMD
            #Here comes the embedding algorithm
            elif choice == 'emd':
                print("emd")

        return render(request, 'stego.html', {'data': data})
    else:
        form = ImageForm()
        return render(request, 'embedding.html', {'form': form})


def success(request):
    return HttpResponse('successfully uploaded')


def stego(request, data):
    # print(data)
    return HttpResponse('successfully stego uploaded')


def extraction(request):
    
    if request.method == 'POST':
        img=request.FILES.get('Img')
        choice=request.POST.get('choice')
        print(choice)
        imgn=img.name
        stego="media/images/"+imgn
        # print(stego)
        dlen=request.session.get('dlen')
        # print(dlen)
        embedded_image=cv2.imread(stego)
        # decoded_data=lsb_showData(embedded_image)
        # print("Decoded data in string : ",decoded_data)
        data={}
        # data['msg']=decoded_data
        data['Img']=stego

        if choice == '0':
            print("LSB")
            decoded_data=lsb_showData(embedded_image)
            data['msg']=decoded_data
        
        elif choice == '1':
            print("DWT")

        elif choice == '2':
            print("PVD")
            decoded_data=pvd_decode_data(embedded_image,dlen)
            data['msg'] = decoded_data
        
        elif choice == '3':
            print("EMD")

        return render(request ,'Actualmsg.html', {'data' : data })

    else:
        return render(request, 'extraction.html')
