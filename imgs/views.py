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
import random as r
import PIL.Image as Img
# from google.colab.patches import cv2_imshow
from math import log

lem = 0
map = {'a':0,'b':1,'c':2,'d':3,'e':4,'f':5,'g':6,'h':7,'i':8,'j':9,'k':10,'l':11,'m':12,'n':13,'o':14,'p':15,'q':16,'r':17,'s':18,'t':19,'u':20,'v':21,'w':22,'x':23,'y':24,'z':25,'A':26,'B':27,'C':28,'D':29,'E':30,'F':31,'G':32,'H':33,'I':34,'J':35,'K':36,'L':37,'M':38,'N':39,'O':40,'P':41,'Q':42,'R':43,'S':44,'T':45,'U':46,'V':47,'W':48,'X':49,'Y':50,'Z':51,' ':52,',':53,'.':54,'0':55,'1':56,'2':57,'3':58,'4':59,'5':60,'6':61,'7':62,'8':63,'9':64}
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


def dwt_showData(image_e):
  binary_data=""
  for i in range(0,image_e.shape[0]):
    for j in range(0,image_e.shape[1]):
      # print(type(pixel))
      p = messageToBinary(image_e[i][j])
      # print(p[-1])
      binary_data += p[-1]
  all_bytes = [ binary_data[i:i+8] for i in range(0,len(binary_data),8)]
  decoded_data=""
  for byte in all_bytes:
    # byte = int(byte)
    # print(int(byte,2))
    decoded_data = decoded_data + chr(int(byte,2))
    # print(decoded_data)
    if decoded_data[-5:] == "#####":
      break
  # print(all_bytes)
  return decoded_data[:-5]

def dwt_decode_text(cH_round_e):
  text=dwt_showData(cH_round_e)
  return text

# EMD
def EMD_Decode(image, encoded_image, n):
    print(n)
    Mes=[]
    ar = encoded_image
    im = np.array(ar)
    h,w = im.shape
    # sp = np.array((h,w))
    # for i in range(0,h):
    #   for j in range(0,w):
    #     sp[i][j] = (im[i][j][0] << 16) + (im[i][j][1]<<8) + (im[i][j][2])
    sp = np.int8(im)
    p = np.int8(image)
    r = p-sp
    i=0
    f=int
    f=int
    while(i<h-1):
        j=0
        f=0
        while(j<w):
            f = (f + (j+1)*sp[i][j])%(2*n+1)
            j = j + 1
        f = f%(2*n + 1)
        m=0
        s=0
        while(m<w):
            if(r[i][m] == 1):
                s = m
                d = (s + f)%(2*n + 1)
            # print("Mes is ",d)
                Mes.append(d)
            elif(r[i][m] == -1):
                s1 = (2*n+1)-m
                s2 = (2*n+1-m-w)
                while(s2<0):
                    s2 = s2 + (2*n+1)
                if(i != 0):
                    # print("Mes is ",(s1+f)%(2*n+1))
                    Mes.append((s1+f)%(2*n+1))
                else:
                    # print("Mes is ",(s1+f)%(2*n+1))
                    Mes.append((s1+f)%(2*n+1))
            m = m + 1
        i = i + 1
        Message = ""
    for i in Mes:
        for key in map:
            if(map[key] == i):
                Message = Message + str(key)
    print("Message is : ", Message)
    return Message

def EMD_hide_data(image, secret_message,n):
    ar = Img.open(image)
    im = np.int8(ar)
    h,w = im.shape
    # print(h,w)
    while(True):
        if(n<=64 and n>=0):
            n = n + 64
            break
        elif(n<0):
            n = 0 - n
        elif(n>64):
            break
    p = np.array(Img.open(image))
    s = 0
    d=[]
    xi = secret_message
    for c in range(len(xi)):
        d.append(map[xi[c]])
    i=0
    f = int
    while(i<(len(d))):
        j=0
        f=0
        while(j<w):
            f = (f + (j+1)*im[i][j])%(2*n+1)
            j = j + 1
        f = f%(2*n + 1)
        s = int((d[i]-f)%(2*n+1))
        while(s<0):
            s = s + (2*n+1)
        if(s <= n):
            p[i][s] = p[i][s] + 1
        elif(s > n):
            if(2*n+1-s>w):
                if( p[i+1][2*n+1-s-w] == 0 ):
                    p[i+1][2*n+1-s-w] = 255
                else:
                    p[i+1][2*n+1-s-w] = p[i+1][2*n+1-s-w] - 1
            else:
                if( p[i][(2*n+1)-s] == 0 ):
                    p[i][(2*n+1)-s] = 255
                else:
                    p[i][(2*n+1)-s] = p[i][(2*n+1)-s]-1
        i = i + 1
    # cv2.imwrite('Embedded.png',p)
    return p
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
            request.session['emd_original_image'] = img.name
            request.session['dlen'] = lem
            request.session['n_val'] = r.randint(1,500)
            n_val = request.session['n_val']

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
            # Here comes the embedding algorithm
            elif choice == 'emd':
                print("emd")
                print(data['image'])
                image = Img.open(data['image'])
                print("n_val is ", n_val)
                print("image given to ... is", image)
                encoded_image = EMD_hide_data(data['image'], data['msg'],n_val)
                cv2.imwrite("media/images/emd_hidden.png", encoded_image)
                data['image'] = "media/images/"+"emd_hidden.png"

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
        img = request.FILES.get('Img')
        choice = request.POST.get('choice')
        print(choice)
        x = str(request.session.get('emd_original_image'))
        emd_image="media/images/"+x
        emd_I = Img.open(emd_image)
        imgn = img.name
        stego = "media/images/"+imgn
        # print(stego)
        dlen = request.session.get('dlen')
        n_val=request.session.get('n_val')
        # print(dlen)
        # request.session['emd_original_image'] = img.name
        embedded_image = cv2.imread(stego)
        emd = Img.open(stego)
        # decoded_data=lsb_showData(embedded_image)
        # print("Decoded data in string : ",decoded_data)
        data = {}
        # data['msg']=decoded_data
        data['Img'] = stego

        if choice == '0':
            print("LSB")
            decoded_data = lsb_showData(embedded_image)
            data['msg'] = decoded_data

        elif choice == '1':
            print("DWT")
        
        elif choice == '2':
            print("PVD")
            decoded_data = pvd_decode_data(embedded_image, dlen)
            data['msg'] = decoded_data

        elif choice == '3':
            print("EMD")
            decoded_data = EMD_Decode(emd,emd_I,n_val)
            print("Decoded data in string : ", decoded_data)
            data = {}
            data['msg'] = decoded_data
            data['Img'] = stego
        return render(request, 'Actualmsg.html', {'data': data})

    else:
        return render(request, 'extraction.html')
