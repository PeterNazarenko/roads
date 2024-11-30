
# 14:45 31.10.2021
# Based on rm16.py

# 15:47 23.08.2021
# Road Monitor v.1.03
# Big and small file version
# Improved tracking algorithm
# OpenCV visualization

# 12:40 01.11.2024
# v.1.03.503 Saving ANN result to PNG, skeletizing of ANN result
# v.1.03.504 Digital filtering in every string
# v.1.03.505 Very Low frequency only

import sys         # for file operations
import h5py        # for HDF5 files
import numpy as np

import pandas as pd

from scipy.interpolate import InterpolatedUnivariateSpline

from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt

import cv2


activeKey=0 #1   # ключ HDF5-файла с реальными данными

Kscale=100000    # коэффициент нормализации данных на входе нейросети
outPorog=0.2  
step=8 #24 #8           # шаг сканирования по расстоянию    24 - проблемы с интерполяцией
stepT=1
Npolos=96 #24        # число спектральных полос, участвующих в обработке
binLevel=100 #128 #150 # порог отсечения шумов

Nix,Niy=201,1    # размеры входного массива нейросети
Ni=Nix*Niy
Nout=1           # размеры выходного массива нейросети

inp1=np.zeros(Ni)             # входной массив нейросети
w1=np.ones((Nout,Ni+1))       # массив весовых коэффицентов сети
layer1=np.zeros(Nout)         # выходной массив

koefPoly2=0.15 #0.5    # Пороговые коэффициенты полиномов старших степеней
koefPoly21=1 #0.25 #0.75
koefPoly3=0.2

def vvod(source,dest):            # функция ввода данных на вход нейросети (можно не использовать)
    for i in range(len(source)):
        dest[i]=source[i]

def work(inp,wa,out,alpha):       # работа нейросети (вход, веса, выход, коэффициент наклона активационной функции)
    for o in range(len(out)):
        sum=0.0
        for i in range(len(inp)):
            sum+=inp[i]*wa[o][i]
        sum+=wa[o][len(inp)]
        out[o]=1/(1+np.exp(-sum*alpha))

def LoadFromFile(fname):    # загрузка весовых коэффициентов обученной сети из файла
    f=open(fname,'rt')
    fileBuf=f.read()
    f.close()
    fileList=fileBuf.split()
    Ni=int(fileList[0])-1
    Nout=int(fileList[1])
    for i in range(Ni+1):
        for j in range(Nout):
            w1[j][i]=float(fileList[i*Nout+j+2])

fn='wgt1step.txt'  # файл с весовыми коэффициентами обученной сети
LoadFromFile(fn)   # загрузка весовых коэффициентов обученной сети из файла
fnout=None

def initData(fn,data,keys):  # функция получения служебной информации о HDF5-файле (число ключей и размерность массива данных)
    global fnout
    if len(sys.argv)>1:
        fn=sys.argv[1]
    else:
        fn='6960025_09-13-50_.hdf5'
        fnout='KrasnyjYar_list1.txt'
#        fn='Samara_21_06_12_23_statistics.hdf5'
#        fn='6960059_23-15-49_.hdf5'
#        fnout='KrasnyjYar_list3.txt'

    with h5py.File(fn,'r') as file:  # опрос ключей HDF5-файла
        for key in file.keys():
            keys.append(key)
    file.close()

    file=h5py.File(fn,'r')
    data=file.get(keys[activeKey])  # получение инфомациии о размерности и размерах массива данных
    t,x,f=data.shape
    file.close()

    return fn,data,keys,t,x,f


def procTime(fn,keys,out,x,t):  # Функция обработки данных (имя HDF5-файла, массив ключей в нём, выходной массив, длина участка, число тактов)
    N=x

    file=h5py.File(fn,'r')  # Открытие HDF5-файла
    data=file.get(keys[activeKey])

#    data2=data[0][:N]

    entry=np.zeros(x+200)             # одномерный массив блоков данных на входе нейросети, дополненный с обеих сторон по 100 метров

    for j in range(t):
        data2=data[j][:N]                 # чтение очередного блока из HDF5-файла (очередной такт)
#        print(data2.shape)  # debug (2400,96)
        for i in range(N):                   # просмотр всей дороги на очередном такте
            if data2[i][0]>binLevel:         # Бинаризация сигнала на входе нейросети
                data2[i][0]=Kscale
            else:
                data2[i][0]=0

            entry[i+100]=data2[i][0]/Kscale         # заполнение массива входов нейросети с одновременной нормализацией
        for i in range(0,len(entry)-201,step):      # просмотр всей дороги с заданным шагом на очередном такте
            work(entry[i:i+201:],
              w1,
              out[j][int(i/step)],2.0)  # работа нейросети
            if out[j][int(i/step)]>outPorog:
                out[j][int(i/step)]=1.
            else:
                out[j][int(i/step)]=0.
    file.close()

def reStep(mas,step):
  if step==1:
    return mas
  else:
    t,x,_=mas.shape
    mas2=np.zeros((t,x*step),np.uint8)
    for t1 in range(t):
      for x1 in range(x):
        for x2 in range(step):
          mas2[t1][x1*step+x2]=mas[t1][x1]
    return mas2

def impulseFiltH(mas):
  t,x,_=mas.shape
  for t1 in range(t):
    for x1 in range(1,x-1):
      if mas[t1][x1]==1 and mas[t1][x1-1]==0 and mas[t1][x1+1]==0:
        mas[t1][x1]=0

def impulseFiltV(mas):
  t,x,_=mas.shape
  for x1 in range(x):
    for t1 in range(1,t-1):
      if mas[t1][x1]==1 and mas[t1-1][x1]==0 and mas[t1+1][x1]==0:
        mas[t1][x1]=0

class Object:     # класс идентифицируемого объекта
  def __init__(self):
    self.id1=0    # Идентификатор объекта
    self.objID=0  # Групповой идентификатор
    self.t=0      # Текущий тактовый интервал
    self.x=0      # Текущая координата вдоль дороги (передний фронт импульса)
    self.dx=0     # dx/dt (dt=1)
    self.mdx=0    # Среднее значение dx/dt
    self.xMax=0   # Значение максимума сигнала, соответсвующее объекту (импульсу)
    self.x0=0     # Координаты начала отслеживания (расстояние)
    self.t0=0     # Координаты начала отслеживания (время)
    self.x1=0     # Задний фронт импульса
    self.te=0     # Координаты конца отслеживания (время)
    self.xe=0     # Координаты конца отслеживания (расстояние)
    self.xc=0     # center (x+x1)/2
    self.xc0=0    # first center
    self.xce=0    # last center
    self.coordList=[] # Список связанных координат
    self.listLen=0    # Длина списка
    self.mark=0   # служебная метка
    self.maxX=0   # максимальная координата конца импульса
    self.minX=0   # минимальная координата начала импульса
    self.long=0   # длина трека (после объединения треков может отличаться от te-t0)

  def add(self,id1,t,x):  # Ввод в объект выбранных данных
    self.id1=id1
    self.objID=id1
    self.t=t
    self.x=x

  def show(self):
#    print('id=',self.id1,self.t,self.x,self.xc,self.x1,end=' ')
    print('id=',self.id1,self.t,self.x,self.x1,end=' ')

  def show2(self):
    print('id=',self.id1,self.t0,self.xc0,self.t,self.xc,end=' ')
    print(self.coordList)

  def save(self,file):
    file.write(str(self.id1)+' ')
#    file.write(str(self.objID)+' ')
    file.write(str(self.t)+' ')
    file.write(str(self.x)+' ')
    file.write(str(self.dx)+' ')
    file.write(str(self.mdx)+' ')
    file.write(str(self.x0)+' ')
    file.write(str(self.t0)+' ')
    file.write(str(self.x1)+' ')
    file.write(str(self.te)+' ')
    file.write(str(self.xe)+' ')
    file.write(str(self.xc)+' ')
    file.write(str(self.xc0)+' ')
    file.write(str(self.xce)+' ')
    file.write(str(self.maxX)+' ')
    file.write(str(self.minX)+' ')
    file.write(str(self.listLen)+' ')
    file.write(str(self.long)+' ')
    for i in self.coordList:
      file.write(str(i[0])+' ')
      file.write(str(i[1])+' ')
      file.write(str(i[2])+' ')
      file.write(str(i[3])+' ')
      file.write(str(i[4])+' ')

# end of Object class

class listObjects:     # класс списка объектов
  def __init__(self):
    self.objList=[]    # собственно список
    self.Nobjects=0    # количество объектов

  def add(self,obj):
    self.objList.append(obj)
    self.Nobjects+=1

  def removeElement(self,data):
    self.objList.remove(data)
    self.Nobjects-=1

  def show(self):
    for i in self.objList:
      i.show()
    print()

  def show2(self):
    for i in self.objList:
      i.show2()
    print()

  def showGraph(self,data):      # Визуализация для отладки
    t,x,_=data.shape
    fig, ax = plt.subplots()
    y=np.zeros((t,x))
    for i in self.objList:
      for j in i.coordList: 
        y[j[0]][int(j[1])]=i.id1  # Трек каждого объекта окрашивается в цвет его номера
    ax.imshow(y)
    plt.show()
    return y

  def showGraph2(self,data):
    t,x,_=data.shape
    fig, ax = plt.subplots()
    y=np.zeros((t,x))
    for i in self.objList:
      for j in i.coordList: 
        if int(j[1])<x:
          y[j[0]][int(j[1])]=1  # difference with showGraph()
    ax.imshow(y)
    plt.show()
    return y

  def getAsArray(self,data):  # nepravil'no?
    t,x,_=data.shape
    y=np.zeros((t,x))
    for i in self.objList:
      y[i.t][int(i.xc)]=i.id1
    return y

  def saveList(self,fileName):
    file=open(fileName,'wt')
    file.write(str(self.Nobjects)+' ')
    for i in self.objList:
      i.save(file)

  def readFile(self,fileName):
    fn=open(fileName,'rt')
    txtBuf=fn.read()
    fn.close()
    List=txtBuf.split()
    nn1=int(List[0])
    count=1
    for i in range(nn1):
      obj=Object()
      obj.id1=int(List[count]); count+=1
      obj.objID=int(List[count]); count+=1
      obj.t=int(List[count]); count+=1
      obj.x=int(List[count]); count+=1
      obj.dx=float(List[count]); count+=1
      obj.mdx=float(List[count]); count+=1
      obj.x0=float(List[count]); count+=1
      obj.t0=int(List[count]); count+=1
      obj.x1=int(List[count]); count+=1
      obj.te=int(List[count]); count+=1
      obj.xe=float(List[count]); count+=1
      obj.xc=float(List[count]); count+=1
      obj.xc0=float(List[count]); count+=1
      obj.xce=float(List[count]); count+=1
      obj.maxX=float(List[count]); count+=1
      obj.minX=float(List[count]); count+=1
      obj.listLen=int(List[count]); count+=1
      obj.long=int(List[count]); count+=1
      obj.coordList=[]
      for i in range(obj.listLen):
        buf1=int(List[count]); count+=1
        buf2=float(List[count]); count+=1
        buf3=float(List[count]); count+=1
        buf4=float(List[count]); count+=1
        buf5=float(List[count]); count+=1
#        obj.coordList.append([buf1,buf2,buf3])
        obj.coordList.append([buf1,buf2,buf3,buf4,buf5])
      self.add(obj)

  def listToArray(self,data):
    t,x,_=data.shape
    y=np.zeros((t,x))
    for i in self.objList:
      for j in i.coordList: 
        y[j[0]][int(j[1])]=1
    return y

  def truncateListLeft(self,level):
    for obj in self.objList:
      removeSign=0
      i=0
      while i<obj.listLen:
        if obj.coordList[i][1]<level:
          data=obj.coordList[i]
          if obj.coordList.count(data)>0:
            obj.coordList.remove(data)
            obj.listLen-=1
            if obj.listLen<=0:
              removeSign=1
              break
            i-=1
        i+=1
      if removeSign==1:
        self.removeElement(obj)

  def showGraphOCV(self,data,b,g,r,mode):
    c1=7
    c2=56
    c3=192
    count=1
    t,x,_=data.shape
    y=np.zeros((t,x,3))
    for i in self.objList:
      if mode==1:
        if count==1:
          r1=i.id1 & c1
          r1<<=5
          g1=i.id1 & c2
          g1<<=5
          b1=i.id1 & c3
          b1=b1<<6
        elif count==2:
          b1=i.id1 & c1
          b1<<=5
          r1=i.id1 & c2
          r1<<=5
          g1=i.id1 & c3
          g1=g1<<6
        elif count==3:
          g1=i.id1 & c1
          g1<<=5
          b1=i.id1 & c2
          b1<<=5
          r1=i.id1 & c3
          r1=r1<<6
      elif mode==0:
        r1=r
        g1=g
        b1=b
      x1=int(i.coordList[0][1])
      y1=i.coordList[0][0]
      for j in i.coordList[1::]:
#        y[j[0]][int(j[1])]=1
        x2=int(j[1])
        y2=j[0]
        cv2.line(y,(x1,y1),(x2,y2),(b1,g1,r1),1)
        x1=x2
        y1=y2
      count+=1
      if count>3:
        count=1
    return y

# end of listObjects class


def sign(num):  # Функция расчёта знака аргумента
#    return -1 if num < 0 else 1
    sign1=0
    if num<0:
        sign1=-1
    elif num>0:
        sign1=1
    else:
        sign1=0
    return sign1


def procData(data,Lo):  # Функция обработки данных с выхода нейросети после сканирования (массив выходов сети, список объектов)
    t,x,_=data.shape    # Размеры массива данных
#    print(data.shape)  # Debug

    count=0     # Счётчик числа объектов
    findSign=0  # Признак успеха поиска
    Xst=0 #10
    Xend=x-1 #x-3
#    for j in range(t):      # Перебор тактовых интервалов
    for j in range(1,t):    # Перебор тактовых интервалов
        for i in range(x):  # Перебор расстояний

            if i<Xst or i>Xend: data[j][i][0]=0.  # 
            if i>0 and i<x-1:  # Предварительная импульсная фильтрация
                if data[j][i-1][0]==0. and data[j][i+1][0]==0. and data[j][i][0]==1.:
                    data[j][i][0]=0.
            if i==0 and data[j][i][0]==1 and data[j][i+1][0]==0:  # Обработка левого края массива
                data[j][i][0]=0

            if (i>Xst and data[j][i-1][0]==0. and data[j][i][0]==1.) or (i==Xst and data[j][i][0]==1.):  # Положительный перепад импульса с выхода НС
                findSign=0   # Признак успеха поиска
                a=Object()   # Локальный объект,описывающий положение транспортного средства вдоль дороги
                a.x=i        # Текущая координата вдоль дороги (передний фронт ипульса)
                a.t=j        # Текущий тактовый интервал
                a.x0=i       # Координаты начала отслеживания (расстояние)
                a.t0=j       # Координаты начала отслеживания (время)
                a.xe=i       # Координаты конца отслеживания (расстояние)
                a.te=j       # Координаты конца отслеживания (время)
                a.dx=0       # Градиент dx/dt
                a.mdx=0      # Средний градиент
                a.long=0     # Длина трека
            if (i>Xst and data[j][i-1][0]==1. and data[j][i][0]==0.) or (i==x-1 and data[j][i][0]==1.):  # Отрицательный перепад
                if i<x-1:     # Обработка правого края массива
                    a.x1=i-1  # Задний фронт импульса
                else:
                    a.x1=i
                a.xc=(a.x+a.x1)/2     # Центр импульса

                if Lo.Nobjects==0 or j==0:    # Идентифицированных объектов ещё нет
                    count+=1          # Новый объект
                    a.add(count,j,i)  # Ввод координат в объект
                    a.t0=a.t
#                    a.x0=a.x
                    a.xc0=a.xc        # Начальная x-координата центра импульса
                    a.x0=a.xc0
                    a.xce=a.xc
                    a.dx=0            # Градиент dx/dt
                    a.mdx=0           # Средний градиент
                    a.maxX=a.x1
                    a.minX=a.x
                    a.long=0
#                    a.coordList.append([j,a.xc,a.dx])  # Добавление координат в связанный с объектом список координат
                    a.coordList.append([j,a.xc,a.dx,a.x,a.x1])  # Добавление координат в связанный с объектом список координат
                    a.listLen+=1                  # Увеличение размера списка
                    Lo.add(a)             # Добавление объекта в список
                else:                     # Хотя бы один объект существует
                    for k in Lo.objList:  # Поиск в списке объектов
                        if k.t>j:         # Поиск выполняется только до текущего момента времени в HDF5-файле
                            break
                        elif k.t>j-2:     # Поиск начинается только с предыдущего момента времени в HDF5-файле
                            if ((k.x>a.x and k.x1>a.x1 and a.x1>k.x) or   # Проверка перекрытия выходов нейросети
                                (k.x>a.x and k.x1>=a.x1 and a.x1>k.x) or  # для соседних тактов 
                                (k.x>=a.x and k.x1>a.x1 and a.x1>k.x) or  # (проверяются разные случаи
                                (k.x>a.x and k.x==a.x1 and k.x1>a.x1) or  #  перекрытия)
                                (k.x==a.x and k.x1==a.x1) or
                                (k.x<a.x and k.x1<a.x1 and a.x<k.x1) or
                                (k.x==a.x and k.x1<a.x1 and a.x<k.x1) or
                                (k.x<a.x and k.x1==a.x1 and a.x<k.x1) or
                                (k.x<a.x and k.x1<a.x1 and a.x==k.x1) or
                                (k.x>a.x and k.x1<a.x1) or
                                (k.x<a.x and k.x1>a.x1) or
                                 k.x==a.x1+1 or k.x1==a.x-1):  # new line

##                                if k.xc>a.xc:
##                                    a.dx=a.xc-k.xc  # dx/dt
##                                if k.xc<a.xc:
##                                    a.dx=a.xc-k.xc  # dx/dt

#                                '''
                                if ((k.x>a.x and k.x1>a.x1 and a.x1>k.x and k.xc>a.xc) or
                                    (k.x>a.x and k.x1>=a.x1 and a.x1>k.x and k.xc>a.xc) or
                                    (k.x>=a.x and k.x1>a.x1 and a.x1>k.x and k.xc>a.xc) or
                                    (k.x>a.x and k.x==a.x1 and k.x1>a.x1 and k.xc>a.xc) or k.x==a.x1+1):
                                    a.dx=a.xc-k.xc  # dx = -1

                                if ((k.x<a.x and k.x1<a.x1 and a.x<k.x1 and k.xc<a.xc) or
                                    (k.x==a.x and k.x1<a.x1 and a.x<k.x1 and k.xc<a.xc) or
                                    (k.x<a.x and k.x1==a.x1 and a.x<k.x1 and k.xc<a.xc) or
                                    (k.x<a.x and k.x1<a.x1 and a.x==k.x1 and k.xc<a.xc) or k.x1==a.x-1):
                                    a.dx=a.xc-k.xc  # dx = +1
                
                                if ((k.x==a.x and k.x1==a.x1) or
                                    (k.x>a.x and k.x1<a.x1) or
                                    (k.x<a.x and k.x1>a.x1)):
                                    a.dx=k.dx #'''
#                                a.dx=a.xc-k.xc

##                                if a.dx==k.coordList[k.listLen-1][2] or k.coordList[k.listLen-1][2]==0:

#                                if sign(a.dx)==sign(k.coordList[k.listLen-1][2]) or k.coordList[k.listLen-1][2]==0:  # +++

                                if((sign(a.dx)==sign(k.coordList[k.listLen-1][2]) or k.coordList[k.listLen-1][2]==0) or  # +++
                                   (sign(a.dx)!=sign(k.coordList[k.listLen-1][2]) and sign(a.dx)==sign(k.mdx)) or  # new line
                                   (sign(a.dx)!=sign(k.coordList[k.listLen-1][2]) and k.mark==1)):  # new line
#                                    k.coordList.append([a.t,a.xc,a.dx])  # Подходящий объект найден, добавление координат
                                    k.coordList.append([a.t,a.xc,a.dx,a.x,a.x1])  # Подходящий объект найден, добавление координат
                                    k.listLen+=1  # Увеличение размера списка
                                    k.long=k.listLen
                                    k.x=a.x       # Коррекция текущих координат найденного объекта
                                    k.x1=a.x1     # для правильной работы в следующей итерации
                                    k.xc=a.xc
                                    k.t=a.t
                                    k.te=k.t      # Координаты конца отслеживания (время)
                                    k.xe=k.xc     # Координаты конца отслеживания (расстояние)
                                    k.xce=k.xe
                                    k.dx=a.dx     # Градиент
                                    k.mdx=(k.mdx*(k.listLen-1)+k.dx)/k.listLen
                                    findSign=1  # Признак успеха поиска
                                    k.mark=0    # Сброс метки
                                    if k.x1>k.maxX:
                                      k.maxX=k.x1
                                    if k.x<k.minX:
                                      k.minX=k.x
                                    break       # Выход из поиска в списке объектов
                                else:
                                    k.mark=1    # Установки метки

                    if findSign==0:         # Новый объект при неудачном предыдущем поиске
                        count+=1            # Новый номер объекта
                        a.add(count,j,a.x)  # Ввод координат в объект
                        a.t0=a.t            # Координаты начала отслеживания (расстояние)
                        a.x0=a.x            # Координаты начала отслеживания (время)
                        a.xe=i              # Координаты конца отслеживания (расстояние)
                        a.te=j              # Координаты конца отслеживания (время)
                        a.xc0=a.xc          # Начальная x-координата центра импульса
                        a.xce=a.xc          # Конечная x-координата центра импульса
                        a.dx=0              # Градиент dx/dt
                        a.mdx=0             # Средний градиент
                        a.maxX=a.x1
                        a.minX=a.x
                        a.long=0
#                        a.coordList.append([j,a.xc,a.dx])  # Добавление координат в связанный с объектом список координат
                        a.coordList.append([j,a.xc,a.dx,a.x,a.x1])  # Добавление координат в связанный с объектом список координат
                        a.listLen+=1        # Увеличение размера списка
                        Lo.add(a)           # Добавление объекта в список

    return count  # Возврат числа идентифицированных объектов


def extrp(Lo,data,eps):
    t0,n,_=data.shape
    for obj in Lo.objList[::-1]:         # Просмотр всех объектов  # +++
#        input('D:')
      if obj.listLen>3:
        t=[]
        y=[]
        if obj.listLen<n: #t0:           # если длина трека у текущего объекта меньше заданной
            if obj.listLen<9:  # new
              for point in obj.coordList:  # Формирование массивов для работы функции интерполяции/экстраполяции
                t.append(point[0])
                y.append(point[1])
#            '''
            else:                                 # new
                t.append(obj.coordList[0][0])
                y.append(obj.coordList[0][1])
                t.append(obj.coordList[int(obj.listLen/4)][0])
                y.append(obj.coordList[int(obj.listLen/4)][1])
                t.append(obj.coordList[int(obj.listLen/2)][0])
                y.append(obj.coordList[int(obj.listLen/2)][1])
                t.append(obj.coordList[int(obj.listLen*3/4)][0])
                y.append(obj.coordList[int(obj.listLen*3/4)][1])
                t.append(obj.coordList[obj.listLen-1][0])
                y.append(obj.coordList[obj.listLen-1][1])
#            '''
#        print('t=',t)  # debug
#        print('y=',y)  # debug
        x3=np.arange(obj.te,t0)  # массив аргументов для экстраполяции (координата-время конца трека)
#        print('obj.te=',obj.te)  # debug
#        print('x3=',x3)  # debug
        order = 1 #2 #3                        # порядок функции
        print('id=',obj.id1,'t=',t,'y=',y)
        if len(t)==0 or len(y)==0:
          continue
        s1 = InterpolatedUnivariateSpline(t,y,k=order)  # функция интерполяции/экстраполяции
        s2 = InterpolatedUnivariateSpline(t,y,k=order+1)  # функция интерполяции/экстраполяции
        s3 = InterpolatedUnivariateSpline(t,y,k=order+2)  # функция интерполяции/экстраполяции
        y31=s1(x3)                       # экстраполированные значения
        y32=s2(x3)                       # экстраполированные значения
        y33=s3(x3)                       # экстраполированные значения
#        print('y31=',y31)  # debug
        for obj2 in Lo.objList[::-1]:    # поиск объекта с начальными координатами на продолжении этой линии
#       for obj2 in Lo.objList:
            if obj.id1!=obj2.id1:
                if obj2.t0>obj.te:
                    i=0
                    ii1=y31[i]
                    ii2=y32[i]
                    ii3=y33[i]
                    tt=x3[i]
#                    print('obj=',obj2.id1,obj2.xc0,'ii=',ii1,'len',len(y31)-1,'y3',y3[len(y3)-1])  # debug
                    while i<len(y31): #ii1<=y31[len(y31)-1]:
                        ii1=y31[i]
                        ii2=y32[i]
                        ii3=y33[i]
                        tt=x3[i]
                        if ((np.sqrt((obj2.t0-obj.te)**2+(obj2.xc0-obj.xce)**2)<8) and
                            abs(obj2.mdx-obj.mdx)<0.15 and sign(obj.mdx)==sign(obj2.mdx)):
                            obj.coordList.extend(obj2.coordList)
                            obj.te=obj2.te
                            obj.xce=obj2.xce
                            obj.xe=obj2.xe
#                            obj.listLen=obj2.te-obj.t0+1  # ???
                            obj.listLen+=obj2.listLen  # ???
                            Lo.removeElement(obj2)
                            break
#                        if abs(obj2.xc0-ii1)<=eps and obj2.t0==tt and sign(obj.dx)==sign(obj2.dx):
                        elif (abs(obj2.xc0-ii1)<=eps and obj2.t0==tt and sign(obj.mdx)==sign(obj2.mdx) and
                            abs(obj2.t0-obj.te)<int(n*3/4)):
#                            print('+')  # debug

#                            for ti in range(1,len(x3)):  # Verification visualization
#                                obj.coordList.append([x3[ti],y31[ti]])
#                                if abs(obj2.xc0-y31[ti])<=eps and obj2.t0==x3[i]:
#                                    break

                            obj.coordList.extend(obj2.coordList)
                            obj.te=obj2.te
                            obj.xce=obj2.xce
                            obj.xe=obj2.xe
#                            obj.listLen=obj2.te-obj.t0+1  # ???
                            obj.listLen+=obj2.listLen  # ???
                            Lo.removeElement(obj2)
                            break
                        '''
                        elif abs(obj2.xc0-ii2)<=eps and obj2.t0==tt and sign(obj.mdx)==sign(obj2.mdx):
                            obj.coordList.extend(obj2.coordList)
                            obj.te=obj2.te
                            obj.xce=obj2.xce
                            obj.xe=obj2.xe
#                            obj.listLen=obj2.te-obj.t0+1  # ???
                            obj.listLen+=obj2.listLen  # ???
                            Lo.removeElement(obj2)
                            break
                        elif abs(obj2.xc0-ii3)<=eps and obj2.t0==tt and sign(obj.mdx)==sign(obj2.mdx):
                            obj.coordList.extend(obj2.coordList)
                            obj.te=obj2.te
                            obj.xce=obj2.xce
                            obj.xe=obj2.xe
#                            obj.listLen=obj2.te-obj.t0+1  # ???
                            obj.listLen+=obj2.listLen  # ???
                            Lo.removeElement(obj2)
                            break
#                        '''
                        i+=1
                else:
                    break

# end of extrp() function


def extrp2(Lo,data,eps):
    t0,n,_=data.shape
    for obj in Lo.objList[::-1]:         # Просмотр всех объектов  # +++
      if obj.listLen>3:
        t=[]
        y=[]
        if obj.listLen<n: #t0:           # если длина трека у текущего объекта меньше заданной
            if obj.listLen<9:
              for point in obj.coordList:  # Формирование массивов для работы функции интерполяции/экстраполяции
                t.append(point[0])
                y.append(point[1])
            else:
                t.append(obj.coordList[0][0])
                y.append(obj.coordList[0][1])
                t.append(obj.coordList[int(obj.listLen/4)][0])
                y.append(obj.coordList[int(obj.listLen/4)][1])
                t.append(obj.coordList[int(obj.listLen/2)][0])
                y.append(obj.coordList[int(obj.listLen/2)][1])
                t.append(obj.coordList[int(obj.listLen*3/4)][0])
                y.append(obj.coordList[int(obj.listLen*3/4)][1])
                t.append(obj.coordList[obj.listLen-1][0])
                y.append(obj.coordList[obj.listLen-1][1])

        x3=np.arange(obj.te,t0)  # массив аргументов для экстраполяции (координата-время конца трека)
        order = 3                        # порядок функции
        s1 = InterpolatedUnivariateSpline(t,y,k=order)  # функция интерполяции/экстраполяции
        y3=s1(x3)                        # экстраполированные значения
        for obj2 in Lo.objList[::-1]:    # поиск объекта с начальными координатами на продолжении этой линии
            if obj.id1!=obj2.id1:
                if obj2.t0>obj.te:
                    i=0
                    ii=y3[i]
                    tt=x3[i]
                    while i<len(y3):
                        ii=y3[i]
                        tt=x3[i]
                        if abs(obj2.xc0-ii)<=eps and obj2.t0==tt:
                            print('+')  # debug

#                            for ti in range(1,len(x3)):  # Verification visualization
#                                obj.coordList.append([x3[ti],y3[ti]])
#                                if abs(obj2.xc0-y3[ti])<=eps and obj2.t0==x3[i]:
#                                    break

                            obj2.id1=obj.id1
                            break
                        i+=1
                else:
                    break
# end of extrp2() function


def myPolyFit(x1,x2,y1,y2):
    k=(y2-y1)/(x2-x1)  # y(x)=x0+k*x
    x0=y1-k*x1  # x0=y(x)-k*x
    return k,x0


def extrp3(Lo,data,eps):
    t0,n,_=data.shape
    for obj in Lo.objList[::-1]:         # Просмотр всех объектов  # +++
      lenX=abs(obj.xce-obj.xc0)
      if obj.listLen>3:
        if lenX<9:
          A,B,C,D=obj.t0,obj.te,obj.xc0,obj.xce
        elif lenX<13:
          A=obj.coordList[int(obj.listLen/5)][0]    # t0
          B=obj.coordList[int(obj.listLen*4/5)][0]  # te
          C=obj.coordList[int(obj.listLen/5)][1]    # xc0
          D=obj.coordList[int(obj.listLen*4/5)][1]  # xce
        else:
          A=obj.coordList[int(obj.listLen/6)][0]    # t0
          B=obj.coordList[int(obj.listLen*5/6)][0]  # te
          C=obj.coordList[int(obj.listLen/6)][1]    # xc0
          D=obj.coordList[int(obj.listLen*5/6)][1]  # xce

        k1,x0=myPolyFit(A,B,C,D)

        for obj2 in Lo.objList[::-1]:    # поиск объекта с начальными координатами на продолжении этой линии
#        for obj2 in Lo.objList[obj.id1::]:  # ???
          if obj.id1!=obj2.id1:
            if obj2.t0>=B: #obj.te:
              lenX2=abs(obj2.xce-obj2.xc0)
              if lenX2<9:
                A1,B1,C1,D1=obj2.t0,obj2.te,obj2.xc0,obj2.xce
              elif lenX2<13:
                A1=obj2.coordList[int(obj2.listLen/5)][0]    # t0
                B1=obj2.coordList[int(obj2.listLen*4/5)][0]  # te
                C1=obj2.coordList[int(obj2.listLen/5)][1]    # xc0
                D1=obj2.coordList[int(obj2.listLen*4/5)][1]  # xce
              else:
                A1=obj2.coordList[int(obj2.listLen/6)][0]    # t0
                B1=obj2.coordList[int(obj2.listLen*5/6)][0]  # te
                C1=obj2.coordList[int(obj2.listLen/6)][1]    # xc0
                D1=obj2.coordList[int(obj2.listLen*5/6)][1]  # xce
              x2=x0+k1*A1
              if abs(x2-C1)<=eps:
                obj.coordList.extend(obj2.coordList)
                obj.te=obj2.te
                obj.xce=obj2.xce
                obj.xe=obj2.xe
                obj.listLen+=obj2.listLen
                Lo.removeElement(obj2)
                break
              else:
                for point in obj2.coordList:
                  x3=x0+k1*point[0]
                  if abs(x3-C1)<=eps:
                    obj.coordList.extend(obj2.coordList)
                    obj.te=obj2.te
                    obj.xce=obj2.xce
                    obj.xe=obj2.xe
                    obj.listLen+=obj2.listLen
                    Lo.removeElement(obj2)
                    break
            else:
              break
# end of extrp3() function


def extrp3revers(Lo,data,eps):
#    '''
    for obj in Lo.objList:         # Просмотр всех объектов в обратном направлении
      lenX=abs(obj.xce-obj.xc0)
      if obj.listLen>3:
        if lenX<9:
          A,B,C,D=obj.t0,obj.te,obj.xc0,obj.xce
        elif lenX<13:
          A=obj.coordList[int(obj.listLen/5)][0]    # t0
          B=obj.coordList[int(obj.listLen*4/5)][0]  # te
          C=obj.coordList[int(obj.listLen/5)][1]    # xc0
          D=obj.coordList[int(obj.listLen*4/5)][1]  # xce
        else:
          A=obj.coordList[int(obj.listLen/6)][0]    # t0
          B=obj.coordList[int(obj.listLen*5/6)][0]  # te
          C=obj.coordList[int(obj.listLen/6)][1]    # xc0
          D=obj.coordList[int(obj.listLen*5/6)][1]  # xce

        k1,x0=myPolyFit(A,B,C,D)

#        for obj2 in Lo.objList[::-1]:    # поиск объекта с начальными координатами на продолжении этой линии
        for obj2 in Lo.objList[:obj.id1:]:
          if obj.id1!=obj2.id1:
            if obj2.te<=A: #obj.t0:
#            if obj2.xce<=obj.xc0:
              lenX2=abs(obj2.xce-obj2.xc0)
              if lenX2<9:
                A1,B1,C1,D1=obj2.t0,obj2.te,obj2.xc0,obj2.xce
              elif lenX2<13:
                A1=obj2.coordList[int(obj2.listLen/5)][0]    # t0
                B1=obj2.coordList[int(obj2.listLen*4/5)][0]  # te
                C1=obj2.coordList[int(obj2.listLen/5)][1]    # xc0
                D1=obj2.coordList[int(obj2.listLen*4/5)][1]  # xce
              else:
                A1=obj2.coordList[int(obj2.listLen/6)][0]    # t0
                B1=obj2.coordList[int(obj2.listLen*5/6)][0]  # te
                C1=obj2.coordList[int(obj2.listLen/6)][1]    # xc0
                D1=obj2.coordList[int(obj2.listLen*5/6)][1]  # xce
              x2=x0+k1*B1
              if abs(x2-D1)<=eps:
                obj.coordList.extend(obj2.coordList)
                obj.te=obj2.te
                obj.xce=obj2.xce
                obj.xe=obj2.xe
                obj.listLen+=obj2.listLen
                Lo.removeElement(obj2)
                break
              else:
                for point in obj2.coordList[::-1]:
                  x3=x0+k1*point[0]
                  if abs(x3-D1)<=eps:
                    obj.coordList.extend(obj2.coordList)
                    obj.te=obj2.te
                    obj.xce=obj2.xce
                    obj.xe=obj2.xe
                    obj.listLen+=obj2.listLen
                    Lo.removeElement(obj2)
                    break
            else:
              break #'''
# end of extrp3reverse() function


def removeShorts0(Lo,n):  # Функция удаления объектов с очень короткими треками (список объектов, длина трека)
    for i in Lo.objList:
        if i.listLen<n:
            Lo.removeElement(i)


def removeShorts(Lo,n):  # Функция удаления объектов с очень короткими треками (список объектов, длина трека)
#    for i in Lo.objList:
#        if i.listLen<n:
#            Lo.removeElement(i)

    i=0         # +++++++
    b=Lo.Nobjects
    while i<b:
        if Lo.objList[i].listLen<n:
            if Lo.objList.count(Lo.objList[i])>0:
                Lo.removeElement(Lo.objList[i])
                b-=1
        i+=1


def meanGradient(Lo):  # Функция расчёта средних градиентов для каждого трека (аргумент - список объектов)
    dxs=np.zeros((Lo.Nobjects+1))   # Массивы координат - расстояние
    dts=np.zeros((Lo.Nobjects+1))   # время
    grad=np.zeros((Lo.Nobjects+1))  # Массив градиентов
    c=0                   # Индекс объекта
    for i in Lo.objList:  # Просмотр списка объектов
        c+=1
#        e=len(i.coordList)-1
#        dxs[c]=i.coordList[e][1]-i.x0
#        dts[c]=i.coordList[e][0]-i.t0
        dxs[c]=i.xe-i.x0
        dts[c]=i.te-i.t0
        if dts[c]!=0:
            grad[c]=dxs[c]/dts[c]
        else:
            grad[c]=0
    return grad


def meanGradient1(Lo,x,y,n):  # Функция расчёта среднего градиента на заданном интервале (список, выбранный объект, начало интервала, длина)
    meanGrad=0                # Средний градиент
    i = Lo.objList[x]         # Объект, выбранный по номеру
    e=len(i.coordList)        # Длина списка координат для этого объекта
    for j in range(y,y+n):    # Просмотр этого списка на заданном интервале
        if j+1<e:             # Проверка невыхода за конец списка
            dxs=i.coordList[j+1][1]-i.coordList[j][1]
            dts=i.coordList[j+1][0]-i.coordList[j][0]
            if dts!=0:
                grad=dxs/dts
            else:
                grad=0
            meanGrad+=grad
    return meanGrad


def processGradient(Lo,grads,n,eps):  # Функция обработки градиентов
    pivotPoints=[]
    pivotCount=0
    for i in range(Lo.Nobjects):
        a=Lo.objList[i]
        e=len(a.coordList)
        for j in range(len(a.coordList)-1):
            if j+n<e:
                grad=meanGradient1(Lo,i,j,n)
                if abs(grad-grads[i])>eps:
                    pivotPoints.append([a.id1,a.coordList[j][0],a.coordList[j][1]])
                    pivotCount+=1
    return pivotPoints,pivotCount


def calPoly(Lo):  # Функция вычисления полиномов (аргумент - список объектов)
    idxs = []
    min_points = 8 #6 #8
    polynomials = pd.DataFrame()
    for obj in Lo.objList:
        t = []
        y = []
        if obj.listLen >= min_points:
            idxs.append(obj.id1)    
            for point in obj.coordList:
                t.append(point[0]*stepT)
                y.append(point[1]*step)

            t = np.array(t).reshape(1, -1)
            y = np.array(y).reshape(1, -1)

            fit = np.polyfit(t[0], y[0], 3)
            p = np.poly1d(fit)
            polynomials = pd.concat([polynomials,pd.DataFrame([[t.min(),t.max(),fit[-1],fit[-2],fit[-3],fit[-4]]])], ignore_index=True)

    polynomials.index = idxs  # Присвоение уникальных идентификаторов объектов
    polynomials.columns = ['t0', 't1', 'x0', 'k1', 'k2', 'k3']
    return polynomials
#end of calPoly() functions


def myPolyFit(x1,x2,y1,y2):
    k=(y2-y1)/(x2-x1)  # y(x)=x0+k*x
    x0=y1-k*x1  # x0=y(x)-k*x
    return k,x0

def Poly2(Lo):
    idxs = []
    min_points = 8 #6 #8
    polynomials = pd.DataFrame()
    for obj in Lo.objList:
        t = []
        y = []
        if obj.listLen >= min_points:
            idxs.append(obj.id1)
            for point in obj.coordList:
                t.append(point[0]*stepT)
                y.append(point[1]*step)
            t = np.array(t)
            y = np.array(y)
            k1,x0=myPolyFit(t.min(),t.max(),y[0],y[obj.listLen-1])
            polynomials = pd.concat([polynomials,pd.DataFrame([[t.min(),t.max(),x0,k1,0,0]])], ignore_index=True)
    polynomials.index = idxs  # Присвоение уникальных идентификаторов объектов
    polynomials.columns = ['t0', 't1', 'x0', 'k1', 'k2', 'k3']
    return polynomials


def calPoly_simple(Lo):  # Функция вычисления полиномов (аргумент - список объектов)
    idxs = []
    min_points = 25 #6 #8
    polynomials = pd.DataFrame()
    for obj in Lo.objList:
        t = []
        y = []
        if len(obj.coordList) >= min_points:
            idxs.append(obj.id1)    
            for point in obj.coordList:
                t.append(point[0])
                y.append(point[1])

            t = np.array(t).reshape(1, -1)
            y = np.array(y).reshape(1, -1)

            fit = np.polyfit(t[0], y[0], 1)
            p = np.poly1d(fit)
            polynomials = pd.concat([polynomials,pd.DataFrame([[t.min(),t.max(),fit[-1],fit[-2]]])], ignore_index=True)

    polynomials.index = idxs  # Присвоение уникальных идентификаторов объектов
    polynomials.columns = ['t0', 't1', 'x0', 'k1'] #, 'k2', 'k3']
    return polynomials
#end of calPoly_simple() functions


def polyComp2(p1,p2,eps1=1e-9,eps2=1e-9):  # compare polynomes
    res=0
    if abs(p1[2]-p2[2])>eps2:
        res+=1
    if abs(p1[3]-p2[3])>eps1:
        res+=1
    if abs(p1[4]-p2[4])>eps1:
        res+=1
    if abs(p1[5]-p2[5])>eps1:
        res+=1
    return res

def scanPoly(poly):
    count=0
    fp=None
    for i in range(len(poly)):
        for j in range(len(poly)):
            if i!=j:
#                if polyComp(fit2[i],fit2[j],1e-9)==0:
                if polyComp2(poly[i],poly[j],1e-9,30)==0:
                    fp=poly[i]
                    count+=1
    return count


def meanPolyPos(poly):
    meanPoly0=0
    meanPoly1=0
    meanPoly2=0
    meanPoly3=0
    count=0
    for i in poly:
        if i[3]>=0:
            meanPoly0+=i[2]
            meanPoly1+=i[3]
            meanPoly2+=i[4]
            meanPoly3+=i[5]
            count+=1
    return meanPoly0/count,meanPoly1/count,meanPoly2/count,meanPoly3/count,count


def meanPolyNeg(poly):
    meanPoly0=0
    meanPoly1=0
    meanPoly2=0
    meanPoly3=0
    count=0
    for i in poly:
        if i[3]<0:
            meanPoly0+=i[2]
            meanPoly1+=i[3]
            meanPoly2+=i[4]
            meanPoly3+=i[5]
            count+=1
    return meanPoly0/count,meanPoly1/count,meanPoly2/count,meanPoly3/count,count

def houghLines(Lo,mas):
  '''
  for objA in Lo:
    if objA.listLen>3:
      for objB. in Lo:
        if objB!=objA:
          edges=mas
  '''
  edges=np.array(mas,np.uint8)*255
  lines=cv2.HoughLines(edges,1,np.pi/180,20)
  return lines


def hough2(lines,img):
  t,x,_=img.shape
  img=np.zeros((t,x,3),np.uint8)
  if t>x:
    param1=t
  else:
    param1=x
  param1=22 #350
  for line in lines:
    rho,theta=line[0]
    a = np.cos(theta)    # Stores the value of cos(theta) in a
    b = np.sin(theta)    # Stores the value of sin(theta) in b
    x0 = a*rho           # x0 stores the value rcos(theta)
    y0 = b*rho           # y0 stores the value rsin(theta)
    x1 = int(x0 + param1*(-b))    # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
    y1 = int(y0 + param1*(a))     # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
    x2 = int(x0 - param1*(-b))    # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
    y2 = int(y0 - param1*(a))     # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
#    print(x1,y1,x2,y2)
    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),1)    # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
  return img


def removeShortLens(Lo,n):  # Функция удаления объектов с очень короткими треками (список объектов, длина трека)
    i=0         # +++++++
    while i<Lo.Nobjects:
        if Lo.objList[i].listLen<=n:
            if Lo.objList.count(Lo.objList[i])>0:
                Lo.removeElement(Lo.objList[i])
                i-=1
        i+=1


def removeShortLensX(Lo,n):  # Функция удаления объектов с очень короткими треками (список объектов, длина трека)
    i=0         # +++++++
    while i<Lo.Nobjects:
        if abs(Lo.objList[i].xce-Lo.objList[i].xc0)<=n:
            if Lo.objList.count(Lo.objList[i])>0:
                Lo.removeElement(Lo.objList[i])
                i-=1
        i+=1


def removeZeroDX(Lo):  # Функция удаления объектов с очень короткими треками (список объектов, длина трека)
    i=0         # +++++++
    while i<Lo.Nobjects:
        if Lo.objList[i].mdx==0:
            if Lo.objList.count(Lo.objList[i])>0:
                Lo.removeElement(Lo.objList[i])
                i-=1
        i+=1


def dropTail(Lo,size,n):  # Функция удаления "хвостов" треков с пересчётом средних градиентов
    for obj in Lo.objList:
      if obj.listLen>size:
        count=0
        high=obj.listLen-n-1
        i=obj.listLen-1         # drop tail
        while i>high:
            a=obj.coordList[i]
            if obj.coordList.count(a)>0:
                obj.mdx=(obj.mdx*obj.listLen-obj.coordList[i][2])/(obj.listLen-1)
                obj.coordList.remove(a)
                obj.listLen-=1
#                i+=1
            i-=1
            count+=1

        i=0
        while i<count:  # restore tail
            newXC=obj.coordList[obj.listLen-1][1]+obj.mdx
            newT=obj.coordList[obj.listLen-1][0]+1
            newDX=(obj.mdx*obj.listLen+obj.mdx)/(obj.listLen+1)
            obj.coordList.append([newT,newXC,newDX])
            obj.listLen+=1
            i+=1


def dropTail2(Lo,size,n):  # Функция удаления "хвостов" треков с пересчётом средних градиентов
    for obj in Lo.objList:
      if obj.listLen>size:
        high=obj.listLen-n-1
        i=obj.listLen-1         # drop tail
        while i>high:
            a=obj.coordList[i]
            if obj.coordList.count(a)>0:
                obj.mdx=(obj.mdx*obj.listLen-obj.coordList[i][2])/(obj.listLen-1)
                obj.coordList.remove(a)
                obj.listLen-=1
#                i+=1
            i-=1


def dropHead(Lo,size,n):  # Функция удаления начал треков с пересчётом средних градиентов
    for obj in Lo.objList:
      if obj.listLen>size:# and sign(obj.coordList[obj.listLen-1][2])!=sign(obj.mdx):
        count=0
        if obj.listLen<26:
          n=int(obj.listLen/5)
        for i in range(n):
          a=obj.coordList[0]
          obj.coordList.remove(a)
          obj.mdx=(obj.mdx*obj.listLen-a[2])/(obj.listLen-1)
          obj.listLen-=1

        for i in range(n):
          a=obj.coordList[0]
          obj.mdx=(obj.mdx*obj.listLen+obj.mdx)/(obj.listLen+1)
          obj.coordList.insert(0,[a[0]-1,a[1]-obj.mdx,obj.mdx])
          obj.listLen+=1


def dropHead2(Lo,size,n):  # Функция удаления начал треков с пересчётом средних градиентов
    for obj in Lo.objList:
      if obj.listLen>size:# and sign(obj.coordList[obj.listLen-1][2])!=sign(obj.mdx):
        if obj.listLen<26:
          n=int(obj.listLen/5)
        for i in range(n):
          a=obj.coordList[0]
          obj.coordList.remove(a)
          obj.mdx=(obj.mdx*obj.listLen-a[2])/(obj.listLen-1)
          obj.listLen-=1


def dilation(out):
    t,s,_=out.shape
    out1=np.zeros((t,s,1),np.float32)
    for y in range(1,t-1):
        for x in range(1,s-1):
            if out[y][x][0]==1:
                out1[y-1][x-1][0]=1
                out1[y-1][x][0]=1
                out1[y-1][x+1][0]=1
                out1[y][x-1][0]=1
                out1[y][x][0]=1
                out1[y][x+1][0]=1
                out1[y+1][x-1][0]=1
                out1[y+1][x][0]=1
                out1[y+1][x+1][0]=1
    return out1


def dilation1(out):
    t,s,_=out.shape
    out1=np.zeros((t,s,1))
    for y in range(1,t-1):
        for x in range(1,s-1):
            if (out[y-1][x-1][0]==1 or out[y-1][x][0]==1 or out[y-1][x+1][0]==1 or
                out[y][x-1][0]==1 or out[y][x][0]==1 or out[y][x+1][0]==1 or
                out[y+1][x-1][0]==1 or out[y+1][x][0]==1 or out[y+1][x+1][0]==1):
              out1[y][x][0]=1
    return out1


def erosion(out):
    t,s,_=out.shape
    out1=np.zeros((t,s,1))
    for y in range(1,t-1):
        for x in range(1,s-1):
            if (out[y-1][x-1][0]==1 and out[y-1][x][0]==1 and out[y-1][x+1][0]==1 and
                out[y][x-1][0]==1 and out[y][x][0]==1 and out[y][x+1][0]==1 and
                out[y+1][x-1][0]==1 and out[y+1][x][0]==1 and out[y+1][x+1][0]==1):
              out1[y][x][0]=1
    return out1


def delArt(Lo):
    index1=0         # Коррекция коротких объектов со слишком большими квадратичными коэффициентами
    while index1<Lo.Nobjects: #b:
      print(Lo.objList[index1].id1)
      if abs(polynomials.values[Lo.objList[index1].id1-1][4])>=koefPoly21:
        for obj in Lo.objList[Lo.objList[index1]::-1]:
          pass #obj

#        data=Lo.objList[index1]
#        if Lo.objList.count(data)>0:
#          Lo.objList.remove(data)
#          Lo.Nobjects-=1
#          index1-=1
#        for a1 in
      index1+=1


def bitwise_or(a1,a2):
  c3=np.zeros(a1.shape,np.uint8)
  for a in range(len(a1)):
    for b in range(len(a1[0])):
      if a1[a][b]==0 and a2[a][b]==0:
        c3[a][b]=0
      else:
        c3[a][b]=1
  return c3


def skelet3(img):
#    img=img*255
    print('img.shape=',img.shape)
    skel = np.zeros(img.shape,np.uint8)
    print('skel.shape=',skel.shape)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))  # Get a Cross Shaped Kernel
    count=0
    while True:  # Repeat steps 2-4
        open1 = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)  #Step 2: Open the image
        temp = cv2.subtract(img, open1)                         #Step 3: Substract open from the original image
#        temp=temp.reshape(img.shape)
        print('count=',count)
        print('temp.shape=',temp.shape)
        eroded = cv2.erode(img, element)             #Step 4: Erode the original image and refine the skeleton
        print('skel.shape=',skel.shape)
#        skel = cv2.bitwise_or(skel,temp)
        skel = bitwise_or(skel,temp)
        img = eroded.copy()  # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
        if cv2.countNonZero(img)==0:
            break
        count+=1
#    skel=skel/255
    return skel


def showPoly(polynomials,t,x):  # Визуализация функций полиномов (набор полиномов, макс. время, макс. расстояние)
    fig,ax = plt.subplots()
    ax.set_ylim(0,x)
    x1 = np.arange(0,t,1)
    for i in range(len(polynomials)):
        x1=np.arange(polynomials.values[i][0],polynomials.values[i][1],1)
        y0=polynomials.values[i][2]+polynomials.values[i][3]*x1+polynomials.values[i][4]*x1**2+polynomials.values[i][5]*x1**3
        ax.plot(x1,y0,linewidth=1)
    plt.show()


def main():
    fn=None
    keys=[]
    data=None
    fn,data,keys,t,x,f=initData(fn,data,keys)  # Получение информации о размерах данных в HDF5-файле
    if x%step!=0:
        x=int(x/step)*step

#    t=1500  # ограничение количества временнЫх интервалов (отладочное)
#    x=2400  # ограничение расстояния (отладочное)
#    x=30000  # ограничение расстояния (отладочное)

    out=np.zeros((t,int(x/step),1),np.uint8)#np.float32)  # выделение памяти для массива результатов
    print(datetime.now().time())     # фиксация времени начала обработки нейросетью (отладочная)
    procTime(fn,keys,out,x,t)        # обработка данных (1-й этап)
#    out=dilation(out)
#    out=dilation(out)
    print(datetime.now().time())     # фиксация времени окончания 1-го этапа обработки (отладочная)

    impulseFiltH(out)
#    impulseFiltV(out)

    out2=reStep(out,step)
    cv2.imwrite('nn_out.png',out2*255)  # new in version 1.03.501  # +++
    cv2.waitKey(0)

#    out3=skelet3(out2)*255  # new in version 1.03.503
#    cv2.imwrite('skel_1.png',out3)

    Lo=listObjects()    # список объектов

    dammyArray=np.zeros((int(t/stepT),int(x/step),1))

    c=procData(out,Lo)  # обработка данных (2-й этап)

    Lo.saveList(fnout)

    print('c=',c)

#    removeShortLens(Lo,5)  # +++
    removeShortLensX(Lo,5)
    count=0
    for i in Lo.objList:  # Перенумерация объектов после удаления коротких
        count+=1
        i.id1=count
    print('c2=',count)

    removeZeroDX(Lo)
    count=0
    for i in Lo.objList:  # Перенумерация объектов после удаления коротких
        count+=1
        i.id1=count
    print('c21=',count)

#    dropTail(Lo,5,4)
##    dropTail2(Lo,5,4)
    count=0
    for i in Lo.objList:  # Перенумерация объектов после удаления коротких
        count+=1
        i.id1=count
#    print('c22=',count)
    '''
    mas=Lo.showGraphOCV(out,0,255,0,1)
    cv2.imshow('Color objects1',mas)
    cv2.waitKey(0)
    cv2.imwrite('no_tails.png',mas) #'''

#    dropHead(Lo,5,3)
#    dropHead2(Lo,5,3)
    count=0
    for i in Lo.objList:  # Перенумерация объектов после удаления коротких
        count+=1
        i.id1=count
#    print('c23=',count)

#    for i in range(3):
#      removeShorts0(Lo,9)  # Удаление объектов с короткими треками
##        removeShorts(Lo,3)  # Удаление объектов с короткими треками

    removeZeroDX(Lo)
    count=0
    for i in Lo.objList:  # Перенумерация объектов после удаления коротких
        count+=1
        i.id1=count
    print('c3=',count)

    Lo.showGraph2(out)

#    extrp(Lo,out,6.1) #5.1) #3.51) #3.1) #2.51) #1.51)
    extrp3(Lo,out,6.1)  # +++

    count=0
    for i in Lo.objList:  # Перенумерация объектов после удаления коротких
        count+=1
        i.id1=count

#    extrp3revers(Lo,out,6.1)

#    Lo.show2()
    count=0
    for i in Lo.objList:  # Перенумерация объектов после удаления коротких
        count+=1
        i.id1=count
    print('c4=',count)


    removeShortLens(Lo,7)
    count=0
    for i in Lo.objList:  # Перенумерация объектов после удаления коротких
        count+=1
        i.id1=count
    print('c4=',count)

    print(datetime.now().time())
#    Lo.showGraph2(out)

    Tr2=[]
    for i in Lo.objList:       # анализ длин траекторий
#        print(i.id1,i.listLen)
        Tr2.append(i.listLen)
    Tr2.sort()
#    for i in Tr2:       # анализ длин траекторий
#        print(i)

    fig, ax = plt.subplots()
    xx1 = np.arange(0,len(Tr2),1)
    ax.vlines(xx1,0,Tr2)  # +++
    plt.show()

##    Lo.show2()  # Вывод списка объектов для отладки
#    Lo.showGraph2(out)  # Визуализация для отладки

#    meanGrads=meanGradient(Lo)  # Расчёт средних градиентов
#    print(meanGrads)  # +++
    '''
    c111=0
    for i in meanGrads:
        if i!=0:
            print(i,end=' ')
            c111+=1
    print()
    print('Objects with non-zero gradients=',c111)
    '''

#    pivots,pivotCount=processGradient(Lo,meanGrads,3,0.5)  # Расчёт текущих градиентов

    polynomials=calPoly(Lo)  # Расчёт полиномов
    print(polynomials)
#    print(datetime.now().time())  # фиксация времени окончания 2-го обработки (отладочная)
    showPoly(polynomials,t,x)

    '''
    mas=Lo.showGraphOCV(dammyArray,0,255,0,1)
    cv2.imshow('Color objects2',mas)
    cv2.imwrite('p2.png',mas)
    cv2.waitKey(0)  # '''


    '''
    listToDel=[]
    index1=0         # Удаление коротких объектов со слишком большими квадратичными коэффициентами
    while index1<Lo.Nobjects: #b:
      print('index=',index1,'data.id1=',Lo.objList[index1].id1,'listLen=',Lo.objList[index1].listLen,'poly=',polynomials.values[Lo.objList[index1].id1-1][4])
      if abs(polynomials.values[Lo.objList[index1].id1-1][4])>=koefPoly21:
#      if abs(polynomials.values[index1][4])>=koefPoly21:  # Работает правильно, непонятен результат
        data=Lo.objList[index1]
        print('data.id1=',data.id1,'listLen=',data.listLen,'poly=',polynomials.values[Lo.objList[index1].id1-1][4])
        if Lo.objList.count(data)>0:
          Lo.objList.remove(data)
          Lo.Nobjects-=1
          index1-=1
      index1+=1 #'''

    '''
    count=0
    for i in Lo.objList:  # Перенумерация объектов после удаления коротких
        count+=1
        i.id1=count
    print('c5=',count) #'''

    '''
    for obj in Lo:
      if abs(polynomials.values[obj.id1][4])>koefPoly21:
        data=obj
        if Lo.objList.count(data)>0:
          Lo.objList.remove(data)
          Lo.Nobjects-=1
          index1-=1
      index1+=1 '''

#    polynomials=calPoly(Lo)  # Пересчёт полиномов
#    print(datetime.now().time())  # фиксация времени окончания 2-го обработки (отладочная)
#    print(polynomials)  # Debug

    poly2=Poly2(Lo)
    print(poly2)

##    print(polynomials.values)  # Debug
##    print(len(polynomials.values))  # Debug
#    for i in polynomials.values:
#        print(i[5])  #(i)
#    print()

#    print(scanPoly(polynomials.values))  # Поиск и подсчёт совпадающих полиномов
    print(datetime.now().time())  # фиксация времени окончания обработки (отладочная)

    '''
    k0p,k1p,k2p,k3p,c1=meanPolyPos(polynomials.values)
    print('Far=',c1)
    print(k1p,'*x+',k2p,'*x^2+',k3p,'*x^3')

    k0n,k1n,k2n,k3n,c2=meanPolyNeg(polynomials.values)
    print('Near=',c2)
    print(k1n,'*x+',k2n,'*x^2+',k3n,'*x^3')
    
    Lo.showGraph2(out)  # Визуализация для отладки

    a=Object()
    a.x=0
    a.t=0
    a.xc0=0
    a.x0=0
    a.t0=0
    a.xce=x
    for i in range(t):
        ax=(k1p*i+k2p*i*i+k3p*i*i*i)/step
        print(i,ax)
        if ax<300:
            a.coordList.append([i,ax])
        if ax>x/step:
            break
    Lo.add(a)
    print(datetime.now().time())  # фиксация времени окончания обработки (отладочная)
    Lo.showGraph2(out)  # Визуализация для отладки
    print(Lo.Nobjects)

    for i in Lo.objList:
        print(i.id1,i.listLen)
    #'''

    '''
    fig,ax = plt.subplots()
    ax.set_ylim(0,x)#40*step)
    x1 = np.arange(0,t,1)
    for i in range(len(polynomials)):
#      if abs(polynomials.values[i][4])<koefPoly2:# and abs(polynomials.values[i][5])<koefPoly3:
        x1=np.arange(polynomials.values[i][0],polynomials.values[i][1],1)
        y0=polynomials.values[i][2]+polynomials.values[i][3]*x1+polynomials.values[i][4]*x1**2+polynomials.values[i][5]*x1**3
        ax.plot(x1,y0,linewidth=1)
    plt.show() #'''

#    showPoly(polynomials,t,x)

    showPoly(poly2,t,x)

    '''
    Lo2=listObjects()    # список объектов
    clo2=0
    for i in Lo.objList:
      if i.listLen>20:
        Lo2.add(i)
        clo2+=1
    print('clo2=',clo2)
    Lo2.showGraph2(out) #'''


#    '''
    mas=Lo.showGraphOCV(dammyArray,0,255,0,1)
    cv2.imshow('Color objects2',mas)
    cv2.imwrite('p3.png',mas)
    cv2.waitKey(0)  # '''

#    polynomials.to_hdf('poly.hdf5', key='track_poly', format='table')  # Сохранение полиномов в файле

if __name__ == '__main__':
    main()
