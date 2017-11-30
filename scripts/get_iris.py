import cv2
import matplotlib.pyplot as plt
import matplotlib.image as image
import numpy as np


def find_iris(img):
  #b, g, r = cv2.split(img)
  new_img = img[:,:,0]
  P = 0
  for i in new_img:
    for j in i:
      P += j
   
  # sredni kolor
  P = P/(len(new_img)*len(new_img[1])) 

  P_z = P/4.5
  P_t = P/1.8

  zr = new_img.copy()
  for i in range(len(zr)):
    for j in range(len(zr[1])):
      if zr[i][j] > P_z: zr[i][j] = 0
      else: zr[i][j] = 1
      
  zr_z = zr.copy()
  for i in range(2,len(zr)-2):
    for j in range(2,len(zr[1])-2):
      if zr[i][j] == 1:
        cont = 0
        for k in range(i-2,i+3):
          for l in range(j-2,j+3):
            cont += zr[k][l]
        if cont < 10:
          zr[i][j] = 0
          
  t = new_img.copy()
  for i in range(len(zr)):
    for j in range(len(zr[1])):
      if t[i][j] > P_t: t[i][j] = 0
      else: t[i][j] = 1

  for i in range (0,5):
    for j in range(len(t[1])):
      t[i][j] = 0
  for i in range (len(t)):
    for j in range(0,5):
      t[i][j] = 0    
      
  for i in range(3,len(zr)-3):
    for j in range(3,len(zr[1])-3):
      if t[i][j] == 1:
        cont = 0
        for k in range(i-9,i+10):
          for l in range(j-9,j+10):
            cont += t[k][l]
        if cont < 170:
          t[i][j] = 0


  for pix in range(len(zr[1])):
    zr[len(zr)-1] = 0
  #srodek
  cnt_i = 0
  cnt_j = 0
  cnt = 0
  for i in range(len(zr)):
    for j in range(len(zr[1])):
      if zr[i][j] == 1:
        cnt += 1
        cnt_i +=i
        cnt_j += j
        
  sr_i = cnt_i/cnt
  sr_j = cnt_j/cnt

  cnt_i = 0
  cnt_j = 0
  cnt = 0
  for i in range(len(zr)):
    for j in range(len(zr[1])):
      if t[i][j] == 1:
        cnt += 1
        cnt_i +=i
        cnt_j += j
        
  srt_i = cnt_i/cnt
  srt_j = cnt_j/cnt

  r = 5
  cont = True
  while cont:
    sum_r = 0
    for k in range(sr_i-r, sr_i+r+1):
      for l in range(sr_j-r, sr_j+r+1):
        sum_r += zr[k][l]
    sum_r_next = 0
    for k in range(sr_i-r-1, sr_i+r+2):
      for l in range(sr_j-r-1, sr_j+r+2):
        sum_r_next += zr[k][l]
    r += 1    
    if sum_r_next == sum_r:  
      cont = False
      r -= 1

  rt = r  
  srt_i = sr_i
  srt_j = sr_j  
  cont = True
  while cont:
    sum_r = 0
    for k in range(srt_i-rt, srt_i+rt+1):
      for l in range(srt_j-rt, srt_j+rt+1):
        sum_r += t[k][l]
    sum_r_next = 0
    for k in range(srt_i-rt-1, srt_i+rt+2):
      for l in range(srt_j-rt-1, srt_j+rt+2):
        sum_r_next += t[k][l]
    rt += 1    
    if sum_r_next < sum_r+5:  
      cont = False
      rt -= 1
  return sr_i, sr_j, r, srt_i, srt_j, rt, zr, t
  
  
def polar2cart(r, theta, center):

    x = r  * np.cos(theta) + center[0]
    y = r  * np.sin(theta) + center[1]
    return x, y

def img2polar(img, center, final_radius, initial_radius = None, phase_width = 3000):

    if initial_radius is None:
        initial_radius = 0

    theta , R = np.meshgrid(np.linspace(0, 2*np.pi, phase_width), 
                            np.arange(initial_radius, final_radius))

    Xcart, Ycart = polar2cart(R, theta, center)

    Xcart = Xcart.astype(int)
    Ycart = Ycart.astype(int)

    if img.ndim ==3:
        polar_img = img[Ycart,Xcart,:]
        polar_img = np.reshape(polar_img,(final_radius-initial_radius,phase_width,3))
    else:
        polar_img = img[Ycart,Xcart]
        polar_img = np.reshape(polar_img,(final_radius-initial_radius,phase_width))

    return polar_img  

def get_iris(img):
  [ppl_x, ppl_y, ppl_r, iris_x, iris_y, iris_r, ppl, iris_wppl ] = find_iris(img)

  # where is only iris
  iris_01 = iris_wppl - ppl
  iris_circle = img.copy()
  cv2.circle(iris_circle,(iris_y,iris_x), iris_r, (0,0,255), -1)
  ppl_circle = img.copy()
  cv2.circle(ppl_circle,(ppl_y,ppl_x), ppl_r, (0,0,255), -1)
  iris_cc = ppl_circle - iris_circle

  iris = iris_cc.copy()
  iris[:,:,0] = iris[:,:,0] * iris_01
  iris[:,:,1] = iris[:,:,1] * iris_01
  iris[:,:,2] = iris[:,:,2] * iris_01

  #croppped
  iris_cr = iris[iris_x-iris_r:iris_x+iris_r, iris_y-iris_r:iris_y+iris_r]

  iris_unwrapped = 0

  #cv2.linearPolar(iris_cr, iris_unwrapped, [float(iris_r), float(iris_r)],float(iris_r))
  iris_pol = img2polar(iris_cr, [iris_r,iris_r], iris_r, phase_width = 300)
  iris_pol2 = iris_pol[ppl_r:iris_r]
  return iris_pol2

#circle1 = plt.Circle((ppl_y, ppl_x), ppl_r, color='r', fill=False)
#circle2 = plt.Circle((iris_y, iris_x), iris_r, color='r', fill=False)

if __name__ == '__main__': 
  img = image.imread('../data/oko01.png')
  iris = get_iris(img)
  fig, ax = plt.subplots()
  #ax.add_artist(circle1)
  #ax.add_artist(circle2)
  ax.imshow(iris)
  plt.show()