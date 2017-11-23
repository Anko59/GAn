import pickle
import numpy as np
import tensorflow as tf


cifar_batchG = 1
cifar_batchD = 1
current_batchG = 0
current_batchD = 0
batch_size = 100
init = 0

def showImage(imageMat, t=False):
  if t: imageMat = tf.Session().run(imageMat)    
  plt.imshow(imageMat)
  plt.show()
  
def import_cifar(batch_nbr):
  print('Stating import...')
  path = 'cifar-10-batches-py/data_batch_' + str(batch_nbr)
  massiveDic = []
  with open(path,'rb') as file:
    dic = pickle.load(file, encoding='bytes')
    data = dic[list(dic.keys())[2]]
    for y in range(len(data)):
      massiveDic.append(data[y])
  print('Import finished')
  resized_images = []
  newDic = np.zeros(shape=(10000,3,32,32))
  print('starting exctraction and resizing...')
  for i in range(len(massiveDic)):
    newDic[i] = massiveDic[i].reshape(3,32,32)
  newDic = tf.convert_to_tensor(newDic)
  newDic = tf.transpose(newDic, [0,2,3,1])
  del massiveDic
  resized_images = tf.image.resize_images(newDic, [16,16])
  resized_images = tf.cast(resized_images, np.uint8)
  newDic = tf.cast(newDic, np.uint8)
  print('done')
  return tf.Session().run([newDic, resized_images])

def get_batch(batch_type):
    global current_batchG, cifar_batchG, cifar_batchD, current_batchD, batch_size, img, r_img,init
    if init== 0:
      img = np.load("data/original1.npy")
      r_img = np.load("data/resized1.npy")
      init = 1
    if current_batchG  + batch_size > r_img.shape[0] and batch_type == "resized":
      if cifar_batchG == 5:
          cifar_batchG = 0
      del r_img
      cifar_batchG += 1
      current_batchG = 0
      r_img = np.load("data/resized" + str(cifar_batchG) + ".npy")
      
    if current_batchD  + batch_size > r_img.shape[0] and batch_type == "original":
        if cifar_batchD == 5:
          cifar_batchD = 0
        del img
        cifar_batchD += 1
        current_batchD = 0
        img = np.load("data/original" + str(cifar_batchD) + ".npy")
        
    if batch_type == "original":
        output = img[current_batchD:current_batchD+batch_size]
        current_batchD += batch_size
    if batch_type =="resized":
        output = r_img[current_batchG:current_batchG+batch_size]
        current_batchG += batch_size
        
    return output

def split_data():
  for x in range(4):
    newDic,resized = import_cifar(x+1)
    split_in = 2
    bt_size = len(newDic/split_in)
    for y in range(split_in):
      with open("data/original"+str(x)+str(y)+".dtt", "wb" ) as output_file:
        pickle.dump(newDic[(y)*bt_size:(y+1)*bt_size],output_file)
      with open("data/resized"+str(x)+str(y)+".dtt", "wb" ) as output_file:
        pickle.dump(resized[(y)*bt_size:(y+1)*bt_size],output_file)
