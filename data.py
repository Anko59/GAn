import pickle
import numpy as np
import tensorflow as tf
def import_cifar():
  print('Stating import...')
  path = 'cifar-10-python/cifar-10-batches-py/data_batch_'
  massiveDic = []
  for x in range(5):
    newPath = path+str(x+1)
    print(newPath)
    with open(newPath,'rb') as file:
       dic = pickle.load(file, encoding='bytes')
       data = dic[list(dic.keys())[2]]
       for y in range(len(data)):
         massiveDic.append(data[y])
  print('Import finished')
  return massiveDic
  
dic = import_cifar()
newDic = []
for img in dic:
  tensor = tf.convert_to_tensor(img, dtype=tf.float32)
  tensor = tf.reshape(tensor, [32,32,3])
  newDic.append(tensor)
  
dic = newDic
print(dic[0])

