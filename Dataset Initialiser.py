import ImageSegmentationModule as sg
import pathlib
import matplotlib.pyplot as plt
import cv2
import ntpath
import random

# data_dir = pathlib.Path('C:/Users/jerry/OneDrive/Documents/GitHub/Camera-Calculator/dataset')
data_dir = pathlib.Path('C:/Users/jerry/Downloads/traindata/dataset')
folders = list(data_dir.glob('*'))[1:]
print(folders)
classnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'add', 'div', 'eq', 'mul', 'sub',
              'x']  # removed 'dec', 'y', 'z'

''' segmenting dataset '''
# for i in range(0, len(classnames)):
#     k = 0
#     flag = False
#     for img in list(data_dir.glob(classnames[i] + '/*.png')) + list(data_dir.glob(classnames[i] + '/*.jpg')):
#         img_path = img
#         img = cv2.imread(str(img))
#         # cv2.imshow(str(i), img)
#         resized_img = sg.segmentDataset(img)
#         if resized_img[0] == 'error' or len(resized_img) > 1:
#             print(len(resized_img))
#             print(img_path)
#             resized_img, areas = sg.segment(img, test=True)
#             flag = True
#             break
#         cv2.imwrite('test_data/' + classnames[i] + '/' + str(k) + '.png', resized_img[0])
#         k += 1
#     if flag:
#         break
        # blacklisted images C:\Users\jerry\Downloads\traindata\dataset\x\87omOgfZ.png

''' Check loss '''
# train_data = pathlib.Path('C:/Users/jerry/OneDrive/Documents/GitHub/Camera-Calculator/dataset')
# for i in range(0, len(classnames)):
#     og = len(list(data_dir.glob(classnames[i]+'/*.*'))) # data_dir.glob(classnames[i]+'/*.png'))+
#     list((data_dir.glob(classnames[i]+'/*.jpg'))))
#     output = len(list(train_data.glob(classnames[i]+'/*.png')))
#     print('class : ',classnames[i],', no. of lost images = ', output)


''' individual troubleshooting '''
resized_imgs = sg.segment(cv2.imread('C:/Users/jerry/Downloads/Picture3.jpg'), test=True) ,# groupAll=False)
print(len(resized_imgs))
cv2.imshow('x', resized_imgs[0])
cv2.waitKey(0)

'''Batch troubleshooting'''
# k=0
# for img in list(data_dir.glob('4/*.png')) + list(data_dir.glob('4/*.jpg')):
#     img_path = img
#     img = cv2.imread(str(img))
#
#     def path_leaf(path):
#         head, tail = ntpath.split(path)
#         return tail or ntpath.basename(head)
#
#
#     img_path = path_leaf(img_path)
#     # cv2.imshow(str(img_path), img)
#     # cv2.waitKey(0)
#     resized_img = sg.segmentDataset(img, groupAll=True)
#     if len(resized_img)>1 :
#         print(img_path)
#         resized_imgs = sg.segmentDataset(cv2.imread('C:/Users/jerry/Downloads/traindata/dataset/4/'+img_path), test=True, groupAll=True)
#         cv2.imshow('x', resized_imgs[0])
#         cv2.waitKey(0)
#     cv2.imwrite('dataset/4/' + str(k) + '.png', resized_img[0])
#
#     k+=1
    # print(len(resized_img))
    # cv2.imshow(str(img_path), resized_img[0])
    # cv2.waitKey(0)
    # cv2.destroyWindow(str(img_path))

'''Concatenating datasets'''
# (x_train,y_train),(x_test,y_test) = keras.datasets.mnist.load_data()
# # print(np.shape(x_train))
# x_train = np.array([cv2.resize(img, (50,50)) for img in x_train ])
# x_test = np.array([cv2.resize(img, (50,50)) for img in x_test ])
# cv2.imshow(str(y_test[2]),x_test[2])
# print(x_test[2].shape)
# cv2.waitKey(0)
# for i in range(0, len(resized_imgs)):
#     cv2.imwrite('test_data/%s.png' % i, resized_imgs[i])
#     print("saved")
# cv2.imshow('saved', cv2.imread('test_data/0.jpg'))
# cv2.waitKey(0)


''' Showing dataset'''
# for i in range(9):
#     classname = classnames[random.randint(0,15)]
#     imgpath = (list(data_dir.glob(classname + '/*.png'))+list(data_dir.glob(classname+'/*.jpg')))[random.randint(0,100)]
#     img = cv2.imread(str(imgpath))
#     ax = plt.subplot(3,3,i+1)
#     plt.imshow(img)
#     plt.title(classname + f', {img.shape}')
#     plt.axis("off")
# plt.show()


