import os
from os import listdir, getcwd
from os.path import join

if __name__ == '__main__':
    source_folder =r'/home/kaushik/ObjectDetection/CADC-2D-Object-Detection-Faster-RCNN/data/VOC-Converted-Data/JPEGImages'
    dest = r'/home/kaushik/ObjectDetection/CADC-2D-Object-Detection-Faster-RCNN/data/VOC-Converted-Data/ImageSets/Main/trainval.txt'
    #dest2 = r'G:\jianfeng\project\rubblish_det\source\train_pic_json\voc_all/VOC2018/ImageSets/Main/val.txt'
    file_list = os.listdir(source_folder)
    train_file = open(dest, 'a')
    #val_file = open(dest2, 'a')
    i=0
    for file_obj in file_list:
        file_name, file_extend = os.path.splitext(file_obj)

        #if (i%4 ==0):
        #    val_file.write(file_name + '\n')
        #else:
        train_file.write(file_name + '\n')
        i+=1
    train_file.close()
#val_file.close()