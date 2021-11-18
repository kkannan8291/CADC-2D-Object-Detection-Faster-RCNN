import os
from os import listdir, getcwd
from os.path import join

if __name__ == '__main__':
    #source_folder =r'/home/kaushik/ObjectDetection/CADC-2D-Object-Detection-Faster-RCNN/data/VOC-Converted-Data/VOC2007/JPEGImages'
    source_folder =r'/home/kaushik/ObjectDetection/CADC-2D-Object-Detection-Faster-RCNN/data/VOC-Converted-Data/VOC2007/JPEGImages'
    dest = r'/home/kaushik/ObjectDetection/CADC-2D-Object-Detection-Faster-RCNN/data/VOC-Converted-Data/VOC2007/ImageSets/Main/trainval.txt'
    dest2 = r'/home/kaushik/ObjectDetection/CADC-2D-Object-Detection-Faster-RCNN/data/VOC-Converted-Data/VOC2007/ImageSets/Main/test.txt'
    file_list = os.listdir(source_folder)
    train_file = open(dest, 'a')
    test_file = open(dest2, 'a')
    i=0
    for file_obj in file_list:
        file_name, file_extend = os.path.splitext(file_obj)

        if (i%8 ==0):
            test_file.write(file_name + '\n')
        else:
            train_file.write(file_name + '\n')
        i+=1
    train_file.close()
test_file.close()