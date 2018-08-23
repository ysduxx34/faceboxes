import os
root = "/home/ysdu/personBox/train/"
with open('./train_annotations.txt') as f:
    lines = f.readlines()
    nums = len(lines)
    
output = open('/home/ysdu/personBox/label/box_label_train.txt', 'w')

for i in range(nums):
    line = lines[i]
    
    splited = line.strip().split()
    imgName = root+splited[0]
#     print (imgName)
#     img = cv2.imread(imgName, cv2.IMREAD_COLOR)
    boxesNum = int((len(splited)-1) / 5)
    
    if not os.path.exists(imgName) or boxesNum < 1:
        continue
        
    imgName = imgName + ' ' + str(boxesNum) + ' '
    
    for j in range(1,len(splited),5):
        imgName = imgName + splited[j+1] + ' '
        imgName = imgName + splited[j+2] + ' '
        imgName = imgName + splited[j+3] + ' '
        imgName = imgName + splited[j+4] + ' '
        imgName = imgName + '1' + ' '

    # print(i)    
    output.writelines(imgName+'\n')