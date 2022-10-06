import cv2
import numpy as np
ogimg = cv2.imread('math3.jpg')
img = cv2.imread('math2.jpg',cv2.IMREAD_GRAYSCALE)
im_copy = img.copy()

# cv2.imshow("prethresh",thresh)
blur = cv2.GaussianBlur(im_copy, (11, 11), 0)
thresh = cv2.threshold(blur, 0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
#thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 201, 4)

#cv2.imshow("thresh",thresh)
#cv2.imshow('blur',blur)
cv2.waitKey(0)
edged=cv2.Canny(thresh,70,200)
contours, hierarchy=cv2.findContours(edged,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
#cv2.imshow('thresh',edged)
cv2.waitKey(0)
counter = 1
area = []
CentreOfMass = []
Coords = []
for ctr in contours:
    x, y, w, h = cv2.boundingRect(ctr)
    #cv2.imshow()
    area.append(w*h)
area = sorted(area)
print(area)
areaMean = np.mean(area)
areaSD = np.std(area)
print("avarage area =", areaMean)
print("areaSD = ",areaSD)
for ctr in contours:
    x, y, w, h = cv2.boundingRect(ctr)
    area = w*h
    if (area >= areaMean/50 and area <= areaMean*10):
        cv2.rectangle(im_copy, (x, y), (x + w, y + h), (0,0,255) , 2)
        Coords.append([x, y, w, h, counter-1])
        print(counter-1)
        CentreOfMass.append([(x+w)/2,(y+h)/2])
        counter += 1
        #print(CentreOfMass[counter - 1])
    cv2.rectangle(im_copy, (x, y), (x + w, y + h), (0,0,255) , 2)
Coords = sorted(Coords,key = lambda x:(x[0]))

print(Coords)

#none of the shit below works lmao

# for index in range(len(Coords)):
#     for possibleMatch in range(len(Coords)-index-1):
#         xdiff = abs(Coords[index][2]-Coords[possibleMatch+index][0])
#         if Coords[index][0]<=Coords[possibleMatch+index][0]<=Coords[index][0]+Coords[index][2]:
#             if xdiff<= 2*min(Coords[index][2], Coords[possibleMatch+index][2]):
#                 newX = max(Coords[index][0], Coords[possibleMatch+index][0])
#                 newY = max(Coords[index][1], Coords[possibleMatch+index][1])
#                 newW = max(Coords[index][2], Coords[possibleMatch+index][2])
#                 newH = Coords[index][3]+Coords[possibleMatch+index][3]
#                 cv2.rectangle(im_copy, (newX, newY), (newX + newW, newY + newH), (0,0,255) , 2)
#                 print(temp)
#                 Coords.remove(Coords[Coords[index][4]])
#                 Coords.remove(temp)
#                 print(Coords[index][4], Coords[possibleMatch][4])

print(CentreOfMass)
cv2.imshow('contours',im_copy)
cv2.waitKey(0)
