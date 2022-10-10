import cv2
import numpy as np

img = cv2.imread('math1.jpg')
im_copy = img.copy()
contrast = 1.5  # Contrast control (1.0-3.0)
brightness = 0  # Brightness control (0-100)

# result = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)


# Remove lines
vertical_size = img.shape[0] // 3
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
vertical_mask = 255 - cv2.morphologyEx(img, cv2.MORPH_CLOSE, vertical_kernel)

vertical_extract = cv2.morphologyEx(img, cv2.MORPH_CLOSE, vertical_kernel)

just_vertical = cv2.add(img, vertical_mask)

horizontal_size = img.shape[1] // 9
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
horizontal_mask = 255-cv2.morphologyEx(img, cv2.MORPH_CLOSE, horizontal_kernel)
just_horizontal = cv2.add(img, horizontal_mask)

result = just_vertical #cv2.addWeighted(just_horizontal, 1, just_vertical, 0, 0) #placeholder values

cv2.imshow("original image", img)
cv2.waitKey(0)
cv2.destroyWindow("original image")
cv2.imshow("extracted features", vertical_extract)
cv2.waitKey(0)
cv2.destroyWindow("extracted features")
cv2.imshow("result", just_horizontal)
cv2.waitKey(0)
cv2.destroyWindow("result")

# Converting to greyscale
gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
# applying Gaussian blur
blur = cv2.GaussianBlur(gray, (5, 5), 0)
# experimenting with thresholding functions
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)[1]

cv2.imshow("thresh", thresh)
cv2.waitKey(0)
cv2.destroyWindow('thresh')

edged = cv2.Canny(thresh, 70, 200)

cv2.imshow('edged', edged)
cv2.waitKey(0)
cv2.destroyWindow('edged')

contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
counter = 1
area = []
CentreOfMass = []
coords = []
for ctr in contours:
    x, y, w, h = cv2.boundingRect(ctr)
    area.append(w * h)
area = sorted(area)
areaMean = np.mean(area)    # average area
areaSD = np.std(area)       # standard deviation of area
print("average area =", areaMean)
print("areaSD = ", areaSD)
for ctr in contours:
    x, y, w, h = cv2.boundingRect(ctr)
    area = w * h
    if area >= areaMean / 5:  # and area <= areaMean*10):
        cv2.rectangle(im_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)
        coords.append([x, y, w, h, counter - 1])
        print(counter)
        CentreOfMass.append([(x + w) / 2, (y + h) / 2])
        counter += 1
Coords = sorted(coords, key=lambda x: (x[0]))

print(Coords)

# none of the shit below works lmao

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
cv2.imshow('contours', im_copy)
cv2.waitKey(0)

