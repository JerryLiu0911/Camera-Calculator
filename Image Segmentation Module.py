import cv2
import numpy as np

img = cv2.imread('images/math4.jpg')
im_copy = img.copy()
final_copy = img.copy()
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
horizontal_mask = 255 - cv2.morphologyEx(img, cv2.MORPH_CLOSE, horizontal_kernel)
just_horizontal = cv2.add(img, horizontal_mask)

result = just_vertical  # cv2.addWeighted(just_horizontal, 1, just_vertical, 0, 0) #placeholder values

cv2.imshow("original image", img)
cv2.waitKey(0)
cv2.destroyWindow("original image")
cv2.imshow("extracted features", vertical_extract)
cv2.waitKey(0)
cv2.destroyWindow("extracted features")
cv2.imshow("result", just_vertical)
cv2.waitKey(0)
cv2.destroyWindow("result")

# Converting to greyscale
gray = cv2.cvtColor(just_vertical, cv2.COLOR_BGR2GRAY)
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
areaMean = np.mean(area)  # average area
areaSD = np.std(area)  # standard deviation of area
print("average area =", areaMean)
print("areaSD = ", areaSD)
for ctr in contours:
    x, y, w, h = cv2.boundingRect(ctr)
    area = w * h
    if area >= areaMean / 10:  # and area <= areaMean*10):
        cv2.rectangle(im_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)
        coords.append([x, y, w, h])
        print(counter)
        CentreOfMass.append([(x + w) / 2, (y + h) / 2])
        counter += 1
Coords = sorted(coords, key=lambda x: (x[0]))

print(Coords)


# none of the shit below works lmao


#groupedBoxes = cv2.groupRectangles()


def getOverlap(box1, box2):
    return min([box2[0] + box2[2] - box1[0], box1[0] + box1[2] - box2[0]])


for box in Coords:
    temp = box
    Coords.remove(box)
    print("Current box coords", temp)
    print(sorted(Coords, key=lambda x: (x[0])))
    for i in range(0, 2):
        for otherBox in Coords:
            if (temp[0] <= otherBox[0] <= temp[0] + temp[2]) or (temp[0] <= otherBox[0] + otherBox[2] <= temp[0] + temp[2]):
                overlap = getOverlap(temp, otherBox)
                if overlap >= temp[2] / 3:
                    temp = [min(temp[0], otherBox[0]), min(temp[1], otherBox[1]),
                            max(temp[2], otherBox[0] + otherBox[2] - temp[0], temp[0]+temp[2]-otherBox[0]),
                            max(temp[3], otherBox[1] + otherBox[3] - temp[1], temp[1]+temp[3]-otherBox[1])]
                    Coords.remove(otherBox)
                    print("box removed")
        Coords.append(temp)
    i+=1
for sortedBox in Coords:
    cv2.rectangle(final_copy, (sortedBox[0], sortedBox[1]), (sortedBox[0] + sortedBox[2], sortedBox[1] + sortedBox[3]),
                  (0, 255, 0), 2)


cv2.imshow('contours', im_copy)
cv2.imshow('grouped contours', final_copy)
print(len(Coords))
cv2.waitKey(0)
