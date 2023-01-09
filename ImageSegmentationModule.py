import cv2
import numpy as np
import math


# contrast = 1.5  # Contrast control (1.0-3.0)
# brightness = 0  # Brightness control (0-100)


# result = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)


# Remove lines
def removeStructure(img, direction):
    # performs an 'opening' transformation (erosion followed by dilation) to the img to further remove noise)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    vertical_size = img.shape[0] // 3
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    vertical_mask = 255 - cv2.morphologyEx(img, cv2.MORPH_CLOSE, vertical_kernel)

    vertical_extract = cv2.morphologyEx(img, cv2.MORPH_CLOSE, vertical_kernel)

    just_vertical = cv2.add(img, vertical_mask)

    horizontal_size = img.shape[1] // 9
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal_mask = 255 - cv2.morphologyEx(img, cv2.MORPH_CLOSE, horizontal_kernel)
    just_horizontal = cv2.add(img, horizontal_mask)

    horizontal_extract = cv2.morphologyEx(img, cv2.MORPH_CLOSE, horizontal_kernel)

    if direction == 1:
        result = just_vertical
    else:
        result = just_horizontal

    # cv2.imshow("extracted features", features)
    # cv2.waitKey(0)
    # cv2.destroyWindow("extracted features")

    # result = cv2.addWeighted(just_horizontal, 1, just_vertical, 0, 0) placeholder values
    return result


def getDirection(img, img_copy):
    # Converting to greyscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # applying Gaussian blur
    # blur = cv2.GaussianBlur(gray, (5, 5), 0)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    # experimenting with thresholding functions
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)[1]

    print(img.shape[:3])
    lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, 50, minLineLength=min(img.shape[:2]) / 1.5, maxLineGap=5)
    gradients = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            if x2 - x1 != 0:
                gradients.append((y2 - y1) / (x2 - x1))
            else:
                direction = 1
                return direction
            cv2.line(img_copy, (x1, y1), (x2, y2), (0, 0, 255), 3)
    mean_gradient = np.mean(gradients)
    if mean_gradient < -2 or mean_gradient > 2:
        direction = 1
    else:
        direction = 0
    cv2.imshow('detected lines', img_copy)
    cv2.waitKey(0)
    return direction


def findContourBoxes(edged, im_copy):
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    counter = 1
    area = []
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
        if area >= areaMean / 20:  # and area <= areaMean*10):
            cv2.rectangle(im_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)
            coords.append([x, y, w, h])
            print(counter)
            counter += 1
    Coords = sorted(coords, key=lambda x: (x[0]))
    print(Coords)

    return Coords


def getOverlap(box1, box2):
    return min([box2[0] + box2[2] - box1[0], box1[0] + box1[2] - box2[0]])


def drawGroupContours(Coords, img):
    Coords = sorted(Coords, key=lambda x: (x[0]))
    groupedCoords = [Coords[0]]
    Coords.remove(Coords[0])
    for i in range(0, len(Coords)):
        box = Coords[0]
        print('new length = ', Coords)
        if (groupedCoords[-1][0] <= box[0] <= groupedCoords[-1][0] + groupedCoords[-1][2]) or (
                groupedCoords[-1][0] <= box[0] + box[2] <= groupedCoords[-1][0] + groupedCoords[-1][2]):
            overlap = getOverlap(groupedCoords[-1], box)
            if (overlap >= groupedCoords[-1][2] / 10 or ((groupedCoords[-1][0] <= box[0] <= groupedCoords[-1][0] +
                                                          groupedCoords[-1][2]) and (
                                                                 groupedCoords[-1][0] <= box[0] + box[2] <=
                                                                 groupedCoords[-1][0] + groupedCoords[-1][2]))):
                groupedCoords[-1] = [min(groupedCoords[-1][0], box[0]), min(groupedCoords[-1][1], box[1]),
                                     max(groupedCoords[-1][2], box[0] + box[2] - groupedCoords[-1][0]),
                                     max(groupedCoords[-1][3], box[1] + box[3] - groupedCoords[-1][1],
                                         groupedCoords[-1][1] +
                                         groupedCoords[-1][3] - box[1])]
            else:
                groupedCoords.append(box)
        else:
            groupedCoords.append(box)
            print('box appended = ', box)
        Coords.remove(box)
    for sortedBox in groupedCoords:
        cv2.rectangle(img, (sortedBox[0], sortedBox[1]), (sortedBox[0] + sortedBox[2], sortedBox[1] + sortedBox[3]),
                      (255, 255, 255), 2)
    return groupedCoords


def cropContourBoxes(coords, img):
    img_list = []
    coords = sorted(coords, key=lambda x: (x[0]))
    for box in coords:
        symbol = img[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
        symbol = cv2.morphologyEx(symbol, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        symbol = cv2.morphologyEx(symbol, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        img_list.append(symbol)
        # visualize cropped images
        # cv2.imshow("Cropped_image", symbol)
        # cv2.waitKey(0)
        # cv2.destroyWindow("Cropped_image")
    return img_list


def resize(img_list, size):
    resized_imgs = []
    for img in img_list:
        # Get the aspect ratio of the input image
        height, width = img.shape[:2]
        # area_ratio = int(height*width/size**2)
        # matrix_size = int(area_ratio/3)
        # print('size of matrix = ', matrix_size)
        if height > width:
            padded_img = cv2.copyMakeBorder(
                img,
                top=0,
                bottom=0,
                left=math.ceil((height - width) / 2),
                right=math.floor((height - width) / 2),
                borderType=cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
        elif height < width:
            padded_img = cv2.copyMakeBorder(
                img,
                top=math.ceil((width - height) / 2),
                bottom=math.floor((width - height) / 2),
                left=0,
                right=0,
                borderType=cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
        else:
            padded_img = img

        # Resize the image while preserving its aspect ratio
        resized_img = cv2.resize(padded_img, (int(size * 0.8), int(size * 0.8)))
        resized_img = cv2.copyMakeBorder(
            resized_img,
            top=math.ceil((size - int(size * 0.8)) / 2),
            bottom=math.floor((size - int(size * 0.8)) / 2),
            left=math.ceil((size - int(size * 0.8)) / 2),
            right=math.floor((size - int(size * 0.8)) / 2),
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        resized_img = cv2.morphologyEx(resized_img, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        resized_imgs.append(resized_img)
    return resized_imgs


def segment(file):
    img = cv2.imread(file)
    im_copy = img.copy()
    result = img.copy()

    # cv2.imshow("original image", img)
    # cv2.waitKey(0)
    # cv2.destroyWindow("original image")

    result = removeStructure(img, 1)
    direction = getDirection(result, im_copy)
    print(direction)
    result = removeStructure(result, direction)

    cv2.imshow("result", result)
    cv2.waitKey(0)
    cv2.destroyWindow("result")

    # Converting to greyscale
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    # applying Gaussian blur
    # blur = cv2.GaussianBlur(gray, (5, 5), 0)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    # experimenting with thresholding functions
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)[1]

    cv2.imshow("thresh", thresh)
    cv2.waitKey(0)
    cv2.destroyWindow('thresh')

    edged = cv2.Canny(thresh, 70, 200)

    # cv2.imshow('edged', edged)
    # cv2.waitKey(0)
    # cv2.destroyWindow('edged')

    coords = findContourBoxes(edged, im_copy)
    cv2.imshow('contours', im_copy)
    cv2.waitKey(0)
    coords = drawGroupContours(coords, img)
    print(coords)
    cv2.imshow('grouped contours', img)
    cv2.waitKey(0)
    cv2.destroyWindow('grouped contours')
    imglist = cropContourBoxes(coords, thresh)
    resized_imgs = resize(imglist, 28)
    print(len(coords))
    print(coords)
    for img in imglist:
        print(img.shape)
    for img in resized_imgs:
        cv2.imshow('input', img)
        cv2.waitKey(0)

    return resized_imgs


#segment('Images/math4.jpg')
# [4 7 7 1 6 6 2 2 3]s
