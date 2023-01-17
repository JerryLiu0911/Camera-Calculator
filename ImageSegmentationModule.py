import cv2
import numpy as np
import math
# from tensorflow import keras
import pathlib

# tf.compat.v1.enable_eager_execution()

# contrast = 1.5  # Contrast control (1.0-3.0)
# brightness = 0  # Brightness control (0-100)


# result = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)


# Remove lines
def removeStructure(img, direction, v_size = 3, h_size = 9):
    # performs an 'opening' transformation (erosion followed by dilation) to the img to further remove noise)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    vertical_size = img.shape[0] // v_size # should be //3. Change back after dataset mod
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    vertical_mask = 255 - cv2.morphologyEx(img, cv2.MORPH_CLOSE, vertical_kernel)

    vertical_extract = cv2.morphologyEx(img, cv2.MORPH_CLOSE, vertical_kernel)

    just_vertical = cv2.add(img, vertical_mask)

    horizontal_size = img.shape[1] // h_size # should be //9, Change back after dataset mod
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal_mask = 255 - cv2.morphologyEx(img, cv2.MORPH_CLOSE, horizontal_kernel)
    just_horizontal = cv2.add(img, horizontal_mask)

    horizontal_extract = cv2.morphologyEx(img, cv2.MORPH_CLOSE, horizontal_kernel)

    if direction == 1:
        result = just_vertical
    else:
        result = just_horizontal

    # cv2.imshow("extracted features", vertical_extract)
    # cv2.waitKey(0)
    # cv2.destroyWindow("extracted features")

    # result = cv2.addWeighted(just_horizontal, 1, just_vertical, 0, 0) placeholder values
    return result


def getDirection(img, img_copy, threshold = 1.5):
    # initial guess
    direction = 1
    img = removeStructure(img, direction)
    # Converting to greyscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # applying Gaussian blur
    # blur = cv2.GaussianBlur(gray, (5, 5), 0)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    # experimenting with thresholding functions
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)[1]

    # print(img.shape[:3])
    lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, 50, minLineLength=min(img.shape[:2]) / threshold, maxLineGap=5)
    gradients = []
    if lines is not None:
        # print(lines)
        for l in lines:
            for line in l:
                x1, y1, x2, y2 = line
                if x2 - x1 != 0:
                    gradients.append((y2 - y1) / (x2 - x1))
                else:
                    direction = 1
                    return direction
                cv2.line(img_copy, (x1, y1), (x2, y2), (0, 0, 255), 3)
    mean_gradient = np.mean(gradients)
    if 2 > mean_gradient > -2:
        direction = 0
    else:
        direction = 1
    # cv2.imshow('detected lines', img_copy)
    # cv2.waitKey(0)
    return direction


def findContourBoxes(edged, im_copy, threshold = 22):
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # print(contours)
    counter = 1
    area = []
    coords = []
    for ctr in contours:
        x, y, w, h = cv2.boundingRect(ctr)
        area.append(w * h)
    areaMean = np.mean(area)  # average area
    # areaSD = np.std(area)  # standard deviation of area
    # print("average area =", areaMean)
    # print("areaSD = ", areaSD)
    for ctr in contours:
        x, y, w, h = cv2.boundingRect(ctr)
        area = w * h
        if area >= areaMean / threshold:  # / 20 and area <= areaMean*10):
            cv2.rectangle(im_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)
            coords.append([x, y, w, h])
            # print(counter)
            counter += 1
    if not coords:
        return ['error']
    Coords = sorted(coords, key=lambda x: (x[0]))
    # print(Coords)

    return Coords


def getOverlap(box1, box2):
    return min([box2[0] + box2[2] - box1[0], box1[0] + box1[2] - box2[0]])


def drawGroupContours(Coords,img, threshold = 5, groupAll = False):
    Coords = sorted(Coords, key=lambda x: (x[0]))
    if groupAll:
        min_x = Coords[0][0]
        min_y = sorted(Coords, key=lambda x: (x[1]))[0][1]
        groupedCoords = [[min_x,
                          min_y,
                          max([c[0]+c[2] for c in Coords])-min_x,
                          max([c[1] + c[3] for c in Coords]) - min_y]]
    else:
        groupedCoords = [Coords[0]]
        Coords.remove(Coords[0])
        for i in range(0, len(Coords)):
            box = Coords[0]
            # print('new length = ', Coords)
            if (groupedCoords[-1][0] <= box[0] <= groupedCoords[-1][0] + groupedCoords[-1][2]) or (
                    groupedCoords[-1][0] <= box[0] + box[2] <= groupedCoords[-1][0] + groupedCoords[-1][2]):
                overlap = getOverlap(groupedCoords[-1], box)
                if (overlap >= groupedCoords[-1][2] / threshold or ((groupedCoords[-1][0] <= box[0] <= groupedCoords[-1][0] +
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
                # print('box appended = ', box)
            Coords.remove(box)
    for sortedBox in groupedCoords:
            cv2.rectangle(img, (sortedBox[0], sortedBox[1]), (sortedBox[0] + sortedBox[2], sortedBox[1] + sortedBox[3]),
                          (0, 0, 255), 2)

    return groupedCoords


def cropContourBoxes(coords, img):
    img_list = []
    areas = []
    coords = sorted(coords, key=lambda x: (x[0]))
    # print('thresh = ', img.shape[0] * img.shape[1] / 70)
    for box in coords:
        symbol = img[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
        areas.append(box[3]*box[2])
        # if symbol.shape[0] * symbol.shape[1] <= img.shape[0] * img.shape[1] / 60 or sum(sum(s) for s in symbol)/255 > 0.3 * \
        #         symbol.shape[0] * symbol.shape[1]:
        #     print(symbol)
        #     print('symbol no. = ', coords.index(box))
        #     print("symbol shape = ", symbol.shape)
        #     print("symbol area = ", symbol.shape[0] * symbol.shape[1])
        #     print("colour density = ", (sum(sum(s) for s in symbol)/255) / (symbol.shape[0] * symbol.shape[1]))
        #     print("coloured area = ", sum(sum(s) for s in symbol)/255)
        #     print("yeah this shit gotta be eroded")
        #     symbol = cv2.erode(symbol, np.ones((3, 3), np.uint8))
        # elif symbol.shape[0] * symbol.shape[1] >= img.shape[0] * img.shape[1] / 20 and sum(
        #         sum(s) for s in symbol) < 0.3 * symbol.shape[0] * symbol.shape[1]:
        #     print("yeah this shit gotta be dilated")
        #     symbol = cv2.dilate(symbol, np.ones((3, 3), np.uint8))
        # symbol = cv2.morphologyEx(symbol, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        # symbol = cv2.morphologyEx(symbol, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        img_list.append(symbol)
        # print(symbol.shape)
        # visualize cropped images
        # cv2.imshow("Cropped_image", symbol)
        # cv2.waitKey(0)
        # cv2.destroyWindow("Cropped_image")
    return img_list, areas


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
        resized_img = cv2.resize(padded_img, (int(size * 0.6), int(size * 0.6)))
        resized_img = cv2.copyMakeBorder(
            resized_img,
            top=math.ceil((size - int(size * 0.6)) / 2),
            bottom=math.floor((size - int(size * 0.6)) / 2),
            left=math.ceil((size - int(size * 0.6)) / 2),
            right=math.floor((size - int(size * 0.6)) / 2),
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        resized_img = cv2.morphologyEx(resized_img, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        resized_imgs.append(resized_img)
    return resized_imgs

def segment(file, test = False):
    img = cv2.imread(file)
    im_copy = img.copy()
    result = img.copy()
    initial = img.copy()

    if test:
        cv2.imshow("original image", img)
        cv2.waitKey(0)
        cv2.destroyWindow("original image")

    initial = removeStructure(initial, 1)
    direction = getDirection(initial, im_copy)
    print(direction)
    result = removeStructure(result, direction)

    # Converting to greyscale
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(20, 20))

    # Apply CLAHE to the image
    # result = clahe.apply(gray)

    if test:
        cv2.imshow("result", result)
        cv2.waitKey(0)
        cv2.destroyWindow("result")
    # applying Gaussian blur
    # blur = cv2.GaussianBlur(gray, (5, 5), 0)
    blur = cv2.bilateralFilter(gray, 3, 75, 75)
    if test:
        cv2.imshow('blur', blur)
        cv2.waitKey(0)
    # experimenting with thresholding functions
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)[1]

    if test:
        cv2.imshow("thresh", thresh)
        cv2.waitKey(0)
        cv2.destroyWindow('thresh')

    edged = cv2.Canny(thresh, 70, 200)

    if test:
        cv2.imshow('edged', edged)
        cv2.waitKey(0)
        cv2.destroyWindow('edged')

    coords = findContourBoxes(edged, initial, 30)
    if test:
        cv2.imshow('contours', im_copy)
        cv2.waitKey(0)
    coords = drawGroupContours(coords, img)
    # cv2.imshow('grouped contours', img)
    # cv2.waitKey(0)
    # cv2.destroyWindow('grouped contours')
    imglist, areas = cropContourBoxes(coords, thresh)
    resized_imgs = resize(imglist, 50)
    # resized_imgs = resize(resized_imgs, 100)
    # print(len(coords))
    # print(coords)
    centroids = []
    for coord in coords:
        x,y,w,h = coord
        centroids.append(((x + w) / 2, (y + h) / 2))
        cv2.circle(initial,(math.floor( x + w/2), math.floor(y + h / 2)),10, (255,0,0),5 )
    if test:
        cv2.imshow('centroids', initial)
        cv2.waitKey(0)
    # print(centroids)
    if test:
        for img in imglist:
            print(img.shape)
        for img in resized_imgs:
            cv2.imshow('input', img)
            cv2.waitKey(0)
            # cv2.imwrite('images_test/input1', img)
    return resized_imgs, areas


def segmentDataset(img, test = False, groupAll = False):
    img = cv2.erode(img, np.ones((3, 3), np.uint8))
    im_copy = img
    initial = img
    initial = removeStructure(initial, 1, 1, 1)
    if test:
        cv2.imshow('initial', initial)
        cv2.waitKey(0)
    direction = getDirection(initial, im_copy, threshold = 1)
    if test:
        print('direction : ',direction)
    # img = removeStructure(img, direction, 1, 1)
    if test:
        cv2.imshow('result',img)
        cv2.waitKey(0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.erode(gray, np.ones((3, 3), np.uint8))
    blur = cv2.bilateralFilter(gray, 5, 75, 75)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)[1]
    if test:
        cv2.imshow('thresh', thresh)
        cv2.waitKey(0)
    edged = cv2.Canny(thresh, 70, 200)
    if test:
        cv2.imshow('edged', edged)
        cv2.waitKey(0)
    coords = findContourBoxes(edged, im_copy, 100)
    if test:
        cv2.imshow('contours', im_copy)
        cv2.waitKey(0)
    if coords[0] == 'error':
        return ['error']
    coords = drawGroupContours(coords,initial, 100, groupAll)
    if test:
        cv2.imshow('grouped', initial)
        cv2.waitKey(0)
    img_list, areas = cropContourBoxes(coords, thresh)
    resized_imgs = resize(img_list, 50)
    if test:
        for img in resized_imgs:
            cv2.imshow('input', img)
            cv2.waitKey(0)
    return resized_imgs

segment('Images\math.png', test = True)