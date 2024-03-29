import cv2
from kivy.graphics import Color, RoundedRectangle
from kivy.uix.camera import Camera
from kivy.uix.label import Label
from kivymd.app import MDApp
from kivymd.uix.button import MDFillRoundFlatButton
from kivymd.uix.screen import Screen
from kivymd.uix.widget import MDWidget
import ImageSegmentationModule as sg
from tensorflow import keras
import numpy as np


class CropBox(MDWidget):
    '''As the cropping box is interactive, a seperate class needs to be created
    for the cropping box for it to be imposed on top of the camera footage, as well as for ease
    of retrieving coordinate data for cropping. '''
    def __init__(self, **kwargs):
        super(CropBox, self).__init__(**kwargs)     # inheriting the constructor from MDWidget class
        self.TouchDirectionY = None                 # initialise all variables to None as they will soon be updated
        self.TouchDirectionX = None                 # accordingly to the size of the window in on_size
        self.touch_x = None
        self.touch_y = None
        self.current_x = None
        self.current_y = None
        self.isTouchInBox = False
        self.cropBoxHeight = self.height * 1
        self.cropBoxWidth = self.width * 1
        with self.canvas:               # draws cropbox rectangle
            Color(rgba=(1, 1, 1, 0.4))
            self.cropBox = RoundedRectangle(
                pos=(self.center_x - self.cropBoxWidth / 2, self.center_y + self.height * 0.1),
                size=(self.cropBoxWidth, self.cropBoxHeight),
                width=2,
            )
            self.cropPrompt = Label(        # text prompt
                text="Align your math equation within the box",
                color=(1, 1, 1),
                halign='center',
                pos=(self.center_x, self.cropBox.pos[1] + self.cropBoxHeight - self.height * 0.05),
                font_size=12
            )

    def on_size(self, *args):           # updates the dimensions of the box dynamically.
        self.clear_widgets()
        self.cropBoxHeight = self.height * 0.3
        self.cropBoxWidth = self.width * 0.8
        self.cropBox.pos = self.center_x - self.cropBoxWidth / 2, self.center_y + self.height * 0.1 - self.cropBoxHeight / 2
        self.cropBox.size = self.cropBoxWidth, self.cropBoxHeight
        self.cropPrompt.pos = self.center_x - 50, self.cropBox.pos[1] + self.cropBoxHeight - self.height * 0.05
        self.cropPrompt.halign = 'center'
        print(self.cropPrompt.size)

# Locates and compares the location of the touch event on the screen
    def on_touch_down(self, touch):

        # determines if the touch is within the box
        self.touch_x, self.touch_y = touch.pos
        if (self.cropBox.pos[0] - 20 <= self.touch_x <= self.cropBox.pos[0] + self.cropBoxWidth + 20) and (
                self.cropBox.pos[1] - 20 <= self.touch_y <= self.cropBox.pos[1] + self.cropBoxHeight + 20):
            print("within")
            self.isTouchInBox = True
            self.current_x = self.touch_x
            self.current_y = self.touch_y

            # determines the direction which the box will be expanding/shrinking depending on the position of the
            # initial touch event
            self.TouchDirectionX = (self.touch_x - self.center_x) / abs(self.touch_x - self.center_x)
            self.TouchDirectionY = (self.touch_y - self.cropBox.pos[1] - self.cropBoxHeight / 2) / abs(
                self.touch_y - self.cropBox.pos[1] - self.cropBoxHeight / 2)
            print(self.TouchDirectionX)
        else:
            self.isTouchInBox = False

    def on_touch_move(self, touch):
        if self.isTouchInBox:
            self.cropBoxWidth += self.TouchDirectionX * (touch.pos[0] - self.current_x) * 2
            self.cropBoxHeight += self.TouchDirectionY * (touch.pos[1] - self.current_y) * 2
            self.cropBox.pos = self.center_x - self.cropBoxWidth / 2, self.center_y + self.height * 0.1 - self.cropBoxHeight / 2
            self.cropBox.size = self.cropBoxWidth, self.cropBoxHeight
            self.cropPrompt.pos = self.center_x - 50, self.cropBox.pos[1] + self.cropBoxHeight - self.height * 0.05
            print("delta x = ", touch.pos[0] - self.current_x)
            self.current_x = touch.pos[0]
            self.current_y = touch.pos[1]
            print("dragging")


class GUI(Screen):
    def __init__(self, **kwargs):
        super(GUI, self).__init__(**kwargs)
        self.camera = None
        self.clear_widgets()
        self.s = 70
        # self.camera = Camera(resolution=self.size, size=self.size, allow_stretch=True, play=True, index=0)
        self.crop = CropBox()
        with self.canvas.after:
            self.Title = Label(
                text="CamCalc",
                # font_name=  'button',
                color=(1, 1, 1, 1)
            )

    def crop(self):
        pass

    def on_size(self, *args):
        self.clear_widgets()
        self.Title.pos = 10, self.height - self.height * 0.13
        #self.camera = Camera(resolution=self.size, size=self.size, allow_stretch=True, play=True, index=0)
        self.camera = Camera(allow_stretch=True, play=True, index=0)
        self.add_widget(self.camera)
        self.add_widget(self.crop)
        self.add_widget(MDFillRoundFlatButton(size=(70, 70), pos=(self.center_x - 35, self.y + 50), text='Capture',
                                              on_release=lambda x: self.capture())
                        )

    def capture(self):
        self.camera.export_to_png("input.jpg")
        img = cv2.imread('input.jpg')
        cv2.imshow('img', img)
        print('img proportions : ', img.shape[0], img.shape[1])
        print('window proportions : ', self.width, self.height)
        print(self.crop.cropBox.pos[0], self.crop.cropBox.pos[1])
        print(img.shape[1], self.crop.cropBox.pos[1])
        img = img[img.shape[0] - int(self.crop.cropBox.pos[1]) - int(self.crop.cropBoxHeight): img.shape[0] - int(
            self.crop.cropBox.pos[1]),
              int(self.crop.cropBox.pos[0]):int(self.crop.cropBox.pos[0] + self.crop.cropBoxWidth)]
        cv2.imwrite('croppedinput.jpg', img)
        print("capture")
        model = keras.models.load_model('CNN_symbols')
        imgs, areas, aspect_ratios = sg.segment('croppedinput.jpg', test = True, Contour_thresh=40)
        img_array = keras.preprocessing.image.img_to_array(imgs)
        predicted_labels = model.predict(img_array)
        predicted_values = [max(p) for p in predicted_labels]
        predicted_labels = predicted_labels.argmax(axis=1)
        classNames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'add', 'div', 'eq', 'mul', 'sub', 'x']
        ans = [classNames[i] for i in predicted_labels]
        symbols = ['add', 'div', 'eq', 'mul', 'sub']
        for i in range(0, len(ans)):
            if ans[i] == 'mul' and ans[i + 1] in symbols:
                ans[i] = 'x'
            if ans[i]=='6' and areas[i] <= np.mean(areas) / 2.2:
                ans[i] = 'div'
            if (ans[i] == '2' or ans[i] =='8') and areas[i] <= np.mean(areas) / 2.2:
                ans[i] = 'eq'
            if predicted_values[i]<0.65:
                print('error. Retaking of photo is advised')
        print(np.mean(areas))
        print(areas)
        print(ans)
        print(predicted_values)

        # inputdata = sg.segment('croppedinput.jpg')
        # inputdata = np.array(inputdata)/255
        # print(inputdata)
        # print(inputdata.shape)
        # model = keras.models.load_model('CNN')
        # y_predicted = model.predict(inputdata)
        # print(y_predicted.argmax(axis=1))


class CamCalc(MDApp):
    pass


CamCalc().run()
