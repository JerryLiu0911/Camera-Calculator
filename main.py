from kivy.animation import Animation
from kivy.utils import platform
from kivy.graphics import Color, RoundedRectangle
from kivy.properties import ObjectProperty, StringProperty
from kivy.uix.camera import Camera
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivymd.app import MDApp
import re
import os
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager
import cv2
from kivymd.uix.boxlayout import BoxLayout, MDBoxLayout
from kivymd.uix.button import MDFlatButton
from kivymd.uix.dialog import MDDialog
from kivymd.uix.floatlayout import MDFloatLayout
from kivymd.uix.screen import MDScreen
from kivymd.uix.widget import MDWidget
import ImageSegmentationModule as sg
import numpy as np
import tensorflow as tf

Builder.load_string(
    '''

<CropBox>

<CustomCamera>

<CameraClick>:
    size: root.width, root.height
    display: Display
    FloatLayout:
        CustomCamera:
            id: camera
            resolution: root.size
            allow_stretch: False
            # keep_ratio: True
            play: True

        MDIconButton:
            icon: "information"
            # md_bg_color: app.theme_cls.primary_color
            pos_hint: {"center_x": 0.1, "center_y": 0.9}
            on_press: 
                root.show_information_dialog()

        MDFloatingActionButton:
            icon: "camera"
            md_bg_color: app.theme_cls.primary_color
            pos_hint: {"center_x": 0.5, "center_y": 0.1}
            on_press: 
                root.when_pressed()
                camera.play = False
            on_release: 
                root.ids['Display'].open()
        BoxLayout:
            orientation : 'vertical'
            CropBox:
                id:crop
        Display:
            id: Display 

<MainScreen>:
    CameraClick:
        id: camera_click 

<Results>:
    img: image
    MDGridLayout
        cols: 1
        adaptive_height: True
        padding: 0, 8
        OneLineAvatarListItem:
            id: second
            text: "Segmented Numbers"
            _no_ripple_effect: True
        Image:
            id: image
            size_hint_y: None
            keep_ratio: True
            height: "144dp"  #make adaptive
            source: 'byCapture.jpg'
        OneLineListItem:
            text: "Recognized Numbers"
            _no_ripple_effect: True
        OneLineAvatarIconListItem:
            height: "70dp"
            size_hint_y: None
            text: app.string
            font_style: "H5"
            text_size: 80
            _no_ripple_effect: True
            IconRightWidget:
                id: "pen"
                icon: "pencil"
                on_press: app.show_confirmation_dialog()
        OneLineListItem:
            text: "Solution"
            _no_ripple_effect: True
        OneLineAvatarIconListItem:
            height: "70dp"
            size_hint_y: None
            text: app.solution
            font_style: "H5"
            text_size: 80
            _no_ripple_effect: True
            IconRightWidget:
                icon: "calculator"
                _no_ripple_effect: True
<Display>:
    results: results
    MDCard:
        id: card
        md_bg_color:
            app.theme_cls.bg_normal
        orientation: "vertical"
        size_hint_y: None
        height: root.height
        padding: [0, 0, 0, 0]
        radius: [24, 24, 0, 0]
        pos:(0,-root.height)
        OneLineAvatarListItem:
            id: header_button
            text: 'Return to camera'
            divider: None
            _no_ripple_effect: True
            on_press: 
                app.root.ids.main_screen.ids.camera_click.ids.camera.play = True
                root.close()
            pos_hint: {'top' :1}
            IconLeftWidget:
                icon: "close"
                on_press:
                    app.root.ids.main_screen.ids.camera_click.ids.camera.play = True
                    root.close()
                _no_ripple_effect: True
        MDBoxLayout:
            id: front_layer
            padding: 0, 0, 0, "10dp"
            Results:
                id: results

<InputBoxContents>:
    textField: textField
    id: InputBoxContents
    orientation: "vertical"
    spacing: "12dp"
    size_hint_y: None
    height: "50dp"
    # string: app.root.ids.Display.ids.card.ids.front_layer.ids.Layer.ids.layout.ids.recognized.text
    MDTextField:
        id: textField
        # hint_text: "Recognized Numbers: "
        text: app.string


<GUI>:
    id: GUI
    # transition: NoTransition()
    MainScreen:
        id: main_screen
        name: 'Main Screen'
        manager: GUI
'''
)


class Display(MDFloatLayout):
    results = ObjectProperty(None)
    isOpen = False

    def open(self):
        if self.isOpen:
            return self.close()
        else:
            Animation(y=-self.height * 0.1, d=0.2, t="out_quad").start(
                self.ids.card
            )
            # self.isOpen = True

    def close(self):
        Animation(y=-self.height, d=0.2, t="out_quad").start(
            self.ids.card
        )
        self.isOpen = False


class MainScreen(MDScreen):
    pass


class CustomCamera(Camera):
    def on_auth_status(self, general_status, status_message):
        if general_status == 'provider-enabled':
            pass
        else:
            self.open_permissison_popup()

    def open_permission_popup(self):
        self.dialog = MDDialog(
            title="Camera Error",
            text="Camera permissions should be enabled"
        )
        self.dialog.open()

    def get_auth(self, ):
        # get permissions from camera
        if platform == 'android':
            from android.permissions import Permission, request_permissions
            from jnius import autoclass

            self.index = autoclass('android.hardware.Camera2$CameraInfo').CAMERA_FACING_BACK

            def callback(permission, results):
                if all([res for res in results]):
                    print("Got all permissions")
                else:
                    print("Permission error")

            request_permissions([Permission.CAMERA, Permission.WRITE_EXTERNAL_STORAGE])


class CropBox(MDWidget):

    def __init__(self, **kwargs):
        super(CropBox, self).__init__(**kwargs)  # inheriting the constructor from MDWidget class
        self.TouchDirectionY = None  # initialize all variables to None as they will soon be updated
        self.TouchDirectionX = None  # accordingly to the size of the window in on_size
        self.touch_x = None
        self.touch_y = None
        self.current_x = None
        self.current_y = None
        self.isTouchInBox = False
        self.cropBoxHeight = self.height * 1
        self.cropBoxWidth = self.width * 1
        self.corners = [self.center_x - self.cropBoxWidth / 2, self.center_y + self.height * 0.1 + self.cropBoxHeight,
                        self.center_x + self.cropBoxWidth / 2, self.center_y + self.height * 0.1]
        with self.canvas:  # draws cropbox rectangle
            Color(rgba=(1, 1, 1, 0.4))
            self.cropBox = RoundedRectangle(
                pos=(self.corners[0], self.corners[3]),
                size=(self.cropBoxWidth, self.cropBoxHeight),
                width=2,
            )
            # self.corner1 = Triangle(
            #     points=[self.corners[0], self.corners[3],
            #             self.corners[0] + self.width * 0.1]
            # )
            self.cropPrompt = Label(  # text prompt
                text="Align your math equation within the box",
                color=(0, 0, 0, 1),
                halign='center',
                pos=(self.center_x, self.cropBox.pos[1] + self.cropBoxHeight - self.height * 0.05),
                font_size=12
            )

    def on_size(self, *args):  # updates the dimensions of the box dynamically.
        self.clear_widgets()
        self.cropBoxHeight = self.height * 0.3
        self.cropBoxWidth = self.width * 0.8
        self.cropBox.pos = self.center_x - self.cropBoxWidth / 2, self.center_y + self.height * 0.1 - self.cropBoxHeight / 2
        self.cropBox.size = self.cropBoxWidth, self.cropBoxHeight
        self.cropPrompt.pos = self.center_x - 50, self.cropBox.pos[1] + self.cropBoxHeight - self.height * 0.05
        self.cropPrompt.halign = 'center'

    # Locates and compares the location of the touch event on the screen
    def on_touch_down(self, touch):

        # determines if the touch is within the box
        self.touch_x, self.touch_y = touch.pos
        if (self.cropBox.pos[0] - 20 <= self.touch_x <= self.cropBox.pos[0] + self.cropBoxWidth + 20) and (
                self.cropBox.pos[1] - 20 <= self.touch_y <= self.cropBox.pos[1] + self.cropBoxHeight + 20):
            self.isTouchInBox = True
            self.current_x = self.touch_x
            self.current_y = self.touch_y

            # determines the direction which the box will be expanding/shrinking depending on the position of the
            # initial touch event
            self.TouchDirectionX = (self.touch_x - self.center_x) / abs(self.touch_x - self.center_x)
            self.TouchDirectionY = (self.touch_y - self.cropBox.pos[1] - self.cropBoxHeight / 2) / abs(
                self.touch_y - self.cropBox.pos[1] - self.cropBoxHeight / 2)
        else:
            self.isTouchInBox = False

    def on_touch_move(self, touch):
        if self.isTouchInBox:
            self.cropBoxWidth += self.TouchDirectionX * (touch.pos[0] - self.current_x) * 2
            self.cropBoxHeight += self.TouchDirectionY * (touch.pos[1] - self.current_y) * 2
            self.cropBox.pos = self.center_x - self.cropBoxWidth / 2, self.center_y + self.height * 0.1 - self.cropBoxHeight / 2
            self.cropBox.size = self.cropBoxWidth, self.cropBoxHeight
            self.cropPrompt.pos = self.center_x - 50, self.cropBox.pos[1] + self.cropBoxHeight - self.height * 0.05
            self.current_x = touch.pos[0]
            self.current_y = touch.pos[1]


class CameraClick(BoxLayout):
    display = ObjectProperty(None)
    img = np.ones((50, 50, 3))

    def __init__(self, **kwargs):
        super(CameraClick, self).__init__(**kwargs)
        self.croppedimage = None
        self.crop = None
        self.ratio = None
        self.cropCoords = None
        self.texture = None
        self.camera = None
        self.img_data = None

    def show_information_dialog(self):
        self.dialog = MDDialog(
            text="Acceptable inputs:\n "
                 "\n   -integers"
                 "\n   -symbols: +, ÷, -, =, *"
                 "\n   -variable x\n"
                 "\nFor the most accurate results,\n"
                 "\n   -The equation should fill as much of the input box as possible"
                 "\n   -The input box should contain nothing other than the desired equation"
                 "\n   -A clear background with good lighting, as well as clear and visible ink will lead to the best results."
        )
        self.dialog.open()

    def capture(self):
        '''
        Function to capture the images and give them the names
        according to their captured time and date.
        '''
        self.camera = self.ids['camera']
        self.camera.export_to_png(
            os.path.join(os.getcwd(),
                         'byCapture.jpg'))  # returns what is essentially a screenshot of the window with the edges transparent.
                                            # although functionally the same and arguably more efficient at capturing
        self.texture = self.camera.texture  # It requires post-processing to remove the edges of the image
        height, width = self.texture.height, self.texture.width
        img_data = np.frombuffer(self.texture.pixels, dtype=np.uint8)
        img_data = img_data.reshape(height, width, 4)
        self.img = cv2.cvtColor(img_data, cv2.COLOR_BGRA2RGB)
        self.ratio = self.img.shape[1] / self.camera.resolution[0]
        self.crop = self.ids['crop']
        self.cropCoords = np.array([self.crop.cropBox.pos[0] - (self.camera.resolution[0] - self.img.shape[1]) // 2,
                                    self.crop.cropBox.pos[1] - (self.camera.resolution[1] - self.img.shape[0]) // 2,
                                    self.crop.cropBoxWidth + self.crop.cropBox.pos[0] - (
                                            self.camera.resolution[0] - self.img.shape[1]) // 2,
                                    self.crop.cropBoxHeight + self.crop.cropBox.pos[1] - (
                                            self.camera.resolution[1] - self.img.shape[0]) // 2])
        '''
        This part caused confusion as Kivy defines (0, 0) as the bottom left corner of the window, whereas OpenCV defines
        (0, 0) as the top left corner of the window. The coordinates here therefore show using the kivy coordinate system
        in this order, (leftmost x, bottom y, right x, top y). 
        '''
        # print("Shape of image = ", cv2.imread(os.path.join(os.getcwd(),'byCapture.jpg')))
        # print(f"window resolution : {self.height}, {self.width}")
        # print("camera resolution : ", self.camera.resolution)
        # print("texture shape : ", self.img.shape)
        # print(self.cropCoords)
        cv2.imwrite(os.path.join(os.getcwd(), 'texture.jpg'), self.img)
        cv2.waitKey(0)

        self.img = self.img[int(self.img.shape[0] - self.cropCoords[3]):
                            int(self.img.shape[0] - self.cropCoords[1]),
                   int(self.cropCoords[0]): int(self.cropCoords[2])]
        cv2.imwrite(os.path.join(os.getcwd(), 'croppedinput.jpg'), self.img)

    def when_pressed(self):
        self.capture()
        image, areas, aspect_ratios = sg.segment(os.path.join(os.getcwd(), 'croppedinput.jpg'), Contour_thresh=30)
        print("areas", areas)
        img_array = np.reshape(image, [len(image), 50, 50, 1])
        img_array = np.array(img_array, np.float32)
        self.display.results.img.reload()
        predictions = []
        for input in img_array:
            input = input.reshape(1, input.shape[0], input.shape[1], input.shape[2])
            model.set_tensor(model.get_input_details()[0]['index'], input)
            model.invoke()
            prediction = model.get_tensor(model.get_output_details()[0]['index'])
            predictions.append(prediction)
        predictions = np.reshape(predictions, (len(predictions), 16))
        app.setString(
            ''.join(self.contextClassification(predictions, areas, aspect_ratios)))  # concatenates array to string

    def get_img(self):
        return self.img

    def contextClassification(self, predictions, areas, aspect_ratios):
        '''Context assisted classification : after taking in the predicted values from the neural network, factors such as
        size of the cropping box and mathematical syntax are taken into account to correct misclassified symbols.'''

        predictedIndex = predictions.argmax(axis=1)
        certainty = [np.max(p) for p in predictions]
        classNames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '÷', '=', '*', '-', 'x']
        ans = [classNames[i] for i in predictedIndex]
        symbols = ['+', '÷', '=', '*', '-']
        for i in range(0, len(ans)):
            if ans[i] == '4' or ans[i] == '8' and areas[i] <= np.max(areas) / 1.8 and 0.75 < aspect_ratios[i] < 1.2 and certainty[i]<=0.8:
                ans[i] = 'x'
            if (ans[i] == '8' or ans[i] == '2') and areas[i] <= np.mean(areas) / 3:
                ans[i] = '='
            if ans[i] == '6' and areas[i] <= np.mean(areas) / 2 and 0.8 < aspect_ratios[i] < 1.2:
                ans[i] = '='

        if ans.count('=') >= 2:
            currentMinEq = 0  # stores the index of the equal sign which has the lowest certainty
            currentMinCert = 2  # stores the certainty of the lowest certain equal sign
            max = 0
            secondMaxIndex = 0
            maxIndex = 0
            for i in range(0, len(ans)):
                if ans[i] == '=':
                    if certainty[i] < currentMinCert:
                        currentMinEq = i
                        currentMinCert = certainty[i]
            for i in range(0, len(symbols)):  # Searches for a second possible symbol
                if predictions[currentMinEq][i + 9] > max:
                    max = predictions[currentMinEq][i]
                    secondMaxIndex = maxIndex
                    maxIndex = i
            ans[currentMinEq] = classNames[maxIndex + 9]
            if classNames[maxIndex + 9] == '=':
                ans[currentMinEq] = classNames[secondMaxIndex + 9]

        for k in range(0, len(ans) - 1):
            if ans[k] == '*' and ans[k + 1] in symbols:
                ans[k] = 'x'
            elif ans[k] in symbols and ans[k + 1] == '*':
                ans[k + 1] = 'x'
            if ans[k] in symbols and ans[k + 1] in symbols:
                ans = ["Unclear image. Please retake photo."]
                break

        return ans


class Results(ScrollView):
    img = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(Results, self).__init__(**kwargs)


class GUI(ScreenManager):
    pass


class InputBoxContents(MDBoxLayout):
    textField = ObjectProperty(None)

    # layout = MyBackdropFrontLayer()

    def __init__(self, **kwargs):
        super(InputBoxContents, self).__init__(**kwargs)


class CamCalc(MDApp):
    string = StringProperty("x+1=0")
    solution = StringProperty("x=-1")

    def build(self):
        self.theme_cls.theme_style = "Light"
        self.theme_cls.primary_palette = 'Orange'

        # self.GUI.ids.main_screen.ids.Display.open()
        return GUI()

    def show_confirmation_dialog(self):
        self.dialog = MDDialog(
            title="Recognized Numbers:",
            type="custom",
            content_cls=InputBoxContents(),
            buttons=[
                MDFlatButton(
                    text="CANCEL",
                    theme_text_color="Custom",
                    text_color=self.theme_cls.primary_color,
                    on_release=self.close_confirmation_dialogue
                ),
                MDFlatButton(
                    text="OK",
                    theme_text_color="Custom",
                    text_color=self.theme_cls.primary_color,
                    on_release=self.confirm_dialog
                ),
            ],
        )
        self.dialog.open()

    def close_confirmation_dialogue(self, obj):
        self.dialog.dismiss()

    def confirm_dialog(self, obj):
        app.setString(self.dialog.content_cls.textField.text)
        # self.contents = InputBoxContents()
        self.dialog.dismiss()

    def setString(self, input):
        self.string = input
        self.solve(input)

    def solve(self, txt):
        txt = txt.replace(' ', '')
        txt = txt.replace('÷', '/')
        symbols = "[-+/*=]"

        def getCoefficients(string):
            xCoefficient = ''.join(re.findall(r'(?:(?:[0-9]+[*/])+|(?:[+-]))?x[*/]?[0-9]*', string))  # finds all x's

            constant = ''.join(re.sub(r'(?:(?:[0-9]+[*/])+|(?:[+-]))?x[*/]?[0-9]*', '',
                                      string))  # removes x's from original equation to leave constants
            if constant == '':
                constant = 0
            else:
                if constant[0] == '*' or constant[0] == '/':
                    constant = '1' + constant
                constant = eval(constant)

            if string[0] == 'x':
                xCoefficient = xCoefficient + '+1'
            elif xCoefficient == '':
                return 0, constant

            xCoefficient = ''.join(
                re.sub(f"({symbols})x", r'\1(1)', xCoefficient))  # replaces x with symbols in front of them with 1
            xCoefficient = ''.join(re.sub("x", '', xCoefficient))  # removes remaining x's and calculates coefficient
            xCoefficient = eval(xCoefficient)
            return xCoefficient, constant

        if len(re.findall("=", txt)) > 1:
            self.solution = "ERROR: Multiple = signs detected."
        elif not re.findall('x', txt):
            if txt[-1] != "=" and '=' in txt:
                self.solution = "ERROR: For calculations, do not leave anything after ="
            else:
                txt = txt.replace('=', '')
                try:
                    self.solution = str(eval(txt))
                except:
                    self.solution = "ERROR: Invalid expression"
        else:
            equation = re.split("=", txt)
            if len(equation) < 2:
                self.solution = "ERROR: Invalid expression"
                return None
            # print(f"{symbols}{symbols}*")
            lhs = equation[0]
            rhs = equation[1]
            if re.findall("[0-9]/[0-9]*x", txt):
                self.solution = "ERROR: Only linear equations. For fractions use ax/b."
            elif re.findall(f"{symbols}{symbols}+", lhs) or re.findall(f"{symbols}{symbols}+", lhs):
                self.solution = "ERROR: Consecutive symbols"
            elif re.findall("x[0-9]", txt):
                self.solution = "ERROR: Coefficients should be in front of x"
            else:
                '''using ax + b = cx +d '''
                a, b = getCoefficients(lhs)
                c, d = getCoefficients(rhs)
                if (a - c) == 0:
                    self.solution = "ERROR: Zero error occurred."
                else:
                    x = (d - b) / (a - c)
                    self.solution = 'x = ' + str(x)


app = CamCalc()
model = tf.lite.Interpreter(os.path.join(os.getcwd(), 'CNN.tflite'))
model.allocate_tensors()

app.run()
