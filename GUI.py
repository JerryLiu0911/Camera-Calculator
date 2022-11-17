from kivymd.app import MDApp
from kivymd.uix.button import MDRaisedButton
from kivy.graphics import Rectangle, Color
from kivy.uix.widget import Widget
from kivy.clock import Clock
import cv2
from kivy.core.window import Window
from kivy.uix.button import Button
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.uix.floatlayout import FloatLayout
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
import time
# Builder.load_string('''
# <CameraClick>:
#     orientation: 'vertical'
#     Camera:
#         id: camera
#         resolution: (640, 480)
#         play: True
#     # ToggleButton:
#     #     text: 'Play'
#     #     on_press: camera.play = not camera.play
#     #     size_hint_y: None
#     #     height: '48dp'
#     Button:
#         text: 'Capture'
#         size_hint_y: None
#         height: '48dp'
#         on_press: root.capture()
# ''')
#
#
# class CameraClick(BoxLayout):
#     def __init__(self,**kwargs):
#         super().__init__(**kwargs)
#         Window.size = (300,600)
#     def capture(self):
#         '''
#         Function to capture the images and give them the names
#         according to their captured time and date.
#         '''
#         camera = self.ids['camera']
#         timestr = time.strftime("%Y%m%d_%H%M%S")
#         camera.export_to_png("IMG_{}.png".format(timestr))
#         print("Captured")
#
#
# class TestCamera(App):
#
#     def build(self):
#         return CameraClick()


#TestCamera().run()
class Camera(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        x = 1
    def drawRectangle(self):
        self.canvas.add(Color(self.x/10,0.4,0.7, 0.4))
        self.canvas.add(Rectangle( pos = (self.center_x-100, self.center_y-100), pos_hint = {'center_x': .5, 'center_y':.5}, size = (200,200)))
class CropBox(Widget):
    pass
class CamCalc(MDApp):

    def build(self):
        Window.size=(300,600)
        layout = FloatLayout()
        self.image = Image()
        layout.add_widget(self.image)
        self.captureButton = MDRaisedButton(
            text="Capture",
            pos_hint = {'center_x' : 0.5, 'center_y': 0.1},
            size_hint = (0.3, 0.06),
        )
        self.captureButton.bind(on_press = self.takePhoto())
        layout.add_widget()

        layout.canvas.add(Color(1,1,1,0.4))
        layout.canvas.add(
            Rectangle(
                #size = (200, 80),
                size_hint = (0.7, 0.06),
                pos = (50, 350)
            )
        )
        CropBox()
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.load_video, 1.0/30.0)
        self.captureButton.bind(on_press=self.takePhoto())
        return layout

    def load_video(self, *args):
        ret, frame = self.capture.read()
        #initializing frame
        self.image_frame = frame
        buffer = cv2.flip(frame,0).tostring()
        texture = Texture.create(colorfmt='bgr', size=(frame.shape[1], frame.shape[0]))
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt = 'ubyte')
        self.image.texture = texture
        return frame

    def takePhoto(self, *args):
        image_name = "pic"
        cv2.imwrite(image_name, self.image_frame)


CamCalc().run()
