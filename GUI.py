import kivy
from kivy.app import App
from kivy.graphics import Rectangle, Color
from kivy.uix.widget import Widget
from kivy.clock import Clock
import cv2
from kivy.uix.button import Button
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.uix.floatlayout import FloatLayout
from kivy.core.window import Window
class Camera(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        x = 1
    def drawRectangle(self):
        self.canvas.add(Color(self.x/10,0.4,0.7, 0.4))
        self.canvas.add(Rectangle( pos = (self.center_x-100, self.center_y-100), pos_hint = {'center_x': .5, 'center_y':.5}, size = (200,200)))
class CropBox(Widget):
    pass
class CamCalc(App):

    def build(self):
        Window.size=(300,600)
        layout = FloatLayout()
        self.image = Image()
        layout.add_widget(self.image)
        self.captureButton = Button(
            text="Capture",
            pos_hint = {'center_x' : 0.5, 'center_y': 0.1},
            size_hint = (0.3, 0.06),
        )
        layout.add_widget(self.captureButton)
        layout.canvas.add(Color(1,1,1,0.4))
        layout.canvas.add(
            Rectangle(
                size = (200, 80),
                size_hint = (0.7, 0.06),
                pos = (50, 350)
            )
        )
        CropBox()
        #self.captureButton.bind(on_press = self.takePhoto())
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.load_video, 1.0/30.0)
        return layout

    def load_video(self, *args):
        ret, frame = self.capture.read()
        #initializing frame
        self.image_frame = frame
        buffer = cv2.flip(frame, 0).tostring()
        texture = Texture.create(colorfmt='bgr', size=(frame.shape[1], frame.shape[0]))
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt = 'ubyte')
        self.image.texture = texture

    def takePhoto(self, *args):
        image_name = "pic"
        cv2.imwrite(image_name, self.image_frame)


CamCalc().run()
