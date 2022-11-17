from kivymd.app import App
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
Builder.load_string('''
<CameraClick>:
    orientation: 'vertical'
    Camera:
        id: camera
        resolution: (640, 480)
        play: True
    # ToggleButton:
    #     text: 'Play'
    #     on_press: camera.play = not camera.play
    #     size_hint_y: None
    #     height: '48dp'
    Button:
        text: 'Capture'
        size_hint_y: None
        height: '48dp'
        on_press: root.capture()
''')


class CameraClick(BoxLayout):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        Window.size = (300,600)
    def capture(self):
        '''
        Function to capture the images and give them the names
        according to their captured time and date.
        '''
        camera = self.ids['camera']
        timestr = time.strftime("%Y%m%d_%H%M%S")
        camera.export_to_png("IMG_{}.png".format(timestr))
        print("Captured")


class TestCamera(App):

    def build(self):
        return CameraClick()


TestCamera().run()
