from kivy.clock import Clock
from kivy.graphics import Line, Color, RoundedRectangle, Ellipse, Rectangle
from kivy.graphics.texture import Texture
from kivy.uix.button import Button
from kivymd.app import MDApp
from kivymd.uix.screen import Screen
from kivy.core.window import Window
from kivymd.uix.widget import MDWidget
from kivy.uix.image import Image
from kivymd.uix.fitimage import fitimage
from kivy.core.camera import Camera
import datetime
import os
from kivy.clock import mainthread
from kivy.lang import Builder
from kivy.properties import ObjectProperty, StringProperty, NumericProperty
from kivy.resources import resource_add_path
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.label import Label
from kivy.utils import platform
from kivy_garden.xcamera import XCamera
import cv2
from kivymd.uix.fitimage import fitimage


class camera(Camera):
    pass


class GUI(Screen):
    def __init__(self, **kwargs):
        super(GUI, self).__init__(**kwargs)
        self.TouchDirectionY = None
        self.TouchDirectionX = None
        self.touch_x = None
        self.touch_y = None
        self.current_x = None
        self.current_y = None
        self.isTouchInBox = False
        self.s = 70
        self.cropBoxHeight = self.height * 1
        self.cropBoxWidth = self.width * 1
        with self.canvas.after:
            Color(rgba=(1, 1, 1, 0.4))
            self.cropBox = RoundedRectangle(
                pos=(self.center_x - self.cropBoxWidth / 2, self.center_y + self.height * 0.1),
                size=(self.cropBoxWidth, self.cropBoxHeight),
                width=2,
            )
            self.Title = Label(
                text="CamCalc",
                # font_name=  'button',
                color=(1, 1, 1, 1)
            )
            self.cropPrompt = Label(
                text="Align your math equation within the box",
                color=(1, 1, 1),
                halign='center',
                pos=(self.center_x, self.cropBox.pos[1] + self.cropBoxHeight - self.height * 0.05),
                font_size=12
            )

    def on_size(self, *args):
        self.cropBoxHeight = self.height * 0.15
        self.cropBoxWidth = self.width * 0.65
        self.cropBox.pos = self.center_x - self.cropBoxWidth / 2, self.center_y + self.height * 0.1 - self.cropBoxHeight / 2
        self.cropBox.size = self.cropBoxWidth, self.cropBoxHeight
        self.cropPrompt.pos = self.center_x - 50, self.cropBox.pos[1] + self.cropBoxHeight - self.height * 0.05
        self.cropPrompt.halign = 'center'
        self.Title.pos = 10, self.height - self.height * 0.13
        print(self.cropPrompt.size)

    def on_touch_down(self, touch):
        self.touch_x, self.touch_y = touch.pos
        # print(touch.pos)
        # print(self.cropBox.pos[0])
        if (self.cropBox.pos[0] - 20 <= self.touch_x <= self.cropBox.pos[0] + self.cropBoxWidth + 20) and (
                self.cropBox.pos[1] - 20 <= self.touch_y <= self.cropBox.pos[1] + self.cropBoxHeight + 20):
            print("within")
            self.isTouchInBox = True
            self.current_x = self.touch_x
            self.current_y = self.touch_y
            self.TouchDirectionX = (self.touch_x - self.center_x) / abs(self.touch_x - self.center_x)
            self.TouchDirectionY = (self.touch_y - self.cropBox.pos[1] - self.cropBoxHeight / 2) / abs(
                self.touch_y - self.cropBox.pos[1] - self.cropBoxHeight / 2)
            print(self.TouchDirectionX)
        else:
            button = self.ids['button']
            button.on_release = self.capture()
            self.isTouchInBox = False
            print("someone's doing sum shit")

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

    def crop(self):
        pass

    def capture(self):
        camera = self.ids['cam']
        camera.export_to_png("input.jpg")
        print("capture")


class CamCalc(MDApp):
    pass
    # def build(self):
    # Window.size = (350,650)


CamCalc().run()