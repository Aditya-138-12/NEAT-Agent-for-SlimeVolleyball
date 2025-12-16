import time
from pynput.keyboard import Key, Controller

keyboard = Controller()

print("Pressing keys in loop now")

while True:
    keyboard.press('w')
    keyboard.release('w')

