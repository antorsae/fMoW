# if you have iterm2 for osx (www.iterm2.com) this is a like print(...) for images in the console

import base64
#import cStringIO
from io import StringIO
import numpngw
import numpy as np

def show_image(a):
    if a.dtype != np.uint8:
        a = a.astype(np.uint8)
    png_array = StringIO.StringIO()
    numpngw.write_png(png_array, a)
    encoded_png_array = base64.b64encode(png_array.getvalue())
    png_array.close()
    image_seq = '\033]1337;File=[width=auto;height=auto;inline=1]:'+encoded_png_array+'\007'
    print(image_seq)
