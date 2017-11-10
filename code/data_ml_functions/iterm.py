# if you have iterm2 for osx (www.iterm2.com) this is a like print(...) for images in the console

import base64
import cStringIO
import numpngw

def show_image(a):
    png_array = cStringIO.StringIO()
    numpngw.write_png(png_array, a)
    encoded_png_array = base64.b64encode(png_array.getvalue())
    png_array.close()
    image_seq = '\033]1337;File=[width=auto;height=auto;inline=1]:'+encoded_png_array+'\007'
    print(image_seq)