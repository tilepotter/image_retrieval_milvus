# -*- coding: utf-8 -*-
import base64

if __name__ == '__main__':
    img_path = "D:/Download/Train_Images_Set/set01_500/2007_007080.jpg"
    f = open(img_path, 'rb')
    # base64编码
    base64_data = base64.b64encode(f.read())
    f.close()
    base64_data = base64_data.decode()
    print(base64_data)
