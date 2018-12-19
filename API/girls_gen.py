import os  # ファイルやディレクトリを扱えるようにする
from flask import Flask, request, redirect, url_for, flash, render_template
from flask import jsonify, abort, make_response, Response
from keras.models import Sequential, load_model
import keras
import sys
import numpy as np
from PIL import Image
import keras.backend as K

SAVE_DIR = "/tmp/"
z_dim = 100
# 自身の名称をappという名前でインスタンス化
app = Flask(__name__)

# curl -X GET -o result.jpg http: // 0.0.0.0: 5050/api_dragan


@app.route('/api_dragan', methods=['GET'])
def api_dragan():
    model = load_model('static/model-1000-epoch.h5')
    noise = np.random.normal(size=(1, z_dim)).astype('float32')
    gen_imgs = model.predict(noise)
    gen_imgs = ((gen_imgs - gen_imgs.min()) * 255 /
                (gen_imgs.max() - gen_imgs.min())).astype(np.uint8)
    gen_imgs = gen_imgs.reshape(1, -1, 64, 64, 3).swapaxes(1,
                                                           2).reshape(1*64, -1, 3)
    Image.fromarray(gen_imgs).save(SAVE_DIR + 'result.jpg')
    f = open(SAVE_DIR + 'result.jpg', 'rb')
    image = f.read()
    return Response(response=image, content_type='image/jpeg')


''' Flask起動 '''
if __name__ == '__main__':
    '''already in use? '''
    app.run(debug=True, host='0.0.0.0', port=5050)
