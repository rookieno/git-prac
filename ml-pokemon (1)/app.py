from flask import Flask, render_template, request, redirect, url_for, jsonify
from datetime import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from uuid import uuid4
import numpy as np
import os

# 학습시킨 binary classification model 불러오기 (출력층을 sigmoid 로 설정했기에, predict 하면 아웃풋이 0~1 로 나옴)
model = tf.keras.models.load_model('static/model/model.h5')
# 해당 모델은 아웃풋이 0이면 고양이, 1이면 강아지라고 판별한 것
# 아웃풋이 어떤지는 모델 생성 시 출력층을 어떻게 구성했는지에 따라 얼마든지 달라질 수 있음에 유의
# 모델 생성 시 출력층을 softmax 로 설정했다면 카테고리 갯수만큼 아웃풋이 나올 것
# 모델 생성 시 출력층을 sigmoid 로 설정했다면 0~1로 아웃풋이 나올 것
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/test', methods=['POST'])
def test():
    file = request.files['file_give']
    image_name = 'test.jpg'
    file.save('./static/upload_img/pokemon/' + image_name)
    return jsonify({'result': 'success'})


@app.route('/result')
def result():
    pokemon_list = ['이상해씨', '버터플 이', '파이리', '삐삐', '탕구리', '디그다', '메타몽 이', '망나뇽 이', '파오리', '팬텀 이', '푸린 이', '또가스', '내루미', '괴력몬',
                    '잉어킹 이', '나옹 이', '뮤', '뚜벅초', '피죤 이', '피카츄', '고라파덕 이', '꼬렛 이', '모래두지', '야도란 이', '잠만보', '꼬부기']
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_dir = './static/upload_img'
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        color_mode="rgb",
        shuffle=False,
        class_mode=None,
        batch_size=1)
    pred = model.predict(test_generator)
    print(pred)
    a = int(np.argmax(pred[-1]))
    print(type(a))
    print(a)
    name = pokemon_list[a]
    per = pred[0][a] * 100
    percent = '{:.2f}%'.format(per)
    path = f'../static/img/result/{a}.png'
    return render_template('result.html', name=name, percent=percent, path=path)


if __name__ == '__main__':
    app.run('0.0.0.0', port=5000, debug=True)