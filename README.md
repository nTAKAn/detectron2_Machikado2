# 続・detectron2 for まちカドまぞく

前編である「detectron2 for まちカドまぞく」にカスタム機能を追加して精度を高めます。

---

今回は本当の推論画像です。上手くいったのでドヤ顔にしてみました。

<img src=https://user-images.githubusercontent.com/33882378/79108398-34fe8400-7db1-11ea-9b26-08e09e13243f.jpg>

---
## 1.使い方

### (1) 

### (2) detectron2_Machikado2 は detectron2 ディレクトリの下に clone してください。
detectron2 を git clone して出来た detectron2 ディレクトリの直下で clone してください。別に変えてもいいけど、その場合ディレクトリ関係のパスは修正してください。

### (3) weight ファイルをダウンロードします。

https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md からダウンロードし、coco_models/ へコピーしてください。

### (5) machikado データセットをダウンロード

https://github.com/nTAKAn/detectron2_Machikado/releases/download/v1.0/machikado60_vott-json-export.zip

ダウンロードし zipを展開後、vott-json-export ディレクトリの中身を detectron2_Machikado/vott-json-export ディレクトリにコピーしてください。

---
## 2. 学習


---
## 3. 推論

