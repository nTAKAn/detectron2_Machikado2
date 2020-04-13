# 続・detectron2 for まちカドまぞく

前編である「detectron2 for まちカドまぞく」がデフォルトのトレーナー、プレデクターを
使用しているのをカスタム版に切り替えてみます。

---
<img src=https://user-images.githubusercontent.com/33882378/79041969-d4473e00-7c2e-11ea-9072-b24d55bb4762.jpg>

AI学習モチベーション維持のために、まちカドまぞくが好きすぎるので detectron2 用のデータセットを　VoTT で作って試してみました。

## 1.使い方

### (1) 

### (2) detectron2_Machikado2 は detectron2 ディレクトリの下に clone してください。
detectron2 を git clone して出来た detectron2 ディレクトリの直下で clone してください。別に変えてもいいけど、その場合ディレクトリ関係のパスは修正してください。

### (3) weight ファイルをダウンロードします。

https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md からダウンロードし、coco_models/ へコピーしてください。

### (5) machikado データセットをダウンロード

https://github.com/nTAKAn/detectron2_Machikado/releases/download/v1.0/machikado60_vott-json-export.zip

ダウンロードし zipを展開後、vott-json-export ディレクトリの中身を detectron2_Machikado/vott-json-export ディレクトリにコピーしてください。

## 2. 学習


## 3. 推論

