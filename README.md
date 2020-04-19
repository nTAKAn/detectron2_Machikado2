# 続・detectron2 for まちカドまぞく

前編である「detectron2 for まちカドまぞく」にカスタム機能を追加して精度を高めます。

> 基本的な内容は前回「detectron2 for まちカドまぞく」になります。
>
> https://github.com/nTAKAn/detectron2_Machikado

---

今回は本当の推論画像です。上手くいったのでドヤ顔にしてみました。

<img src=https://user-images.githubusercontent.com/33882378/79108398-34fe8400-7db1-11ea-9b26-08e09e13243f.jpg>

---

| 日付 | 内容 |
| --- | --- |
| 2020/04/19 | 精度がちょっと向上しました。 （mAP 70 -> 80%）|
| | * 画像入力サイズの変更。（設定で変更できます）|
| | * それに伴い、バッチ数を増加。|
| 2020/04/18 | mAPの計算ミスを修正しました。|

---
## 1.使い方

* detectron2_Machikado2 は detectron2 ディレクトリの下に clone してください。

* 重みファイル、データセットは前回と同じなので、前回を参照してください。

https://github.com/nTAKAn/detectron2_Machikado

---
## 2. カスタム編の内容

### [データセットマッパー](https://github.com/nTAKAn/detectron2_Machikado2/blob/master/custom1_DatasetMapper.ipynb)

オリジナルの水増し方法です。

### [トランスフォーム](https://github.com/nTAKAn/detectron2_Machikado2/blob/master/custom2_Transform.ipynb)

オリジナルで、画像に変形を加える方法です。

### [トレーナー](https://github.com/nTAKAn/detectron2_Machikado2/blob/master/custom3_training.ipynb)

カスタムした内容で訓練する方法です。

### [プレディクタ](https://github.com/nTAKAn/detectron2_Machikado2/blob/master/custom4_evaluate.ipynb)

カスタムしてバッチで推論して、ついでに mAP を見てみます。

---
## 3. その他

* データセットじゃなくて、個別に画像を推論したい場合は前回を参照ください。