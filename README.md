# pythonStudy

Pythonの学習用

※特に表記のない限りはPython3前提

## 言語仕様、文法関連

* 内包表記
    * map(), filter()等で書くことも可能だが、速度面では内包表記が優れているらしい

* 文字列
    * フォルダパスを書く際は、raw文字列で書くのがとりあえずは楽か
        * 記載例  
        `foo = r"c:\temp"`
        * ただしraw文字列は末尾が「\」だとシンタックスエラーになる  
        （C#の `@"hoge"` とは仕様が異なるので注意。この辺PythonはWindowsではいまいち感ある）  
        ![](img/README/20220911-09152040.png)

* 配列とリストの違い
    * 配列は固定長で定義時の型のみ
    * リストは動的配列で型はなんでも良い
        * C#のSystem.Collections.Generic.List<System.Object>型が近いかも
    * 両者はメモリ確保の仕方が違う

* メモリ、ヒープ、スタック周り関連
    * 整数はオブジェクトらしい・・・
    * 値はCの実装依存らしい  
    https://docs.python.org/ja/3/library/array.html  
        > 値の実際の表現はマシンアーキテクチャ (厳密に言うとCの実装) によって決まります。値の実際のサイズは itemsize 属性から得られます。

* 用語
    * Python: 言語仕様
    * CPython: Pythonの一実装（C言語で実装）
    * IPython: Pythonの非常に強力な対話型シェル

* pythonのソースコードのリポジトリ
    * どこにあるのかよくわからないが、ここでダウンロードはできる  
        https://www.python.org/downloads/source/
    * とりあえず中身見るとC言語

## 機械学習関連

### 学習データ、検証データの作成

```python
# 変数dataは pandas.core.frame.DataFrame 型とする

# 説明変数、目的変数に分割する例
x_explanatory = data.drop("target_col")
y_target = data["target_col"]

# 説明変数、目的変数に分割する例：ColumnsをListで一旦摂ってきて、Lists指定で分割するパターン
cols = data.columns.tolist()
cols.remove("target_col")
x_explanatory = data[cols]
y_target = data['target_col']

# 学習データ、検証データの作成
# stratify指定：学習データ・検証データそれぞれの正解ラベルの割合を揃える（データの偏りを回避する）
# random_state：int型を指定する。データ分割の再現性を確保する（再現性不要な場合は指定しなくてもよい）
x_train, x_test, y_train, y_test = train_test_split(x_explanatory, y_target, stratify = y_target, random_state = 42)
```

### 分析

* Scikit-learn RandomForest
    * クラス分類: RandomForestClassifier()
    * 回帰分析: RandomForestRegressor()
* XGBoost
    * クラス分類: XGBClassifier()
    * 回帰分析: XGBRegressor()

### 評価指標

* 分類
    * Accuracy
    * f値
        * 分類問題ではこれを見ることが多い
    * classification_repor()で一括で各種指標を見れる
        * 適合率(precision)、再現率(recall)、F1スコア、正解率(accuracy)、マクロ平均、マイクロ平均

* 線形回帰モデル
    * MSE
    * RMSE
    * MAE
    * MAPE
    * R2

