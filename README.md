# pythonStudy

Pythonの学習用

※特に表記のない限りはPython3前提

## 言語仕様、文法関連

* 用語
    * Python: 言語仕様
    * CPython: Pythonで最も普及している実装（C言語で実装）
        * [wikipedia](https://ja.wikipedia.org/wiki/Python)見ると、他にもPython実装は複数ある模様
    * IPython: Pythonの非常に強力な対話型シェル

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
        * というかすべての型がオブジェクト
    * （おそらくCpythonについては）値の表現はCの実装依存らしい  
    https://docs.python.org/ja/3/library/array.html  
        > 値の実際の表現はマシンアーキテクチャ (厳密に言うとCの実装) によって決まります。値の実際のサイズは itemsize 属性から得られます。
    * 公式のドキュメント  
    https://docs.python.org/ja/3/c-api/memory.html
    * これ詳しい
        * https://rushter.com/blog/python-garbage-collector/
        * CPythonは参照カウンタ方式で世代別GCを実装しているようだ。整数がオブジェクト云々も書かれてる  
    * [wikipedia](https://ja.wikipedia.org/wiki/Python)に結構書かれている

* pythonのソースコードのリポジトリ
    * どこにあるのかよくわからないが、ここでダウンロードはできる  
        https://www.python.org/downloads/source/
    * とりあえず中身見るとC言語

## 機械学習関連

一般的な流れ

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
# stratify：学習データ・検証データそれぞれの正解ラベルの割合を揃える（データの偏りを回避する）
# random_state：int型を指定する。データ分割の再現性を確保する（再現性不要な場合は指定しなくてもよい）
# test_size：分割の割合を変更可能（デフォルトは75%-25%）
x_train, x_test, y_train, y_test = train_test_split(x_explanatory, y_target, stratify = y_target, random_state = 42)
```

### 学習データ：モデルのパラメータをチューニング

まずはデフォ設定でパッケージのbaseモデルで学習させてみて、おおよその精度を把握する。（実務ではよくやることらしい）

* Scikit-learn RandomForest
    * クラス分類: RandomForestClassifier()
    * 回帰分析: RandomForestRegressor()
* XGBoost
    * クラス分類: XGBClassifier()
    * 回帰分析: XGBRegressor()

### 検証データ：モデルの予測精度を測定

predict()で検証データを使って予測の精度を測定する。
このとき、予測で用いたデータは使用しない。

score()、accuracy_score()等で精度を取得可能。

* 分類
    * Accuracy
    * f値
        * 分類問題ではこれを見ることが多い
    * 混同行列
    * ROC曲線
        * 分類モデルの閾値をずらしたときの結果を見るもの
            * FPR(False Positive Rate)
            * TPR(True Positive Rate)
            * 良いモデルはFPRが低い時点でTPRが高いらしい
        * scikit-learn.roc_curve()
        * このサイトがわかりやすい
            * https://blog.kikagaku.co.jp/roc-auc ⇒ [キャプ画像](img/README/20220921-10030723.png)
    * AUC
        * ROC曲線の下側の面積
        * 理想的なモデルは1になり、完全にランダムに予測だと0.5、正解と真逆の予測だと0
        * scikit-learn.roc_auc_score()
    * classification_report()で一括で各種指標を見れる
        * 適合率(precision)、再現率(recall)、F1スコア、正解率(accuracy)、マクロ平均、マイクロ平均
        * macro avg、f値は実務では結構見る
        * weighted avgはあまり見ない
    * confusion_matrix
        * scikit-learnの結果は一般的な混同行列と出方が違うらしい（ポジティブ・ネガティブの位置が違う）  
        参考: Qiita 【入門者向け】機械学習の分類問題評価指標解説(正解率・適合率・再現率など) https://qiita.com/FukuharaYohei/items/be89a99c53586fa4e2e4  
        ![](img/README/20220920-20202513.png)
    
* 線形回帰モデル
    * MSE
    * RMSE
    * MAE
    * MAPE
    * R2

### 学習したモデルの可視化

* scikit-learn feature_importances_プロパティ
    * 決定木系のアルゴリズムを選択したとき、特徴量（説明変数）の重要度を算出

### その他

* Numpy便利関数
    * argsort(): 引数で指定した配列をソートした際の、元の配列のインデックス位置を返す
    * NumPy.ndarrayオブジェクトのインデクサの添字で、NumPy.ndarray型を渡せる
