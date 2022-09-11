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
        * ただしraw文字列は末尾が「\」だとシンタックスエラーになる（C#の `@"hoge"` とは仕様が異なるので注意）  
        ![](img/README/20220911-09152040.png)


## 機械学習関連

### 学習データ、検証データの作成

```python
# 変数dataはDataFrame型とする

# 説明変数、目的変数に分割する例
x_explanatory = data.drop("target_col")
y_target = data["target_col"]

# 説明変数、目的変数に分割する例：Listに一度変換するパターン
cols = data.columns.tolist()
cols.remove("target_col")
x_explanatory = data[cols]
y_target = data['target_col']

# 学習データ、検証データの作成
x_train, x_test, y_train, y_test = train_test_split(x_explanatory, y_target)
```