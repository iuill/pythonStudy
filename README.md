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
            * https://blog.kikagaku.co.jp/roc-auc
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
* XGB feature_importances_プロパティ
    * scikit-learn同様？たぶん
* XGB plot_importance()
    * グラフ表示までまとめてやってくれる
    * importance_type引数で重要度算出アルゴリズムを変更可能 weight or gain or cover
        * gainオプションはfeature_importances_プロパティ同様になる
    * 公式リファレンス: https://xgboost.readthedocs.io/en/stable/python/python_api.html
        * > ”weight” is the number of times a feature appears in a tree  
        * > ”gain” is the average gain of splits which use the feature  
        * >”cover” is the average coverage of splits which use the feature where coverage is defined as the number of samples affected by the split

### データの事前整形

* データクリーニング（全角数字を半角数字に統一させたりとか）
    * 不要な列の削除（IDなど）
    * 定義外な値の見直し
* データ統合（複数のデータを扱いやすく統合したりする）
* データ変換（文字列を数値に変換したりする）
* 特徴量エンジニアリング（説明変数を加工して新しい説明変数を作ったりなど）
    * 一般的に、説明変数を増やすと精度向上しやすくなる＆過学習しやすくなる
    * データの偏りは正規分布の形が理想
        * https://uribo.github.io/practical-ds/02/numeric.html  
            > 具体的には線形回帰モデルでは、出力から得られる値の誤差が正規分布に従うことを仮定します。そのため正規分布とは異なる形状の分布をもつデータ、例えば離散値ではその仮定が成立しないことが可能性があります。この問題を解決するため、元のデータを正規分布に近似させるという特徴量エンジニアリングが有効になります。  
            > 良い特徴量というのはデータの特徴を強く反映します。連続的な数値の二値化あるいは離散化により、モデルの精度を改善できる見込みがあります。また数値以外のテキストや画像データを数値化した際、さらなる特徴量エンジニアリングが必要になることがあります。つまり数値データの処理は特徴量エンジニアリングの中で最も基本的な技と言えます。  
            > ～略～  
            > スケール変換では変数のばらつきを元にする変換のために変換後の値でも分布は変わりません。しかし対数変換ではデータの分布が変化します。
    * 対数化の手法
        * https://omathin.com/100knock-61-62/
            * 0は扱えないので一般的には1を追加してから対数化する
            * 常用対数化: numpy.log10()
                * ex: `df_sales_amount['amount_log10'] = np.log10(df_sales_amount['amount']+1)`
            * 自然対数化: numpy.log()
                * ex: `df_sales_amount['amount_loge'] = np.log(df_sales_amount['amount']+1)`
    * 相関が高すぎる説明変数を減らしてみる
    * 不均衡データの調整
        * アップサンプリング
        * ダウンサンプリング
        * 損失関数の調整
* ...など

### EDA

https://qiita.com/ryo111/items/bf24c8cf508ad90cfe2e

#### pandas ProfileReport()

* 解説
    * https://datatechlog.com/how-to-use-pandas-profiling/
* to_file()でhtmlファイルに出力可能。  
    * ファイルサイズが大きい場合、Chrome系ブラウザだと out of meomry を起こすことがあるので、その場合はFirefoxで試してみる。
* オプション
    * 相関を出さない: 
        ```
        df.profile_report(
            title="Report without correlations",
            correlations=None,
        )
        ```
    * オプション一覧
        * https://pandas-profiling.ydata.ai/docs/master/pages/advanced_usage/available_settings.html
    * correlationsの有無、explorativeの有無、いずれも対して実行時間かわらない気がするのと、生成されるHTMLファイルのサイズは対して違いがない
        * サイズや速度が気になるときは `minimal=True` を設定するのが効果的な気がする

#### seaborn

* countplot
* heatmap
* pairplot

#### one hot encoding

* pandas get_dummies()
    * drop_first = Trueにすると、生成される列が一つ減る
* scikit-learn OneHotEncoder()

### 次元削減、次元圧縮

* スケール変換
    * 機械学習時、データの分布が異なるデータを扱うときはほぼ必須
        * Scikit-learn StandardScaler().fit_transform() ※他にもいくつかのスケール変換方法あり
        * https://helve-blog.com/posts/python/scikit-learn-feature-scaling/
* scikit-learn PCA() fit_transform()
    * PCAはあらかじめスケール変換することが多い
* scikit-learn TSNE() fit_transform()
* 可視化は、二次元 or 三次元

### パラメータチューニング

パラメータチューニングは序盤から行うのはあまり適切でない。

パラメータチューニングやる前に、特徴量エンジニアリングなどを行うのが精度改善につながりやすい。

* ランダムフォレスト
    * 以下はほぼ必須っぽい
        * max_depath
            * 深すぎても過学習してしまう（7くらいがよいらしい）
            * scikit-learnのデフォルト値はNONE = 無限
        * n_estimators
    * 各パラメータ詳細: https://data-science.gr.jp/implementation/iml_sklearn_random_forest.html

* XGB
    * 以下は重要
        * max_depath
    * early_stopping
        * 早めに打ち切ることで過学習を防ぐといったテクニックがある

* scikit-learn GridSearchCV()
    * param_grid: 探索対象ハイパーパラメータの辞書・リスト
    * scoring: 評価指標（デフォルト値はaccuracy。必要に応じてf1など適切なものを選択）
    * CV: 交差検証の回数
    * best_estimator_プロパティで、成績の良かったパラメータを取得できる

自動化

* ハイパーパラメータの探索＆特徴量エンジニアリング  https://qiita.com/Hironsan/items/30fe09c85da8a28ebd63

### その他

* Numpy便利関数
    * argsort(): 引数で指定した配列をソートした際の、元の配列のインデックス位置を返す
    * NumPy.ndarrayオブジェクトのインデクサの添字で、NumPy.ndarray型を渡せる
    * NumPy.cut(), qcut()でビニング処理が可能  
    前処理で使用することはわりとあるらしい

* XGBoost
    * early_stopping_round
        * 1.6.0以降は、XGBClassfierなどのコンストラクタで指定するかset_paramsで指定する

* グラフ表示
    * Matplotlibの軸の指数表記の設定
        * https://grapebanana.com/matplotlib-axis-11306/
        * http://www.yamamo10.jp/yamamoto/comp/Python/library/Matplotlib/basic/setting/index.php#INIT-SET

## パッケージ

* インストールされるフォルダ
    * `.\lib\site-packages` っぽい？
* 一括アップデート
    * `conda update --all`

* pipコマンドを使う場合
    * 常にこれを実行しておくのがよいらしい？ `python -m pip install --upgrade pip setuptools`
* condaのアップデート
    * `conda update -n base conda`
* pandas-profiling
    * なぜか `conda install -c conda-forge pandas-profiling` だと1.4.1系
    * `pip install pandas-profiling`
* XGBoost
    * https://xgboost.readthedocs.io/en/stable/install.html#python
    * 2022/09時点 `conda install -c conda-forge py-xgboost` だと1.5.0になる
        * `conda install -c conda-forge py-xgboost==1.6.2` だと `conda-forge/win-64::py-xgboost-1.6.2-cpu_py39ha538f94_0` になる  
        cpu onlyバージョン・・・？
            * ここ見ると linux-64 しか無い・・・  
            https://anaconda.org/conda-forge/py-xgboost-gpu  
            ![](img/README/20220925-10120197.png)
    * pipだとGPUサポートバージョンが含まれている模様  
    ![](img/README/20220925-10143196.png)

## 文字コード関連

### pyファイルの文字コード設定

コード内で日本語文字列が出てくるのであれば、ファイルの文字コードをutf-8にするのがたぶん楽。  
※BOMありにして良いのかはよくわからない

コードの一行目で文字コードを明記する方法もある。ぐぐったら以下がよくまとまっていた。  
https://qiita.com/KEINOS/items/6efc1147b917d7811b5b

## ロギングフレームワーク

Python標準ライブラリ内でロギング用ライブラリが整備されている。

https://docs.python.org/ja/3/howto/logging.html

https://docs.python.org/ja/3/howto/logging.html#configuring-logging

* フォーマッタ、ハンドラ
    * https://www.python.ambitious-engineer.com/archives/693
    * https://www.tohoho-web.com/python/logging.html

## リフレクション系

* 変数名取得
    * locals, globals
* 備忘
    * https://qiita.com/icoxfog417/items/bf04966d4e9706eb9e04
    * https://docs.python.org/ja/3/c-api/reflection.html

## CUDA

* XGBoostでのGPU使用
    * CUDA Toolkitを入れる必要がある
    * condaでインストールされるxgboostは1.5.0だが、GPU非サポート
        ```
        xgboost.core.XGBoostError: [09:12:50] c:\windows\temp\abs_557yfx631l\croots\recipe\xgboost-split_1659548953302\work\src\common\common.h:157: XGBoost version not compiled with GPU support.
        ```
    * GPUのベンチマーク
        * Windowsのanaconda 3だと、`benchmark_tree.py`は入ってないっぽい
        * XGBoostのGitHubリポジトリには入っているのでここから落とせる
        * "Anaconda Prompt (anaconda3)" から以下実行すればよい
            ```
            python tests/benchmark/benchmark_tree.py --tree_method=gpu_hist
            python tests/benchmark/benchmark_tree.py --tree_method=hist
            ```
    * XGBClassifier()がGPU使う設定だと遅い
        * VisualStudio2022でのデバッグ実行でも、Anaconda Promptからの実行でも、実行時間変わらない
            * `tree_method="gpu_hist"`だと12分くらいかかる処理が、`tree_method="hist"`だと2分くらいになる  
            ・・・？？🤔
        * benchmark_tree.py だと、GPUはCPUの半分以下の処理時間で高速
        * 原因不明
            * 相性？ https://teratail.com/questions/296378
        * 自分でビルドする場合の備忘
            * https://sekailab.com/wp/2018/06/29/installation-xgboost-to-winwdows10-gpu-support/
            * https://xgboost.readthedocs.io/en/latest/build.html#building-on-windows
* CUDA
    * https://developer.nvidia.com/cuda-gpus
        * 対応しているGPUは以下箇所で確認可能  
        ![](img/README/20220924-20194027.png)
        * オフライン環境用だとバイナリがでかい
            * CUDA 11で2.5GBある
        * network環境用のインストーラを使うのが吉
            * 以下は「Disable Usage Collection」で問題ない気はする  
            ![](img/README/20220924-20411184.png)
    * 古いバージョンはこちら  https://developer.nvidia.com/cuda-toolkit-archive

## AnacondaやめてMinicondaにしてみる

機械学習前提＆Jupityer Notebook使える状態にしておきたい＆pip使いたいことがある、ってときcondaとpipの混在が微妙っぽいので、以下を参考にAnacondaをアンインストールしてMinicondaを入れ直す。

* 参考
    * https://qiita.com/kawada2017/items/626a80ed5bbfdc2576a5
* Miniconda ダウンロードサイト
    * https://docs.conda.io/en/latest/miniconda.html#windows-installers
    * Python 3.9用を選択
    * インストール後、以下で確認していく
        * `conda -V`
        * `python --version`
        * `conda info`
    * 初期設定直後の状態で以下実行し、結果をとりあえずGitHub（プライベートリポジトリ）に放り込んでおく
        * `pip freeze > requirements.txt`
        * `conda list -e > conda_requirements.txt`
        * requirements.txtは他端末で同じ環境を用意したいときに使える（あまりうまくいかないこともあるみたいだが）

次に仮想環境を作る。  
Pythonの仮想環境構築用の機能は複数あるみたいだが、condaで構築する。

* 各機能の比較
    * https://zenn.dev/mook_jp/articles/1d915a0aef83a7  
![](img/README/20220925-16003149.png)
    * https://qiita.com/KRiver1/items/c1788e616b77a9bad4dd#pyenv-virtualenv%E3%82%92%E3%82%84%E3%82%81pipenv%E3%82%92%E4%BD%BF%E3%81%86

構築手順は以下を参考にする。

* https://zenn.dev/unemployed/articles/cc111706c3167c

* 実施手順
    1. `Anaconda Prompt (miniconda3)` を立ち上げる
    1. `conda create -n DataScienceCompe2022_eLearning python` で仮想環境を作成
        * condaをupdateしたほうがよいみたいなことを言われるが無視する  
        * C:\Users\ユーザ名\.conda\envs\DataScienceCompe2022_eLearning のようにフォルダが作られる
        * インストールするパッケージがずらずら表示されるが、base側の話なのかよくわからない
            * 実行前後でbase側の `conda list` を比較したが特に差分なかったので、仮想環境側の話のようだ。
    1. 環境作成・パッケージインストール後、`conda info -e` で作った仮想環境一覧が見れる
    1. `conda activate 環境名` で仮想環境に入る（source activate 環境名は古い書き方っぽい）
    1. `pip freeze` でインストール済みのパッケージ一覧を確認
        * インストールされている内容がめちゃシンプル
    1. `conda deactivate` で仮想環境を抜ける
    1. `conda remove -n 環境名 --all` で環境を削除できる

* 追加で入れるパッケージ
    1. `Anaconda Prompt (miniconda3)` を立ち上げる
    1. `pip install jupyter notebook`
        * ネット見ると、`conda install notebook ipykernel` の手順も必要なように書かれているが、特にやらなくても jupyter notebook からpythonは動いている模様（過去にAnaconda入れていたからかもしれない）
        * jupyterの起動は `jupyter notebook` のようにするとブラウザが立ち上がるところまで自動で動く
    1. `pip install numpy pandas pandas-profiling seaborn sklearn xgboost` を実行

* Visual Studio Community 2022での設定
    * ソリューションエクスプローラで `Python環境` -> `環境を追加`  
    ![](img/README/20220925-16254675.png)
    * `環境を追加` -> `既存環境` で（baseでない）仮想環境のフォルダを設定すると勝手にVSに認識される  
    ![](img/README/20220925-16265625.png)

## Visual Studio Community 2022ノウハウ

* グラフ化
    * 適当なところでブレークポイントで止めた状態でイミディエイトウィンドウで `sns.histplot(x="BILL_AMT1", data=data, element='step')` のように実行するとその場でグラフ見れる  
    ![](img/README/20220926-22404058.png)
    * グラフのウィンドウを複数枚出したいときは以下のように一度plt.figure()を呼び出せばとりあえずできる（もっとカッコいいやり方がありそうな気はする）
        ```
        plt.figure()
        sns.histplot(x="BILL_AMT1", data=data, element='step')
        ```
    
