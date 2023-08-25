# ml-template

## 環境構築
- python3.9系
    - pyenvの3.9.13で検証済み
- poetry
    - `pip install poetry`
    - `poetry install`

## フォルダ構成
- root
    - data/
        - train.csv
        - test.csv
    - yamls/
        - exp/
            - base.yaml
            - exp0.yaml
        - config.yaml
    - tasks.py
    - src/
    - README.md


## 実行方法
invokeで実行

### 学習と予測
trainから学習後、submit.csvを作成する

```sh
poetry run inv train -e exp0 -c comment
```

- `-e`: 使用するパラメータファイルを指定
- `-c`: フォルダにコメントを付与

### 保存したモデルの検証
学習済みモデルを正しく読み取れるか試す

```sh
poetry run inv predict-test model/20230825_001125_exp0_comment/
```
