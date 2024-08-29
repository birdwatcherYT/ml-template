# ml-template
機械学習コンペ用テンプレート

## 環境構築
- python3.9系
    - pyenvの3.9.13で検証済み
- poetry
    - `pip install poetry`
    - `poetry config virtualenvs.in-project true`
    - `poetry install`
- pre-commit
    - `poetry run pre-commit install`
    - 「.git/hooks/pre-commit: 行 5: $'\r': コマンドが見つかりません」
        - .git/hooks/pre-commitの改行コードをLFに変える
    - 「.git/hooks/pre-commit: 15 行: exec: C:\Users\ユーザー名\Downloads\ml-template\.venv\Scripts\python.exe: 見つかりません」
        - \を/に変える

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
