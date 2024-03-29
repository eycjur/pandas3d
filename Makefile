# 環境構築の際や後処理を記述

target := .

## メタ的なコマンド
# デフォルトコマンド(test sync_notebook lint)
all: test lint

# ヘルプを表示
help:
	@cat $(MAKEFILE_LIST) | python -u -c 'import sys, re; from itertools import tee,chain; rx = re.compile(r"^[a-zA-Z0-9\-_]+:"); xs, ys = tee(sys.stdin); [print(f"""\t{line.split(":")[0]:20s}\t{prev.lstrip("# ").rstrip()}""") if rx.search(line) and prev.startswith("#") else print(f"""\n{prev.lstrip("## ").rstrip()}""") if prev.startswith("##") else "" for prev, line in zip(chain([""], xs), ys)]'

# まとめたもの(lint test sync_notebook sphinx-reflesh)
full: lint test sync-notebook sphinx-reflesh


## 環境構築関連
# 1から環境構築
install:
	poetry install

## python関連のコマンド
# jupyterの起動
lab:
	poetry run jupyter lab
jupyter:
	@make lab

# テストコードの実行
test:
	poetry run pytest

# リンター
lint:
	@make --no-print-directory black
	@make --no-print-directory isort
	@make --no-print-directory flake8
	@make --no-print-directory mypy
mypy:
	poetry run python -m mypy $(target)
black:
	poetry run black $(target)
flake8:
	poetry run flake8 $(target)
isort:
	poetry run isort $(target)

# sphinx（ドキュメント自動作成ツール）関係
sphinx:
	poetry run sphinx-apidoc -f -o ./docs/source ./pandas3d
	poetry run sphinx-build -b html ./docs ./docs/_build
sphinx-reflesh:
	rm -rf docs/_build/* docs/source/*.rst
	@make --no-print-directory sphinx
open:
	open -a "Google Chrome" docs/_build/index.html

# プロファイリング
profile:
	poetry run python -m cProfile -o logs/profile.stats pandas3d/cli.py command
	poetry run snakeviz ./logs/profile.stats
