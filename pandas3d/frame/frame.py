"""pandasのDataFrameを画像データ用に拡張したもの"""
import copy
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional, Tuple, Union, overload

import numpy as np
import pandas as pd
import tqdm
from matplotlib import pyplot as plt

from pandas3d.util import logging

logger = logging.get_logger(__name__)


def check_nan(func: Callable) -> Callable:
    """valuesがnanの場合に例外を投げるデコレーター用の関数

    Args:
        func (Callable): 実行する関数

    Returns:
        Callable: 実行する関数

    Raises:
        ValueError: nanが含まれている場合
    """

    @wraps(func)
    def _wrapper(gf: "GridFrame", *args: tuple, **kwargs: dict) -> Callable:
        if gf.values is None:
            raise ValueError("self.__values is None")
        return func(gf, *args, **kwargs)

    return _wrapper


class GridFrame:
    """pandasのDataFrameに相当

    Args:
        data (np.ndarray): データ
        columns (List): カラム名

    Examples:
        >>> array = np.arange(24).reshape(3, 4, 2)
        >>> gf = GridFrame(array, ["a", "b"])
        >>> type(gf)
        <class 'pandas3d.frame.frame.GridFrame'>
        >>> gf
        a
        [[ 0  2  4  6]
         [ 8 10 12 14]
         [16 18 20 22]]
        b
        [[ 1  3  5  7]
         [ 9 11 13 15]
         [17 19 21 23]]
        shape(3, 4, 2), dtype('int64')

        >>> GridFrame(columns=["a", "b"])
        Traceback (most recent call last):
        ...
        AssertionError: データのサイズとカラム数が不整合です

        >>> GridFrame(np.arange(3), ["a", "b", "c"])
        Traceback (most recent call last):
        ...
        AssertionError: データの次元は2or3である必要があります

        >>> GridFrame(np.arange(3).reshape(1, 3), ["a"])
        a
        [[0 1 2]]
        shape(1, 3, 1), dtype('int64')

        >>> GridFrame(np.arange(3).reshape(1, 3, 1), ["a", "b", "c"])
        Traceback (most recent call last):
        ...
        AssertionError: データのサイズとカラム数が不整合です

        >>> GridFrame(np.arange(3).reshape(1, 1, 3), ["a", "b", "b"])
        Traceback (most recent call last):
        ...
        AssertionError: カラム名はユニークである必要があります

        >>> GridFrame()
        []
        None
    """

    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        columns: Optional[Union[List[str], str]] = None,
    ) -> None:
        if columns is None:
            columns = []

        if data is None:
            assert len(columns) == 0, "データのサイズとカラム数が不整合です"
            self.__values = None
            self.__columns: list[str] = columns  # type: ignore
            return

        assert data.ndim == 3 or data.ndim == 2, "データの次元は2or3である必要があります"
        if data.ndim == 2:
            data = np.expand_dims(data, axis=-1)

        # strであればリストに変換
        if type(columns) == str:
            columns = [columns]

        # カラム数が一致しているか
        assert data.shape[-1] == len(columns), "データのサイズとカラム数が不整合です"

        # カラム名が重複していないかチェック
        assert len(set(columns)) == len(columns), "カラム名はユニークである必要があります"

        self.__values = data
        self.__columns = columns  # type: ignore

    @overload
    def __getitem__(self, index: str) -> np.ndarray:
        ...

    @overload
    def __getitem__(self, index: list) -> "GridFrame":
        ...

    @check_nan  # type: ignore
    def __getitem__(
        self, key: Union[str, list, tuple, np.ndarray]
    ) -> Union[np.ndarray, "GridFrame"]:
        """インデクシングの処理

        Args:
            key (str): インデックス

        Returns:
            np.ndarray: インデックスに対応する配列部を返す

        Hint:
            str: gf["カラム"]
            int: gf[1]
            slice: gf[1:2]
            list[Union[str, int]]: gf[["カラム", 1]]
            tuple[Union[str, int, slice, list[Union[str, int]]]]: gf[2, 2:3, [1, 2]]

        Examples:
            >>> array = np.arange(24).reshape(3, 4, 2)
            >>> gf = GridFrame(array, ["a", "b"])
            >>> type(gf["a"])
            <class 'numpy.ndarray'>
            >>> type(gf[["a"]])
            <class 'pandas3d.frame.frame.GridFrame'>
            >>> gf["a"]
            array([[ 0,  2,  4,  6],
                   [ 8, 10, 12, 14],
                   [16, 18, 20, 22]])
            >>> gf[["a", "b"]].shape == array.shape
            True

            >>> array = np.arange(72).reshape(3, 4, 6)
            >>> gf = GridFrame(array, ["a", "b", "c", "d", "e", "f"])
            >>> gf[gf.values[0][0] >= 3]
            d
            [[ 3  9 15 21]
             [27 33 39 45]
             [51 57 63 69]]
            e
            [[ 4 10 16 22]
             [28 34 40 46]
             [52 58 64 70]]
            f
            [[ 5 11 17 23]
             [29 35 41 47]
             [53 59 65 71]]
            shape(3, 4, 3), dtype('int64')
        """
        logger.debug(f"type(key): {type(key)}")
        try:
            logger.debug(f"type(key[0]): {type(key[0])}")
        except IndexError:
            pass

        if type(key) == str:
            if key in self.__columns:
                return self.__values[:, :, self.__columns.index(key)]  # type: ignore
            raise KeyError(f"{key} is not in columns")

        # ["x", "y"]の場合は該当する配列を返す
        elif type(key) == list:
            if all([type(i) == str for i in key]):
                key_int = [self.__columns.index(i) for i in key]
                return GridFrame(
                    data=self.__values[:, :, key_int],  # type: ignore
                    columns=list(key),
                )
            elif all([type(i) == bool for i in key]):
                return GridFrame(
                    data=self.__values[..., key],  # type: ignore
                    columns=[col for ind, col in zip(key, self.__columns) if ind],
                )
            raise ValueError(f"type unexpected{[type(i) for i in key]}")

        elif type(key) == np.ndarray:
            if type(key[0]) == np.bool_:
                return GridFrame(
                    data=self.__values[..., key],  # type: ignore
                    columns=[col for ind, col in zip(key, self.__columns) if ind],
                )
            raise ValueError(f"type unexpected{[type(i) for i in key]}")

        elif type(key) == int or type(key) == slice:  # type: ignore
            raise ValueError("ilocを利用してください")

        elif type(key) == tuple:
            if len(key) == 3:
                return GridFrame(
                    data=self.__values[key],  # type: ignore
                    columns=self.__columns[key[-1]],
                )
            return GridFrame(
                data=self.__values[key], columns=self.__columns  # type: ignore
            )

        else:
            raise ValueError(f"type unexpected{type(key)}")

    def __setitem__(
        self, key: Union[List[str], str], value: Union[float, np.ndarray]
    ) -> None:
        """値を代入した場合はカラム名を追加する

        Args:
            key (Union[List[str], str]): カラム名
            value (Union[float, np.ndarray]): 値

        Examples:
            >>> array = np.arange(24).reshape(3, 4, 2)
            >>> gf = GridFrame(array, ["a", "b"])
            >>> gf["c"] = np.arange(12).reshape(3, 4)
            >>> gf.columns
            ['a', 'b', 'c']
            >>> gf.shape
            (3, 4, 3)
        """
        # 返り値ではなく内部の値を変更するようにする
        # initとは別にバリデーションを行うのはよくないので、仮でappendしてそれに内部を書き換える
        if self.__values is None:
            if type(value) == int or type(value) == float:
                raise ValueError("value is not np.ndarray")
            gf_ = GridFrame(data=value, columns=key)  # type: ignore

        elif type(key) == str and key in self.__columns:
            print("__setitem__: 値を上書きします")
            gf_ = self
            self.__values[..., self.__columns.index(key)] = value

        elif (
            type(key) == list
            and type(key[0]) == str
            and any([k in self.__columns for k in key])
        ):
            raise ValueError("既存のキーが含まれています")

        else:
            if type(value) == int or type(value) == float:
                value = np.full(self.shape[:2], value)
            gf_ = self.append(GridFrame(data=value, columns=key))  # type: ignore
        self.__columns = gf_.__columns
        self.__values = gf_.__values

    def __iter__(self) -> Iterator[tuple[str, np.ndarray]]:
        """for文でのイテレータ

        Returns:
            Iterator[tuple[str, np.ndarray]]: カラム名と値のタプル

        Examples:
            >>> array = np.arange(24).reshape(3, 4, 2)
            >>> gf = GridFrame(array, ["a", "b"])
            >>> for c, v in gf:
            ...     print(c)
            ...     print(v)
            a
            [[ 0  2  4  6]
             [ 8 10 12 14]
             [16 18 20 22]]
            b
            [[ 1  3  5  7]
             [ 9 11 13 15]
             [17 19 21 23]]
        """
        for c in self.__columns:
            yield (c, self[c])

    def __contains__(self, name: str) -> bool:
        """in演算子でのチェック

        - カラム名がcolumnsに含まれているかを返す

        Args:
            name (str): カラム名

        Returns:
            bool: カラム名が存在するか

        Examples:
            >>> array = np.arange(24).reshape(3, 4, 2)
            >>> gf = GridFrame(array, ["a", "b"])
            >>> "a" in gf
            True
            >>> "c" in gf
            False
        """
        return name in self.__columns

    def __eq__(self, other: Any) -> bool:
        """==演算子でのチェック

        - columns, valuesが同じかを返す

        Args:
            other (GridFrame): 比較対象

        Returns:
            bool: columns, valuesが同じか

        Examples:
            >>> array = np.arange(24).reshape(3, 4, 2)
            >>> gf = GridFrame(array, ["a", "b"])
            >>> gf == gf
            True
            >>> gf == GridFrame(array + 1, ["a", "b"])
            False
            >>> gf == GridFrame(array, ["a", "c"])
            False
        """
        if not isinstance(other, GridFrame):
            return False
        return self.__columns == other.columns and (self.__values == other.values).all()

    @check_nan
    def __getattr__(self, name: str) -> Any:
        """既定以外の属性アクセス

        - np.ndarrayのメソッドを利用する

        Args:
            name (str): 属性名

        Returns:
            Any: 属性の値
        """
        if hasattr(self.__values, name):
            print(f"{name}はGridFrameに存在しないので、__valuesにアクセスします")
            return getattr(self.__values, name)
        raise AttributeError(f"{name}はGridFrameに存在しません")

    @property
    def iloc(self) -> "Iloc":
        """インデクシング

        Returns:
            GridFrame: インデクシングされたGridFrame

        Examples:
            >>> array = np.arange(24).reshape(3, 4, 2)
            >>> gf = GridFrame(array, ["a", "b"])
            >>> type(gf.iloc[:, 1:3, 0:1])
            <class 'pandas3d.frame.frame.GridFrame'>

            >>> array = np.arange(24).reshape(3, 4, 2)
            >>> gf = GridFrame(array, ["a", "b"])
            >>> type(gf.iloc[:, 1, 0:1])
            Traceback (most recent call last):
            ...
            ValueError: 3次元以上のデータが必要です
        """
        return Iloc(self)

    def __repr__(self) -> str:
        """文字列表現

        - print()での表示に利用される
        - カラムごとにカラム名と値を改行区切りで表示する

        Returns:
            str: 文字列表現
        """
        if self.__values is None:
            return f"{self.__columns}\n{self.__values}"
        str_ = "\n".join(
            [f"{col}\n{self.__values[..., i]}" for i, col in enumerate(self.__columns)]
        )
        str_ += f"\nshape{self.__values.shape}, dtype('{self.__values.dtype}')"
        return str_

    def __add__(self, gf: "GridFrame") -> "GridFrame":
        """足し算

        Args:
            gf (GridFrame): 足すGridFrame

        Returns:
            GridFrame: 足し算したGridFrame

        Examples:
            >>> array = np.arange(24).reshape(3, 4, 2)
            >>> gf = GridFrame(array, ["a", "b"])
            >>> gf2 = GridFrame(array, ["c", "d"])
            >>> gf_append = gf + gf2
            >>> gf_append.columns
            ['a', 'b', 'c', 'd']
            >>> gf_append.shape
            (3, 4, 4)

            >>> gf_append_empty = GridFrame() + gf
            >>> gf_append_empty.columns
            ['a', 'b']
            >>> gf_append_empty.shape
            (3, 4, 2)

            >>> gf + gf
            Traceback (most recent call last):
            ...
            AssertionError: キーが重複しています
        """
        if self.__values is None:
            return gf

        assert self.shape[:2] == gf.shape[:2], "サイズが違います"

        col_append = self.__columns + gf.__columns
        assert len(col_append) == len(set(col_append)), "キーが重複しています"

        return GridFrame(
            data=np.concatenate((self.__values, gf.__values), axis=-1),  # type: ignore
            columns=col_append,
        )

    @property
    def values(self) -> np.ndarray:
        """pandasのvaluesに相当

        Returns:
            np.ndarray: 配列部を返す

        Examples:
            >>> array = np.arange(24).reshape(3, 4, 2)
            >>> gf = GridFrame(array, ["a", "b"])
            >>> (gf.values == array).all()
            True
            >>> type(gf.values)
            <class 'numpy.ndarray'>
        """
        return self.__values  # type: ignore

    @property
    def columns(self) -> List[str]:
        """カラム名を返す

        Returns:
            List(str): カラム名

        Examples:
            >>> array = np.arange(24).reshape(3, 4, 2)
            >>> gf = GridFrame(array, ["a", "b"])
            >>> gf.columns
            ['a', 'b']
            >>> type(gf.columns)
            <class 'list'>
        """
        return self.__columns

    @property  # type: ignore
    @check_nan
    def shape(self) -> Tuple[int, ...]:
        """shapeを返す

        Returns:
            Tuple[int, ...]: shape

        Examples:
            >>> array = np.arange(24).reshape(3, 4, 2)
            >>> gf = GridFrame(array, ["a", "b"])
            >>> gf.shape
            (3, 4, 2)
        """
        return self.__values.shape  # type: ignore

    def append(self, gf: "GridFrame") -> "GridFrame":
        """追加する

        Args:
            gf (GridFrame): 追加するGridFrame

        Returns:
            GridFrame: 追加したGridFrame
        """
        return self.__add__(gf)

    @check_nan
    def copy(self) -> "GridFrame":
        """コピーする

        Returns:
            GridFrame: コピーしたGridFrame

        Warning:
            GridFrameのコピーはdeepcopyです。

        Examples:
            >>> array = np.arange(24).reshape(3, 4, 2)
            >>> gf = GridFrame(array, ["a", "b"])
            >>> gf_copy = gf.copy()
            >>> gf_copy is gf
            False
            >>> gf_copy == gf
            True
            >>> gf_copy.values is gf.values
            False
            >>> (gf_copy.values == gf.values).all()
            True
            >>> gf_copy.columns is gf.columns
            False
            >>> gf_copy.columns == gf.columns
            True
        """
        return GridFrame(
            data=copy.deepcopy(self.__values), columns=copy.deepcopy(self.__columns)
        )

    def add_prefix(self, prefix: str) -> "GridFrame":
        """カラムにprefixを追加する

        Args:
            prefix (str): 追加するprefix

        Returns:
            GridFrame: 追加したGridFrame

        Examples:
            >>> array = np.arange(24).reshape(3, 4, 2)
            >>> gf = GridFrame(array, ["a", "b"])
            >>> gf_add_prefix = gf.add_prefix("test_")
            >>> gf_add_prefix.columns
            ['test_a', 'test_b']
        """
        return GridFrame(
            data=self.__values, columns=[prefix + col for col in self.__columns]
        )

    def extract_columns_startswith(self, txt: str) -> "GridFrame":
        """txtで始まるカラムを抽出する

        Args:
            txt (str): 抽出する文字列

        Returns:
            GridFrame: 抽出したGridFrame

        Examples:
            >>> array = np.arange(24).reshape(3, 4, 2)
            >>> gf = GridFrame(array, ["a_test", "b_test"])
            >>> gf_extract = gf.extract_columns_startswith("a")
            >>> gf_extract.columns
            ['a_test']
            >>> gf_extract.shape
            (3, 4, 1)
        """
        return self[[col for col in self.__columns if col.startswith(txt)]]

    def extract_columns_endswith(self, txt: str) -> "GridFrame":
        """txtで終わるカラムを抽出する

        Args:
            txt (str): 抽出する文字列

        Returns:
            GridFrame: 抽出したGridFrame

        Examples:
            >>> array = np.arange(24).reshape(3, 4, 2)
            >>> gf = GridFrame(array, ["test_a", "test_b"])
            >>> gf_extract = gf.extract_columns_endswith("a")
            >>> gf_extract.columns
            ['test_a']
            >>> gf_extract.shape
            (3, 4, 1)
        """
        return self[[col for col in self.__columns if col.endswith(txt)]]

    @check_nan
    def to_pandas(self) -> pd.DataFrame:
        """pandasのDataFrameに変換する

        - gf:shape(a, b, c) -> pd.shape(a * b, c)

        Returns:
            pd.DataFrame: pandasのDataFrame

        Examples:
            >>> array = np.arange(24).reshape(3, 4, 2)
            >>> gf = GridFrame(array, ["a", "b"])
            >>> df = gf.to_pandas()
            >>> type(df)
            <class 'pandas.core.frame.DataFrame'>
            >>> df.shape
            (12, 2)
            >>> list(df.columns)
            ['a', 'b']
        """
        return pd.DataFrame(
            data=self.__values.reshape(-1, self.shape[-1]),  # type: ignore
            columns=self.__columns,
        )

    @check_nan
    def check_isfinite(self) -> None:
        """nan, inf, -infが含まれている場合に例外を投げる

        Examples:
            >>> GridFrame(np.array([[np.inf]]), ["a"]).check_isfinite()
            Traceback (most recent call last):
            ...
            ValueError: nanやinfが含まれています
        """
        if (~np.isfinite(self.__values)).sum() != 0:  # type: ignore
            raise ValueError("nanやinfが含まれています")

    @check_nan
    def draw_distribution(
        self, save_dir: Path, is_axis: bool = True, extension: str = "png"
    ) -> None:
        """分布を画像として保存する

        Args:
            save_dir (Path): 描画した画像を保存するディレクトリ
            is_axis (bool): 分布を描画するかどうか
            extension (str): 拡張子
        """
        save_dir.mkdir(exist_ok=True, parents=True)
        for col in tqdm.tqdm(self.__columns, desc="分布描画の進行状況"):
            plt.figure(figsize=(12, 12))
            plt.title(col)
            cs = plt.imshow(self[col], cmap="gray")
            if not is_axis:
                plt.axis("off")
            plt.colorbar(cs)
            plt.savefig(
                save_dir.joinpath(f"{col}.{extension}"),
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close()


class Iloc:
    """インデクシングを行うクラス

    - gf.ilocで呼ばれる

    Args:
        gf (GridFrame): GridFrame
    """

    def __init__(self, gf: "GridFrame") -> None:
        self.__gf = gf

    def __getitem__(self, key: Union[int, list, tuple, slice]) -> GridFrame:
        """インデクシングを行う

        Args:
            key (Union[int, list, tuple, slice]): インデクシングするkey
        """
        if self.__gf.values is None:
            raise ValueError("データがありません")

        data = self.__gf.values.__getitem__(key)
        if data.ndim < 3:
            raise ValueError("3次元以上のデータが必要です")

        if (type(key) == tuple or type(key) == list) and key.__len__() == 3:
            return GridFrame(
                data=data,
                columns=self.__gf.columns.__getitem__(key[-1]),
            )

        return GridFrame(data=data, columns=self.__gf.columns)

    def __setitem__(
        self, key: Union[int, list, tuple, slice], value: np.ndarray
    ) -> None:
        """インデックスに値を設定する

        Args:
            key (Union[int, list, tuple, slice]): インデクシングするkey
            value (np.ndarray): 設定する値
        """
        self.__gf.values[key] = value


def zeros(
    shape: tuple[int, ...], columns: Optional[Union[List, str]] = None
) -> GridFrame:
    """0を入れるGridFrameを作成する

    Args:
        shape (tuple[int, ...]): shape
        columns (Optional[Union[List, str]]): カラム名

    Returns:
        GridFrame: 0を入れたGridFrame

    Examples:
        >>> gf = zeros((3, 4, 2), ["a", "b"])
        >>> gf.values.sum()
        0.0
    """
    if len(shape) == 2:
        shape = shape + (0,)  # tupleにする
    return GridFrame(np.zeros(shape), columns)


def empty(
    shape: tuple[int, ...], columns: Optional[Union[List, str]] = None
) -> GridFrame:
    """空のGridFrameを作成する

    Args:
        shape (tuple[int, ...]): shape
        columns (Optional[Union[List, str]]): カラム名

    Returns:
        GridFrame: 空のGridFrame

    Examples:
        >>> gf = empty((3, 4))
        >>> gf.columns
        []
        >>> gf.shape
        (3, 4, 0)
    """
    if len(shape) == 2:
        shape = shape + (0,)
    return GridFrame(np.empty(shape), columns)


def from_pandas(df: pd.DataFrame, shape: tuple[int, int]) -> GridFrame:
    """pandasのDataFrameからGridFrameを作成する

    - カラムはそのまま
    - shape=(a, b)のとき(a * b, c) -> (a , b, c)に変換する

    Args:
        df (pd.DataFrame): pandasのDataFrame
        shape (tuple[int, int]): shape

    Returns:
        GridFrame: GridFrame

    Examples:
        >>> df = pd.DataFrame(np.arange(24).reshape(12, 2), columns=["a", "b"])
        >>> gf = from_pandas(df, shape=(3, 4))
        >>> gf.shape
        (3, 4, 2)
        >>> gf.columns
        ['a', 'b']
    """
    data = df.values
    assert np.prod(shape) == data.shape[0], "shapeが不適切です"
    return GridFrame(
        data=data.reshape(list(shape) + [data.shape[-1]]), columns=list(df.columns)
    )
