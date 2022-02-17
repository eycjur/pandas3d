"""pandasのDataFrameを画像データ用に拡張したもの"""
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional, Tuple, Union, overload

import numpy as np
import pandas as pd
import tqdm
from matplotlib import pyplot as plt


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
    def _wrapper(gf: "GridFrame", *args: tuple, **kwargs: dict[Any, Any]) -> Callable:
        if gf.values is None:
            raise ValueError("self.__values is None")
        return func(gf, *args, **kwargs)

    return _wrapper


class GridFrame:
    """pandasのGridFrameに相当

    Args:
        data (np.ndarray): データ
        columns (List): カラム名
    """

    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        columns: Optional[Union[List[str], str]] = None,
    ) -> None:
        """
        Examples:
            >>> array = np.arange(24).reshape(3, 4, 2)
            >>> gf = GridFrame(array, ["a", "b"])
            >>> type(gf)
            <class 'frame.GridFrame'>

            >>> GridFrame(np.array([1, 2, 3]))
            Traceback (most recent call last):
            ...
            AssertionError: (3,)!=0
        """
        if columns is None:
            columns = []

        if data is None:
            assert len(columns) == 0, "データのサイズとカラム数が不整合です"
            self.__values = None
            self.__columns: list[str] = columns  # type: ignore
            return

        assert data.ndim != 3 or data.ndim != 2, f"{data.ndim}は2or3である必要があります"
        if data.ndim == 2:
            data = np.expand_dims(data, axis=-1)

        # strであればリストに変換
        if type(columns) == str:
            columns = [columns]

        # カラム数が一致しているか
        assert data.shape[-1] == len(columns), f"{data.shape}!={len(columns)}"

        # カラム名が重複していないかチェック
        assert len(set(columns)) == len(columns), "カラム名はユニークである必要があります"

        self.__values = data.astype("float32")
        self.__columns = columns  # type: ignore

    @overload
    def __getitem__(self, index: str) -> np.ndarray:
        ...

    @overload
    def __getitem__(self, index: list) -> "GridFrame":
        ...

    @check_nan  # type: ignore
    def __getitem__(
        self, index: Union[str, list, tuple]
    ) -> Union[np.ndarray, "GridFrame"]:
        """インデクシングの処理

        Args:
            index (str): インデックス

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
            <class 'frame.GridFrame'>
            >>> gf["a"]
            array([[ 0.,  2.,  4.,  6.],
                   [ 8., 10., 12., 14.],
                   [16., 18., 20., 22.]], dtype=float32)
            >>> gf[["a", "b"]].shape == array.shape
            True
        """
        if type(index) == str:
            if index in self.__columns:
                return self.__values[:, :, self.__columns.index(index)]  # type: ignore
            raise KeyError(f"{index} is not in columns")

        # ["x", "y"]の場合は該当する配列を返す
        elif type(index) == list:
            if all([type(i) == str for i in index]):
                index_int = [self.__columns.index(i) for i in index]
                return GridFrame(
                    data=self.__values[:, :, index_int],  # type: ignore
                    columns=list(index),
                )
            raise ValueError(f"type unexpected{[type(i) for i in index]}")

        elif type(index) == int or type(index) == slice:  # type: ignore
            raise ValueError("ilocを利用してください")

        elif type(index) == tuple:
            if len(index) == 3:
                return GridFrame(
                    data=self.__values[index],  # type: ignore
                    columns=self.__columns[index[-1]],
                )
            return GridFrame(
                data=self.__values[index], columns=self.__columns  # type: ignore
            )

        else:
            raise ValueError(f"type unexpected{type(index)}")

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
            [[ 0.  2.  4.  6.]
             [ 8. 10. 12. 14.]
             [16. 18. 20. 22.]]
            b
            [[ 1.  3.  5.  7.]
             [ 9. 11. 13. 15.]
             [17. 19. 21. 23.]]
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
            >>> type(gf.iloc[0, 0, 0:1])
            <class 'frame.GridFrame'>
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
        """
        return self.append(gf)

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

        Examples:
            >>> array = np.arange(24).reshape(3, 4, 2)
            >>> gf = GridFrame(array, ["a", "b"])
            >>> gf2 = GridFrame(array, ["c", "d"])
            >>> gf_append = gf.append(gf2)
            >>> gf_append.columns
            ['a', 'b', 'c', 'd']
            >>> gf_append.shape
            (3, 4, 4)

            >>> gf_append_empty = GridFrame().append(gf)
            >>> gf_append_empty.columns
            ['a', 'b']
            >>> gf_append_empty.shape
            (3, 4, 2)

            >>> gf.append(gf)
            Traceback (most recent call last):
            ...
            AssertionError: キーが重複しています
        """
        if self.__values is None:
            return gf

        assert (
            self.shape[:2] == gf.shape[:2]
        ), f"サイズが違います{self.shape[:2]},{gf.shape[:2]}"

        col_append = self.__columns + gf.__columns
        assert len(col_append) == len(set(col_append)), "キーが重複しています"

        return GridFrame(
            data=np.concatenate((self.__values, gf.__values), axis=-1),  # type: ignore
            columns=col_append,
        )

    @check_nan
    def copy(self) -> "GridFrame":
        """コピーする

        Returns:
            GridFrame: コピーしたGridFrame

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
            data=self.__values.copy(), columns=self.__columns.copy()  # type: ignore
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
            ValueError: nanやinfが含まれています1,0,0
        """
        if (~np.isfinite(self.__values)).sum() != 0:  # type: ignore
            raise ValueError(
                f"nanやinfが含まれています{np.isposinf(self.__values).sum()},"  # type: ignore
                + f"{np.isneginf(self.__values).sum()},"  # type: ignore
                + f"{np.isnan(self.__values).sum()}"  # type: ignore
            )

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
        # TODO: 次元が減る場合の処理
        if self.__gf.values is None:
            raise ValueError("データがありません")

        if type(key) == tuple:
            if len(key) == 3:
                return GridFrame(
                    data=self.__gf.values.__getitem__(key),
                    columns=self.__gf.columns.__getitem__(key[-1]),
                )
        return GridFrame(
            data=self.__gf.values.__getitem__(key), columns=self.__gf.columns
        )

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
    assert np.prod(shape) == data.shape[0], f"shapeが不適切です{shape},{data.shape[0]}"
    return GridFrame(
        data=data.reshape(list(shape) + [data.shape[-1]]), columns=list(df.columns)
    )
