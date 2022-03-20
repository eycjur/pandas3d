import logging


def get_logger(name: str) -> logging.Logger:
    """ライブラリ内で使用するロガーを取得する。

    Args:
        name (str): ロガー名(__name__を推奨)
        level (str): ログレベル

    Returns:
        getLogger: ロガーオブジェクト

    Example:
        >>> from pandas3d import util.logging
        >>> logger = logging.get_logger(__name__)
        >>> logger.warning("warning")
    """
    logger = logging.getLogger(name=name)
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.DEBUG)
    logger.propagate = True
    return logger


def root_logger() -> logging.Logger:
    """loggingに関する設定"""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        "\n%(asctime)s %(name)s %(lineno)s [%(levelname)s]:\n%(message)s"
    )

    # ストリームハンドラの設定
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
