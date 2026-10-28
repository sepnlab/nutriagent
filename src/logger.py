"""统一日志配置。

用法:
    from src.logger import get_logger
    logger = get_logger(__name__)
    logger.info("...")
    logger.warning("...")
    logger.error("...")
"""
import logging
import sys


def setup_logging(level: int = logging.INFO):
    """初始化日志格式和级别。在 main() 入口处调用一次。"""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """获取模块级 logger。"""
    return logging.getLogger(name)
