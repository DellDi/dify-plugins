from typing import Any, Dict, Optional
import os
import logging
import asyncio

from dify_plugin import ToolProvider
from dify_plugin.errors.tool import ToolProviderCredentialValidationError
import shutil

# 导入新的EntityFinderMySQL
from provider.entity_finder_mysql import EntityFinderMySQL
import utils.provider

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message:s)"
)
logger = logging.getLogger(__name__)


class FindNewseeStoreProvider(ToolProvider):
    """FindNewseeStore Dify插件提供程序"""

    def __init__(self):
        super().__init__()
        self.entity_finder: Optional[EntityFinderMySQL] = None
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

    def _validate_credentials(self, credentials: Dict[str, Any]) -> None:
        """验证凭据并初始化实体查找器

        Args:
            credentials: 包含数据库连接字符串的字典，键为'mysql_url'
                示例: {"mysql_url": "mysql://username:password@host:port/database"}
        """
        # 检查mysql_url参数是否存在
        if "mysql_url" not in credentials or not credentials["mysql_url"]:
            raise ToolProviderCredentialValidationError("缺少必要的MySQL连接字符串")

        has_chroma_db = os.path.exists(
            os.path.join(self.data_dir, "chroma_db", "chroma.sqlite3")
        )

        try:
            # 如果目录和chroma_db/chroma.sqlite3文件存在则跳过
            if os.path.exists(self.data_dir) and has_chroma_db:
                logger.info("数据目录和chroma.sqlite3文件已存在，跳过创建")
            else:
                # 创建数据目录
                os.makedirs(self.data_dir, exist_ok=True)

            # 安全关闭之前的实例
            if utils.provider.provider_entity_finder is not None:
                logger.info("关闭现有实体查找器实例...")
                utils.provider.provider_entity_finder.close_chroma()
                self.entity_finder and hasattr(self.entity_finder, "close_chroma") and self.entity_finder.close_chroma()

                # 强制垃圾回收
                import gc

                gc.collect()
                import time

                time.sleep(1)  # 等待资源释放
                shutil.rmtree(os.path.join(self.data_dir, "chroma_db"))
                os.makedirs(self.data_dir, exist_ok=True)

            logger.info("现有实体查找器已关闭")

            # 初始化实体查找器
            logger.info("开始初始化新的实体查找器...")
            self.entity_finder = EntityFinderMySQL(
                data_dir=self.data_dir,
                embedding_api_key=credentials.get("embedding_api_key"),
                reset_collections=True,  # 启用集合重置以确保使用正确的嵌入模型
            )
            logger.info("实体查找器初始化成功，已启用集合重置")

            # 解析MySQL连接字符串
            mysql_url = credentials["mysql_url"]
            db_config = self._parse_mysql_url(mysql_url)

            # 异步初始化数据库连接
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._initialize_finder(db_config))
            loop.close()

            # 设置全局实体查找器实例
            utils.provider.provider_entity_finder = self.entity_finder
            logger.info("实体查找器初始化成功")

        except Exception as e:
            error_msg = f"初始化实体查找器失败: {str(e)}"
            logger.error(error_msg, exc_info=True)

            if "Cannot run the event loop while another loop is running" in str(e):
                raise ToolProviderCredentialValidationError(
                    "还在初始化中，别着急，等几分钟"
                ) from e
            raise ToolProviderCredentialValidationError(error_msg) from e

    def _parse_mysql_url(self, url: str) -> Dict[str, Any]:
        """解析MySQL连接字符串

        Args:
            url: MySQL连接字符串，格式: mysql://username:password@host:port/database

        Returns:
            包含连接参数的字典
        """
        if not url.startswith("mysql://"):
            raise ValueError("无效的MySQL连接字符串，必须以'mysql://'开头")

        try:
            # 移除mysql://前缀
            url = url[8:]

            # 分割用户名密码和主机部分
            if "@" in url:
                auth_part, host_part = url.split("@", 1)
                if ":" in auth_part:
                    user, password = auth_part.split(":", 1)
                else:
                    user = auth_part
                    password = ""
            else:
                host_part = url
                user = ""
                password = ""

            # 分割主机和数据库
            if "/" in host_part:
                host_port, database = host_part.rsplit("/", 1)
                if ":" in host_port:
                    host, port = host_port.split(":", 1)
                    port = int(port)
                else:
                    host = host_port
                    port = 3306  # 默认端口
            else:
                raise ValueError("未指定数据库名称")

            # 移除查询参数
            if "?" in database:
                database = database.split("?")[0]

            if not all([host, database]):
                raise ValueError("主机名和数据库名不能为空")

            return {
                "host": host,
                "port": port,
                "user": user,
                "password": password,
                "database": database,
            }

        except Exception as e:
            raise ValueError(f"解析MySQL连接字符串失败: {str(e)}") from e

    async def _initialize_finder(self, db_config: Dict[str, Any]):
        """异步初始化实体查找器"""
        if self.entity_finder is None:
            raise RuntimeError("实体查找器未初始化")

        # 确保端口是整数
        if "port" in db_config and isinstance(db_config["port"], str):
            db_config["port"] = int(db_config["port"])

        # 调用异步初始化方法
        await self.entity_finder.initialize(db_config)

    def __del__(self):
        """清理资源"""
        if hasattr(self, "entity_finder") and self.entity_finder:
            try:
                self.entity_finder.close()
            except Exception as e:
                logger.error(f"关闭实体查找器时出错: {e}")

        # 强制垃圾回收
        try:
            import gc

            gc.collect()
        except:
            pass
