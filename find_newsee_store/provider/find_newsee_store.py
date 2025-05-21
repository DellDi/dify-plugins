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

        # 添加二级变量计算
        reset_collections = False
        
        # 先定义模型特定目录变量，确保全局可访问
        global model_specific_dir

        # 获取当前的嵌入模型类型
        current_embedding_model = "local" if not credentials.get("embedding_api_key") else "tongyi"
        
        # 预先设置模型特定目录，确保在任何条件下都可访问
        if current_embedding_model == "local":
            model_specific_dir = os.path.join(self.data_dir, "chroma_db_local")
        else:
            model_specific_dir = os.path.join(self.data_dir, "chroma_db_tongyi")

        # 如果存在配置文件，读取上次使用的嵌入模型类型
        config_file = os.path.join(self.data_dir, "config.txt")
        previous_embedding_model = None
        if os.path.exists(config_file):
            try:
                with open(config_file, "r") as f:
                    previous_embedding_model = f.read().strip()
            except Exception as e:
                logger.warning(f"读取配置文件失败: {e}")

        # 判断是否需要重置集合
        if previous_embedding_model and previous_embedding_model != current_embedding_model:
            reset_collections = True
            logger.info(f"检测到嵌入模型已更改: {previous_embedding_model} -> {current_embedding_model}，将重置集合")

        try:
            # 创建数据目录
            os.makedirs(self.data_dir, exist_ok=True)

            # 选择使用哪个存储目录，基于当前的嵌入模型
            if reset_collections:
                logger.info("嵌入模型已更改，采用多目录策略...")
                
                # 安全关闭之前的实例并释放资源
                if utils.provider.provider_entity_finder is not None:
                    logger.info("关闭现有实体查找器实例...")
                    try:
                        utils.provider.provider_entity_finder.close_chroma()
                    except Exception as e:
                        logger.warning(f"关闭全局实体查找器失败: {e}")
                    
                if self.entity_finder and hasattr(self.entity_finder, "close_chroma"):
                    try:
                        self.entity_finder.close_chroma()
                    except Exception as e:
                        logger.warning(f"关闭实体查找器失败: {e}")

                # 强制垃圾回收
                import gc
                gc.collect()
                import time
                time.sleep(1)  # 等待资源释放
                
                # 清除引用
                self.entity_finder = None
                utils.provider.provider_entity_finder = None
                gc.collect()
                
                # 基于当前使用的嵌入模型类型，确认我们选择了正确的目录
                if current_embedding_model == "local":
                    logger.info(f"使用本地模型的数据目录: {model_specific_dir}")
                else:  # tongyi
                    logger.info(f"使用通义模型的数据目录: {model_specific_dir}")
                
                # 创建目录（如果不存在）
                try:
                    os.makedirs(model_specific_dir, exist_ok=True)
                    logger.info(f"已确保数据目录 {model_specific_dir} 存在")
                except Exception as e:
                    logger.warning(f"创建目录 {model_specific_dir} 失败: {e}")
                
                # 创建非确定性链接，指向当前所使用的目录
                try:
                    current_symlink = os.path.join(self.data_dir, "chroma_db")
                    
                    # 如果已经存在链接或目录，先删除
                    if os.path.exists(current_symlink) or os.path.islink(current_symlink):
                        if os.path.islink(current_symlink):
                            os.unlink(current_symlink)  # 删除现有的符号链接
                            logger.info(f"已删除现有的符号链接: {current_symlink}")
                        else:
                            # 如果是目录而不是链接，将其重命名为备份
                            backup_dir = f"{current_symlink}_backup_{int(time.time())}"
                            os.rename(current_symlink, backup_dir)
                            logger.info(f"已将非链接的 chroma_db 目录备份为: {backup_dir}")
                            
                    # 创建新的符号链接
                    # 在macOS和Linux上使用相对路径创建符号链接
                    relative_path = os.path.relpath(model_specific_dir, os.path.dirname(current_symlink))
                    os.symlink(relative_path, current_symlink)
                    logger.info(f"已创建符号链接: {current_symlink} -> {relative_path}")
                except Exception as e:
                    logger.error(f"创建符号链接失败: {e}")
                    # 如果创建符号链接失败，则直接复制整个目录
                    try:
                        if os.path.exists(current_symlink):
                            shutil.rmtree(current_symlink)
                        # 我们这里如果太大、复制会很慢，用mkdir替代
                        os.makedirs(current_symlink, exist_ok=True)
                        logger.info(f"创建符号链接失败，改为直接创建目录: {current_symlink}")
                    except Exception as e2:
                        logger.error(f"创建目录也失败: {e2}")
                
            logger.info("现有实体查找器资源已处理")
            
            # 将当前嵌入模型类型保存到配置文件
            try:
                with open(config_file, "w") as f:
                    f.write(current_embedding_model)
                logger.info(f"已将当前嵌入模型类型({current_embedding_model})保存到配置文件")
            except Exception as e:
                logger.warning(f"写入配置文件失败: {e}")

            # 初始化实体查找器
            logger.info("开始初始化新的实体查找器...")
            # 使用模型特定目录而不是符号链接目录
            self.entity_finder = EntityFinderMySQL(
                data_dir=model_specific_dir,  # 使用模型特定目录
                embedding_api_key=credentials.get("embedding_api_key"),
                reset_collections=False,  # 无需重置集合，因为我们使用的是模型特定目录
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
