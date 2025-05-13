import os
import logging
from dify_plugin import Plugin, DifyPluginEnv

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 确保数据目录存在
data_dir = os.path.join(os.path.dirname(__file__), "data", "chroma_db")
os.makedirs(data_dir, exist_ok=True)
logger.info(f"数据目录已创建: {data_dir}")

# 初始化插件
plugin = Plugin(DifyPluginEnv(MAX_REQUEST_TIMEOUT=120))

if __name__ == '__main__':
    logger.info("启动Dify实体查找插件服务...")
    plugin.run()
