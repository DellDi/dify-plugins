import os
import logging
from dify_plugin import Plugin, DifyPluginEnv

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 初始化插件
plugin = Plugin(DifyPluginEnv(MAX_REQUEST_TIMEOUT=120))

if __name__ == '__main__':
    logger.info("启动Dify实体查找插件服务...")
    plugin.run()
