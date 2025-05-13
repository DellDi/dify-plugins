from typing import Any, Dict
import os
import logging

from dify_plugin import ToolProvider
from dify_plugin.errors.tool import ToolProviderCredentialValidationError

from provider.entity_finder import EntityFinder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FindNewseeStoreProvider(ToolProvider):
    def __init__(self):
        super().__init__()
        self.entity_finder = None
        
    def _validate_credentials(self, credentials: dict[str, Any]) -> None:
        try:
            # 验证凭据，这里简单验证一下必要的字段
            required_fields = []
            for field in required_fields:
                if field not in credentials:
                    raise ValueError(f"Missing required credential: {field}")
                    
            # 初始化实体查找器
            data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
            self.entity_finder = EntityFinder(data_dir=data_dir)
            logger.info("Entity finder initialized successfully")
        except Exception as e:
            raise ToolProviderCredentialValidationError(str(e))
