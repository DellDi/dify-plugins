from collections.abc import Generator
from typing import Any, Dict
import logging

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage

# 导入实体查找器模块
import utils.provider

logger = logging.getLogger(__name__)

class FindNewseeStoreTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        try:
            # 获取查询参数
            query = tool_parameters.get('query', '')

            if not query:
                yield self.create_text_message("查询参数为空，请提供有效的查询文本")
                return

            # 获取实体查找器
            entity_finder = utils.provider.provider_entity_finder

            if not entity_finder:
                yield self.create_text_message("实体查找器未初始化，请检查插件配置")
                return

            # 执行实体查找
            result = entity_finder.search(query, entity_type="project", top_k=3)

            if result['found']:
                # 格式化返回结果
                response = {
                    "success": True,
                    "query": query,
                    "entities": result['results'],
                    "message": f"找到 {len(result['results'])} 个匹配实体"
                }
            else:
                response = {
                    "success": False,
                    "query": query,
                    "entities": [],
                    "message": "未找到匹配的实体"
                }

            # 取第一个中的name
            if result["found"]:
                name = response["results"][0]["name"]
                yield self.create_text_message(name)
            yield self.create_json_message(response)

        except Exception as e:
            logger.exception(f"实体查找失败: {str(e)}")
            yield self.create_text_message(f"处理查询时发生错误: {str(e)}")
