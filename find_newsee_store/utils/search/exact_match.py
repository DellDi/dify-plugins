"""
精确匹配模块
实现基于精确字符串匹配的实体搜索
"""

from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class ExactMatcher:
    """精确匹配器，用于实现基于精确字符串匹配的实体搜索"""
    
    def __init__(self):
        """初始化精确匹配器"""
        pass
    
    def search(self, query: str, collections: Dict[str, Any], entity_type: str = None) -> Dict[str, Any]:
        """
        执行精确匹配搜索
        
        Args:
            query: 查询文本
            collections: 集合字典，键为实体类型，值为集合对象
            entity_type: 实体类型，如果指定则只搜索该类型
            
        Returns:
            包含搜索结果的字典
        """
        results = []
        query_lower = query.lower().strip()
        
        # 确定要搜索的集合
        collections_to_search = {}
        if entity_type and entity_type in collections:
            # 如果指定了实体类型，只搜索该类型
            collections_to_search[entity_type] = collections[entity_type]
        else:
            # 否则搜索所有类型
            collections_to_search = collections

        # 在选定的集合中搜索
        for entity_type, collection in collections_to_search.items():
            # 获取所有文档
            items = collection.get()

            # 检查每个文档
            for i, doc_id in enumerate(items["ids"]):
                metadata = items["metadatas"][i]
                name = metadata.get("name", "")
                if not name:
                    continue

                name_lower = name.lower().strip()

                # 精确匹配应该是完全相等的关系
                if name_lower == query_lower:
                    results.append(
                        {
                            "id": metadata.get("id"),
                            "name": name,
                            "type": entity_type,
                            "similarity": 1.0,
                            "match_type": "exact",
                            "metadata": metadata,
                        }
                    )

        return {"found": len(results) > 0, "results": results}
