"""
模糊匹配模块
实现基于Levenshtein距离的模糊字符串匹配
"""

from typing import Dict, List, Any
import logging
from Levenshtein import distance

logger = logging.getLogger(__name__)

class FuzzyMatcher:
    """模糊匹配器，用于实现基于Levenshtein距离的模糊字符串匹配"""
    
    def __init__(self, threshold: float = 0.6):
        """
        初始化模糊匹配器
        
        Args:
            threshold: 模糊匹配阈值，默认为0.6
        """
        self.threshold = threshold
    
    def calculate_similarity(self, s1: str, s2: str) -> float:
        """
        计算两个字符串的相似度（基于编辑距离）
        
        Args:
            s1: 第一个字符串
            s2: 第二个字符串
            
        Returns:
            相似度，范围为0-1，1表示完全相同
        """
        # 预处理字符串
        s1 = s1.lower().strip()
        s2 = s2.lower().strip()

        # 如果有一个是空字符串
        if not s1 or not s2:
            return 0.0

        # 如果完全相同
        if s1 == s2:
            return 1.0

        # 如果一个是另一个的子串，给予较高的相似度
        if s1 in s2 or s2 in s1:
            shorter = s1 if len(s1) < len(s2) else s2
            longer = s2 if len(s1) < len(s2) else s1
            return 0.8 + 0.2 * (len(shorter) / len(longer))

        # 基于编辑距离计算相似度
        max_len = max(len(s1), len(s2))
        edit_dist = distance(s1, s2)

        # 对于短文本，编辑距离的影响更大，因此使用非线性变换
        if max_len < 5:
            # 对于非常短的文本，每个字符的差异都很重要
            return max(0, 1 - (edit_dist / max_len) * 1.5)
        else:
            return 1 - edit_dist / max_len
    
    def search(self, query: str, collections: Dict[str, Any], entity_type: str = None, top_k: int = 5) -> Dict[str, Any]:
        """
        执行模糊匹配搜索
        
        Args:
            query: 查询文本
            collections: 集合字典，键为实体类型，值为集合对象
            entity_type: 实体类型，如果指定则只搜索该类型
            top_k: 返回结果数量
            
        Returns:
            包含搜索结果的字典
        """
        results = []
        query_lower = query.lower().strip()

        # 确定要搜索的集合
        collections_to_search = (
            [entity_type] if entity_type else collections.keys()
        )

        for entity_type in collections_to_search:
            if entity_type not in collections:
                continue

            collection = collections[entity_type]
            items = collection.get()

            # 获取所有文档名称
            names = []
            metadatas = []
            for i, doc_id in enumerate(items["ids"]):
                metadata = items["metadatas"][i]
                name = metadata.get("name", "")
                if name:
                    names.append(name)
                    metadatas.append(metadata)

            # 执行模糊匹配
            for i, name in enumerate(names):
                # 对查询和名称进行预处理
                name_lower = name.lower().strip()

                # 检查包含关系（之前在精确匹配中处理的）
                if name_lower in query_lower or query_lower in name_lower:
                    # 如果是包含关系，给予较高的相似度
                    shorter = name_lower if len(name_lower) < len(query_lower) else query_lower
                    longer = query_lower if len(name_lower) < len(query_lower) else name_lower
                    similarity = 0.85 + 0.15 * (len(shorter) / len(longer))
                else:
                    # 计算双向相似度（查询->名称和名称->查询）
                    similarity1 = self.calculate_similarity(query_lower, name_lower)
                    similarity2 = self.calculate_similarity(name_lower, query_lower)
                    similarity = max(similarity1, similarity2)

                # 对于短文本，降低阈值
                threshold = self.threshold
                if len(name) < 5 or len(query) < 5:
                    threshold = threshold * 0.8

                if similarity >= threshold:
                    logger.info(f"模糊匹配: {query} -> {name} = {similarity}")
                    results.append(
                        {
                            "id": metadatas[i].get("id"),
                            "name": name,
                            "type": entity_type,
                            "similarity": round(similarity, 2),
                            "match_type": "fuzzy",
                            "metadata": metadatas[i],
                        }
                    )

        # 按相似度排序
        results.sort(key=lambda x: x["similarity"], reverse=True)

        return {"found": len(results) > 0, "results": results[:top_k]}
