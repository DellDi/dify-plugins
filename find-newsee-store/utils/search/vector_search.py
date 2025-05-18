"""
向量搜索模块
实现基于向量嵌入的语义搜索
"""

from typing import Dict, List, Any
import logging
import math

from .keyword_extraction import KeywordExtractor

logger = logging.getLogger(__name__)

class VectorSearcher:
    """向量搜索器，用于实现基于向量嵌入的语义搜索"""

    def __init__(self, threshold: float = 0.3):
        """
        初始化向量搜索器

        Args:
            threshold: 向量搜索阈值，默认为0.3
        """
        self.threshold = threshold
        self.keyword_extractor = KeywordExtractor()

    def search(self, query: str, collections: Dict[str, Any], entity_type: str = None, top_k: int = 5) -> Dict[str, Any]:
        """
        执行向量搜索

        Args:
            query: 查询文本
            collections: 集合字典，键为实体类型，值为集合对象
            entity_type: 实体类型，如果指定则只搜索该类型
            top_k: 返回结果数量

        Returns:
            包含搜索结果的字典
        """
        results = []
        query = query.strip()

        # 对长句进行关键词提取
        original_query = query
        if len(query) > 15:  # 对长句进行关键词提取
            keywords_list = self.keyword_extractor.extract(query)
            logger.info(f"长句关键词提取: '{original_query}' -> {keywords_list}")
            # 将关键词列表作为查询文本
            query_for_search = keywords_list
        else:
            # 对于短句，直接使用原始查询，但仍需要包装为列表
            query_for_search = [query]

        # 确定要搜索的集合
        collections_to_search = (
            [entity_type] if entity_type else collections.keys()
        )

        for entity_type in collections_to_search:
            if entity_type not in collections:
                continue

            collection = collections[entity_type]

            try:
                # 执行向量搜索 - 增加结果数量以提高召回率
                search_results = collection.query(
                    query_texts=query_for_search,  # 现在query_for_search本身就是列表
                    n_results=min(top_k * 3, 10)
                )

                # 处理结果
                if search_results["ids"] and search_results["ids"][0]:
                    for i, doc_id in enumerate(search_results["ids"][0]):
                        metadata = search_results["metadatas"][0][i]
                        name = metadata.get('name', '')

                        # 获取原始距离
                        distance = float(search_results["distances"][0][i])

                        # 计算相似度，确保在 0-1 范围内
                        # ChromaDB 返回的是欧几里得距离，需要转换为相似度
                        # 使用指数衰减函数进行转换，以获得更好的相似度分布
                        similarity = max(0.0, min(1.0, math.exp(-distance)))

                        # 对于长句查询，进行额外的相似度调整
                        if len(original_query) > 15:
                            # 检查实体名称是否出现在原始查询中
                            if name in original_query:
                                similarity = max(similarity, 0.75)  # 如果实体名称在原始查询中，给予更高的相似度

                            # 检查实体名称的部分是否出现在原始查询中
                            if len(name) >= 2:
                                for j in range(len(name) - 1):
                                    part = name[j:j+2]
                                    if part in original_query and len(part) >= 2:
                                        similarity = max(similarity, 0.6)  # 如果实体名称的部分在原始查询中，给予较高的相似度

                        # 记录所有向量搜索结果，便于调试
                        logger.info(f"向量搜索: {original_query} -> {name} = {similarity:.2f} (距离: {distance:.2f})")

                        # 对长句查询降低阈值要求
                        threshold = self.threshold
                        if len(original_query) > 15:  # 长句查询
                            threshold = max(0.2, threshold * 0.5)  # 显著降低阈值

                        if similarity >= threshold:
                            results.append(
                                {
                                    "id": metadata.get("id"),
                                    "name": name,
                                    "type": entity_type,
                                    "similarity": round(similarity, 2),
                                    "match_type": "vector",
                                    "metadata": metadata,
                                }
                            )
            except Exception as e:
                logger.error(f"向量搜索出错 ({entity_type}): {e}", exc_info=True)

        # 按相似度排序
        results.sort(key=lambda x: x["similarity"], reverse=True)

        return {"found": len(results) > 0, "results": results[:top_k]}
