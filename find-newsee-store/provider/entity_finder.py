import os
import json
from typing import Dict, List, Optional, Tuple, Any, Union
import jieba
from Levenshtein import distance
import chromadb
from chromadb.utils import embedding_functions
import logging

# 导入示例数据和配置
from .data.sample_data import (
    SAMPLE_PROJECTS,
    SAMPLE_PROPERTIES,
    SAMPLE_TARGETS,
    STOP_WORDS,
    DEFAULT_CONFIG,
)

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EntityFinder:
    """实体查找器，用于从文本中识别项目和房产实体"""

    def __init__(self, data_dir: str = "./data"):
        """初始化实体查找器

        Args:
            data_dir: 数据存储目录
        """
        self.data_dir = data_dir
        os.makedirs(os.path.join(data_dir, "chroma_db"), exist_ok=True)

        # 初始化嵌入函数
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"  # 多语言模型，支持中文
        )

        # 加载配置
        self.config = DEFAULT_CONFIG.copy()
        self.fuzzy_match_threshold = self.config["fuzzy_match_threshold"]
        self.vector_search_threshold = self.config["vector_search_threshold"]

        # 初始化ChromaDB客户端
        self.client = chromadb.PersistentClient(
            path=os.path.join(data_dir, "chroma_db")
        )

        self.target_collection = self.client.get_or_create_collection(
            name="targets",
            embedding_function=self.embedding_function,
            metadata={"description": "指标信息"},
        )

        self.project_collection = self.client.get_or_create_collection(
            name="projects",
            embedding_function=self.embedding_function,
            metadata={"description": "项目信息"},
        )

        self.property_collection = self.client.get_or_create_collection(
            name="properties",
            embedding_function=self.embedding_function,
            metadata={"description": "房产信息"},
        )

        # 初始化数据
        self._initialize_data()

    def _calculate_similarity(self, s1: str, s2: str) -> float:
        """计算两个字符串的相似度（基于编辑距离）

        Args:
            s1: 字符串1
            s2: 字符串2

        Returns:
            float: 相似度分数（0-1之间，1表示完全相同）
        """
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 1.0
        return 1 - distance(s1, s2) / max_len

    def _fuzzy_match(
        self, query: str, candidates: List[str], threshold: float = None
    ) -> List[Tuple[str, float]]:
        """模糊匹配查询字符串与候选字符串列表

        Args:
            query: 查询字符串
            candidates: 候选字符串列表
            threshold: 相似度阈值，低于此值的结果将被过滤

        Returns:
            List[Tuple[str, float]]: 匹配结果列表，每个元素是(候选字符串, 相似度分数)元组
        """
        if threshold is None:
            threshold = self.fuzzy_match_threshold

        results = []
        for candidate in candidates:
            score = self._calculate_similarity(query, candidate)
            if score >= threshold:
                results.append((candidate, score))

        # 按相似度降序排序
        return sorted(results, key=lambda x: x[1], reverse=True)

    def _find_entities_fuzzy(
        self, query: str, collection_name: str, field_name: str, top_k: int = 3
    ) -> List[Dict]:
        """使用模糊匹配查找实体

        Args:
            query: 查询文本
            collection_name: 集合名称（projects或properties）
            field_name: 要匹配的字段名
            top_k: 返回的结果数量

        Returns:
            List[Dict]: 匹配的实体列表
        """
        collection = self.client.get_collection(collection_name)

        # 获取所有候选实体
        results = collection.get(include=["documents", "metadatas"])
        candidates = []
        for doc, metadata in zip(results["documents"], results["metadatas"]):
            candidates.append(
                {
                    "id": metadata.get("id"),
                    "name": metadata.get(field_name, ""),
                    "metadata": metadata,
                }
            )

        # 执行模糊匹配
        matched = self._fuzzy_match(
            query, [c["name"] for c in candidates], threshold=self.fuzzy_match_threshold
        )

        # 构建结果
        results = []
        matched_names = {name for name, _ in matched[:top_k]}
        for candidate in candidates:
            if candidate["name"] in matched_names:
                results.append(
                    {
                        "id": candidate["id"],
                        "name": candidate["name"],
                        "similarity": next(
                            score
                            for name, score in matched
                            if name == candidate["name"]
                        ),
                        "source": "fuzzy_match",
                        "metadata": candidate["metadata"],
                    }
                )

        return results

    def _initialize_data(self, targets: List[Dict] = None, projects: List[Dict] = None, properties: List[Dict] = None):
        """初始化数据
        
        Args:
            targets: 指标数据列表，如果为None则使用示例数据
            projects: 项目数据列表，如果为None则使用示例数据
            properties: 房产数据列表，如果为None则使用示例数据
        """
        # 清空现有数据
        self.project_collection.delete(where={"id": {"$ne": ""}})  # 删除所有文档
        self.property_collection.delete(where={"id": {"$ne": ""}})

        # 使用传入的数据或示例数据
        targets_to_add = targets if targets is not None else SAMPLE_TARGETS
        projects_to_add = projects if projects is not None else SAMPLE_PROJECTS
        properties_to_add = properties if properties is not None else SAMPLE_PROPERTIES

        # 添加指标数据
        if targets_to_add:
            self.target_collection.add(
                documents=[f"{t['name']}" for t in targets_to_add],
                metadatas=[
                    {
                        "id": t["id"],
                        "name": t["name"],
                        "type": "target",
                    }
                    for t in targets_to_add
                ],
                ids=[f"target_{idx}" for idx in range(len(targets_to_add))],
            )
        # 添加项目数据
        if projects_to_add:
            self.project_collection.add(
                documents=[f"{p['name']} 位于{p.get('location', '')}" for p in projects_to_add],
                metadatas=[
                    {
                        "id": p["id"],
                        "name": p["name"],
                        "location": p.get("location", ""),
                        "type": "project",
                    }
                    for p in projects_to_add
                ],
                ids=[f"project_{p['id']}" for p in projects_to_add],
            )
            logger.info(f"已添加{len(projects_to_add)}条项目数据")

        # 添加房产数据
        if properties_to_add:
            self.property_collection.add(
                documents=[f"{p['name']} {p.get('rooms', '')}居室" for p in properties_to_add],
                metadatas=[
                    {
                        "id": p["id"],
                        "name": p["name"],
                        "project_id": p.get("project_id", ""),
                        "rooms": p.get("rooms", 0),
                        "type": "property",
                    }
                    for p in properties_to_add
                ],
                ids=[f"property_{p['id']}" for p in properties_to_add],
            )
            logger.info(f"已添加{len(properties_to_add)}条房产数据")

    def find_entities(
        self, query: str, top_k: int = 3, enable_fuzzy: bool = True
    ) -> Dict[str, Any]:
        """查找实体

        Args:
            query: 查询文本
            top_k: 返回结果数量
            enable_fuzzy: 是否启用模糊匹配

        Returns:
            Dict: 查询结果，包含以下字段：
                - found: 是否找到匹配项
                - type: 匹配类型（'exact', 'fuzzy', 'vector'）
                - results: 匹配结果列表
                - query: 原始查询
        """
        # 预处理查询文本
        processed_query = self._preprocess_text(query)

        # 1. 优先尝试精确匹配
        exact_results = self._exact_match(processed_query)
        if exact_results["found"]:
            logger.info(f"找到精确匹配: {exact_results}")
            return exact_results

        # 2. 如果启用模糊匹配，尝试模糊匹配
        if enable_fuzzy:
            fuzzy_results = []

            # 在项目中查找模糊匹配
            project_matches = self._find_entities_fuzzy(
                query=query, collection_name="projects", field_name="name", top_k=top_k
            )

            # 在房产中查找模糊匹配
            property_matches = self._find_entities_fuzzy(
                query=query,
                collection_name="properties",
                field_name="name",
                top_k=top_k,
            )

            # 合并并排序结果
            all_matches = []
            for match in project_matches:
                all_matches.append(
                    {
                        "type": "project",
                        "id": match["id"],
                        "name": match["name"],
                        "similarity": match["similarity"],
                        "metadata": match["metadata"],
                    }
                )

            for match in property_matches:
                all_matches.append(
                    {
                        "type": "property",
                        "id": match["id"],
                        "name": match["name"],
                        "similarity": match["similarity"],
                        "metadata": match["metadata"],
                    }
                )

            # 按相似度降序排序
            all_matches.sort(key=lambda x: x["similarity"], reverse=True)

            if all_matches:
                logger.info(f"找到模糊匹配: {all_matches[:top_k]}")
                return {
                    "found": True,
                    "type": "fuzzy",
                    "results": all_matches[:top_k],
                    "query": query,
                }

        # 3. 最后尝试向量检索
        vector_results = self._vector_search(query, top_k)
        if vector_results["found"]:
            logger.info(f"向量检索结果: {vector_results}")
            return vector_results

        # 如果所有方法都未找到结果
        return {
            "found": False,
            "type": "none",
            "message": "未找到匹配的实体",
            "query": query,
        }

    def _preprocess_text(self, text: str) -> str:
        """预处理文本，分词并去除停用词"""
        # 使用jieba进行分词
        words = jieba.cut(text)
        # 过滤停用词
        stop_words = STOP_WORDS
        filtered_words = [
            word for word in words if word not in stop_words and len(word) > 1
        ]
        return " ".join(filtered_words)

    def _exact_match(self, query: str) -> Dict[str, Any]:
        """尝试精确匹配

        Args:
            query: 预处理后的查询文本

        Returns:
            Dict: 匹配结果
        """
        # 从项目集合中查询
        project_results = self.project_collection.get()

        # 从房产集合中查询
        property_results = self.property_collection.get()

        matches = []

        # 检查项目名称
        for i, doc_id in enumerate(project_results["ids"]):
            name = project_results["metadatas"][i]["name"]
            if name in query:
                matches.append(
                    {
                        "id": project_results["metadatas"][i]["id"],
                        "name": name,
                        "type": "project",
                        "confidence": 1.0,  # 精确匹配，置信度为1
                        "match_type": "exact",
                    }
                )

        # 检查房产名称
        for i, doc_id in enumerate(property_results["ids"]):
            name = property_results["metadatas"][i]["name"]
            if name in query:
                matches.append(
                    {
                        "id": property_results["metadatas"][i]["id"],
                        "name": name,
                        "type": "property",
                        "confidence": 1.0,
                        "match_type": "exact",
                    }
                )

        if matches:
            return {"found": True, "results": matches}

        return {"found": False}

    def _vector_search(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """向量搜索

        Args:
            query: 原始查询文本
            top_k: 返回结果数量

        Returns:
            Dict: 搜索结果
        """
        # 在项目集合中搜索
        project_results = self.project_collection.query(
            query_texts=[query], n_results=top_k
        )

        # 在房产集合中搜索
        property_results = self.property_collection.query(
            query_texts=[query], n_results=top_k
        )

        results = []

        # 处理项目结果
        if project_results["ids"]:
            for i, doc_id in enumerate(project_results["ids"][0]):
                confidence = 1.0 - min(1.0, project_results["distances"][0][i] / 2.0)
                if confidence >= 0.6:  # 置信度阈值
                    results.append(
                        {
                            "id": project_results["metadatas"][0][i]["id"],
                            "name": project_results["metadatas"][0][i]["name"],
                            "type": "project",
                            "confidence": round(confidence, 2),
                            "match_type": "vector",
                        }
                    )

        # 处理房产结果
        if property_results["ids"]:
            for i, doc_id in enumerate(property_results["ids"][0]):
                confidence = 1.0 - min(1.0, property_results["distances"][0][i] / 2.0)
                if confidence >= 0.6:  # 置信度阈值
                    results.append(
                        {
                            "id": property_results["metadatas"][0][i]["id"],
                            "name": property_results["metadatas"][0][i]["name"],
                            "type": "property",
                            "confidence": round(confidence, 2),
                            "match_type": "vector",
                        }
                    )

        # 按置信度排序
        results.sort(key=lambda x: x["confidence"], reverse=True)

        return {
            "found": len(results) > 0,
            "results": results[:top_k],  # 返回前top_k个结果
        }

    def sync_data(
        self, projects: List[Dict], properties: List[Dict]
    ) -> Tuple[int, int]:
        """同步数据到ChromaDB

        Args:
            projects: 项目数据列表
            properties: 房产数据列表

        Returns:
            Tuple[int, int]: 添加的项目数和房产数
        """
        # 清空现有集合
        self.project_collection.delete(ids=self.project_collection.get()["ids"])
        self.property_collection.delete(ids=self.property_collection.get()["ids"])

        # 添加项目数据
        if projects:
            self.project_collection.add(
                documents=[
                    f"{p['name']} 位于{p.get('location', '')}" for p in projects
                ],
                metadatas=[{**p, "type": "project"} for p in projects],
                ids=[f"project_{idx}" for idx in range(len(projects))],
            )

        # 添加房产数据
        if properties:
            self.property_collection.add(
                documents=[f"{p['name']} {p.get('rooms', '')}居室" for p in properties],
                metadatas=[{**p, "type": "property"} for p in properties],
                ids=[f"property_{idx}" for idx in range(len(properties))],
            )

        return len(projects), len(properties)
