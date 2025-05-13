import os
import json
from typing import Dict, List, Optional, Tuple, Any
import jieba
from Levenshtein import distance
import chromadb
from chromadb.utils import embedding_functions
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
        
        # 初始化ChromaDB客户端
        self.client = chromadb.PersistentClient(path=os.path.join(data_dir, "chroma_db"))
        
        # 创建或获取集合
        self.project_collection = self.client.get_or_create_collection(
            name="projects",
            embedding_function=self.embedding_function,
            metadata={"description": "项目信息"}
        )
        
        self.property_collection = self.client.get_or_create_collection(
            name="properties",
            embedding_function=self.embedding_function,
            metadata={"description": "房产信息"}
        )
        
        # 初始化示例数据
        self._initialize_sample_data()
    
    def _initialize_sample_data(self):
        """初始化示例数据，实际应用中应从数据库读取"""
        # 检查集合是否为空，如果为空则添加示例数据
        if len(self.project_collection.get()["ids"]) == 0:
            # 示例项目数据
            project_data = [
                {"id": "P001", "name": "星河湾项目", "location": "北京朝阳区"},
                {"id": "P002", "name": "翡翠城", "location": "上海浦东新区"},
                {"id": "P003", "name": "金色家园", "location": "广州天河区"},
                {"id": "P004", "name": "阳光小区", "location": "深圳南山区"},
                {"id": "P005", "name": "绿地国际花都", "location": "杭州西湖区"}
            ]
            
            # 添加项目数据到ChromaDB
            self.project_collection.add(
                documents=[f"{p['name']} 位于{p['location']}" for p in project_data],
                metadatas=[{"id": p["id"], "name": p["name"], "location": p["location"], "type": "project"} for p in project_data],
                ids=[f"project_{idx}" for idx in range(len(project_data))]
            )
            
            logger.info(f"已添加{len(project_data)}条项目示例数据")
        
        if len(self.property_collection.get()["ids"]) == 0:
            # 示例房产数据
            property_data = [
                {"id": "R001", "name": "星河湾1号楼", "project_id": "P001", "rooms": 3},
                {"id": "R002", "name": "星河湾2号楼", "project_id": "P001", "rooms": 4},
                {"id": "R003", "name": "翡翠城A区", "project_id": "P002", "rooms": 2},
                {"id": "R004", "name": "金色家园1期", "project_id": "P003", "rooms": 3},
                {"id": "R005", "name": "阳光小区B栋", "project_id": "P004", "rooms": 2}
            ]
            
            # 添加房产数据到ChromaDB
            self.property_collection.add(
                documents=[f"{p['name']} {p.get('rooms', '')}居室" for p in property_data],
                metadatas=[{"id": p["id"], "name": p["name"], "project_id": p["project_id"], "rooms": p["rooms"], "type": "property"} for p in property_data],
                ids=[f"property_{idx}" for idx in range(len(property_data))]
            )
            
            logger.info(f"已添加{len(property_data)}条房产示例数据")
    
    def find_entities(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """查找实体
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            Dict: 查询结果
        """
        # 预处理查询文本
        processed_query = self._preprocess_text(query)
        
        # 优先尝试精确匹配
        exact_results = self._exact_match(processed_query)
        if exact_results["found"]:
            logger.info(f"找到精确匹配: {exact_results}")
            return exact_results
        
        # 若无精确匹配，进行向量检索
        vector_results = self._vector_search(query, top_k)
        logger.info(f"向量检索结果: {vector_results}")
        return vector_results
    
    def _preprocess_text(self, text: str) -> str:
        """预处理文本，分词并去除停用词"""
        # 使用jieba进行分词
        words = jieba.cut(text)
        # 简单过滤停用词
        stop_words = {"的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这"}
        filtered_words = [word for word in words if word not in stop_words and len(word) > 1]
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
                matches.append({
                    "id": project_results["metadatas"][i]["id"],
                    "name": name,
                    "type": "project",
                    "confidence": 1.0,  # 精确匹配，置信度为1
                    "match_type": "exact"
                })
        
        # 检查房产名称
        for i, doc_id in enumerate(property_results["ids"]):
            name = property_results["metadatas"][i]["name"]
            if name in query:
                matches.append({
                    "id": property_results["metadatas"][i]["id"],
                    "name": name,
                    "type": "property",
                    "confidence": 1.0,
                    "match_type": "exact"
                })
        
        if matches:
            return {
                "found": True,
                "results": matches
            }
        
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
            query_texts=[query],
            n_results=top_k
        )
        
        # 在房产集合中搜索
        property_results = self.property_collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        results = []
        
        # 处理项目结果
        if project_results["ids"]:
            for i, doc_id in enumerate(project_results["ids"][0]):
                confidence = 1.0 - min(1.0, project_results["distances"][0][i] / 2.0)
                if confidence >= 0.6:  # 置信度阈值
                    results.append({
                        "id": project_results["metadatas"][0][i]["id"],
                        "name": project_results["metadatas"][0][i]["name"],
                        "type": "project",
                        "confidence": round(confidence, 2),
                        "match_type": "vector"
                    })
        
        # 处理房产结果
        if property_results["ids"]:
            for i, doc_id in enumerate(property_results["ids"][0]):
                confidence = 1.0 - min(1.0, property_results["distances"][0][i] / 2.0)
                if confidence >= 0.6:  # 置信度阈值
                    results.append({
                        "id": property_results["metadatas"][0][i]["id"],
                        "name": property_results["metadatas"][0][i]["name"],
                        "type": "property",
                        "confidence": round(confidence, 2),
                        "match_type": "vector"
                    })
        
        # 按置信度排序
        results.sort(key=lambda x: x["confidence"], reverse=True)
        
        return {
            "found": len(results) > 0,
            "results": results[:top_k]  # 返回前top_k个结果
        }
    
    def sync_data(self, projects: List[Dict], properties: List[Dict]) -> Tuple[int, int]:
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
                documents=[f"{p['name']} 位于{p.get('location', '')}" for p in projects],
                metadatas=[{**p, "type": "project"} for p in projects],
                ids=[f"project_{idx}" for idx in range(len(projects))]
            )
        
        # 添加房产数据
        if properties:
            self.property_collection.add(
                documents=[f"{p['name']} {p.get('rooms', '')}居室" for p in properties],
                metadatas=[{**p, "type": "property"} for p in properties],
                ids=[f"property_{idx}" for idx in range(len(properties))]
            )
        
        return len(projects), len(properties)
