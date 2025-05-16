import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from Levenshtein import distance
import chromadb
from chromadb.utils import embedding_functions
import asyncio

# 导入数据库连接
from .database import DatabaseConnection, create_db_url

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 默认配置
DEFAULT_CONFIG = {
    "fuzzy_match_threshold": 0.8,  # 模糊匹配阈值
    "vector_search_threshold": 0.6,  # 向量搜索阈值
    "top_k": 3,  # 默认返回结果数量
    "enable_fuzzy": True,  # 是否启用模糊匹配
    "enable_vector_search": True,  # 是否启用向量搜索
}


class EntityFinderMySQL:
    """实体查找器，用于从MySQL数据库中加载数据并进行实体查找"""

    def __init__(self, data_dir: str = "./data"):
        """
        初始化实体查找器

        Args:
            data_dir: 数据存储目录
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # 初始化组件
        self.db = None
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"  # 多语言模型，支持中文
        )

        # 初始化ChromaDB客户端
        self.client = chromadb.PersistentClient(path=str(self.data_dir / "chroma_db"))

        # 初始化集合
        self.collections = {
            "project": self.client.get_or_create_collection(
                name="projects",
                embedding_function=self.embedding_function,
                metadata={"description": "项目信息"},
            ),
            "property": self.client.get_or_create_collection(
                name="properties",
                embedding_function=self.embedding_function,
                metadata={"description": "房产信息"},
            ),
            "target": self.client.get_or_create_collection(
                name="targets",
                embedding_function=self.embedding_function,
                metadata={"description": "指标信息"},
            ),
        }

        # 加载配置
        self.config = DEFAULT_CONFIG.copy()
        self.fuzzy_match_threshold = self.config["fuzzy_match_threshold"]
        self.vector_search_threshold = self.config["vector_search_threshold"]

        logger.info("实体查找器初始化完成")

    async def initialize(self, db_config: Dict[str, Any]):
        """
        初始化数据库连接并加载数据

        Args:
            db_config: 数据库配置字典，包含以下键：
                - host: 数据库主机
                - port: 数据库端口
                - user: 用户名
                - password: 密码
                - database: 数据库名
        """
        try:
            # 1. 创建数据库连接
            db_url = create_db_url(
                username=db_config["user"],
                password=db_config["password"],
                host=db_config["host"],
                port=int(db_config.get("port", 3306)),
                database=db_config["database"],
            )

            self.db = DatabaseConnection(db_url)
            logger.info("数据库连接成功")

            # 2. 加载数据到向量存储
            await self._load_initial_data()

        except Exception as e:
            logger.error(f"初始化失败: {e}")
            raise

    async def _load_initial_data(self):
        """从数据库加载初始数据到向量存储"""
        if not self.db:
            raise RuntimeError("数据库未初始化")

        logger.info("开始加载数据到向量存储...")

        # 加载项目数据
        projects = await self._load_entities("project")
        if projects:
            self._add_documents_to_collection("project", projects)

        # 加载房产数据
        properties = await self._load_entities("property")
        if properties:
            self._add_documents_to_collection("property", properties)

        # 加载指标数据
        targets = await self._load_entities("target")
        if targets:
            self._add_documents_to_collection("target", targets)

        logger.info("数据加载完成")

    def _add_documents_to_collection(self, entity_type: str, documents: List[Dict]):
        """将文档添加到对应的集合中"""
        if not documents:
            return

        collection = self.collections[entity_type]

        # 清空现有数据
        collection.delete(where={"id": {"$ne": ""}})

        # 批量大小
        batch_size = 1000
        total_docs = len(documents)
        logger.info(f"开始添加{total_docs}条{entity_type}数据到集合，批量大小: {batch_size}")
        
        # 分批处理
        for start_idx in range(0, total_docs, batch_size):
            end_idx = min(start_idx + batch_size, total_docs)
            batch_docs = documents[start_idx:end_idx]
            
            # 准备批量数据
            texts = []
            metadatas = []
            ids = []
            
            for i, doc in enumerate(batch_docs, start=start_idx):
                # 确保元数据值都是ChromaDB支持的类型
                metadata = {}
                for k, v in doc.items():
                    if k == "text":
                        continue
                    # 转换datetime为ISO格式字符串
                    if hasattr(v, "isoformat"):
                        metadata[k] = v.isoformat()
                    else:
                        metadata[k] = v
                
                texts.append(doc["text"])
                metadatas.append(metadata)
                ids.append(f"{entity_type}_{i}")
            
            # 添加当前批次数据到集合
            collection.add(documents=texts, metadatas=metadatas, ids=ids)
            logger.info(f"已添加批次 {start_idx//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size}: "
                       f"{start_idx+1}-{end_idx} 条{entity_type}数据")
        
        logger.info(f"完成添加{total_docs}条{entity_type}数据到集合")

    async def _load_entities(self, entity_type: str) -> List[Dict[str, Any]]:
        """
        从数据库加载指定类型的实体

        Args:
            entity_type: 实体类型 (project/property/target)

        Returns:
            实体文档列表，每个文档包含text和其他元数据
        """
        try:
            # 根据实体类型构建查询
            if entity_type == "project":
                query = """
                    SELECT 
                        precinct_id as id,
                        house_name as name,
                        house_full_name as full_name,
                        house_type as type
                    FROM `newsee-owner`.owner_house_base_info
                    WHERE house_type = 2
                    AND is_block_up = 0
                """
            elif entity_type == "property":
                query = """
                    SELECT 
                        house_id as id,
                        house_full_name as name,
                        precinct_id as project_id,
                        house_type as type
                    FROM `newsee-owner`.owner_house_base_info
                    WHERE house_type = 6
                    AND is_block_up = 0
                """
            elif entity_type == "target":
                query = """
                    SELECT 
                        id,
                        targetItemName as name,
                        unit,
                        status
                    FROM `newsee-view`.target_targetitem
                    WHERE status = 1  
                """
            else:
                logger.warning(f"未知的实体类型: {entity_type}")
                return []

            # 执行查询
            rows = self.db.execute_query(query)
            if not rows:
                logger.warning(f"没有找到{entity_type}数据")
                return []

            # 转换为文档格式
            documents = []
            for row in rows:
                # 构建文档文本
                doc_text = self._build_document_text(entity_type, row)
                if not doc_text:
                    continue

                # 创建文档
                doc = {
                    "text": doc_text,
                    "id": str(row.get("id", "")),
                    "type": entity_type,
                    **{k: v for k, v in row.items() if v is not None and k != "id"},
                }
                documents.append(doc)

            logger.info(f"从数据库加载了 {len(documents)} 条{entity_type}记录")
            return documents

        except Exception as e:
            logger.error(f"加载{entity_type}数据失败: {e}", exc_info=True)
            return []

    def _build_document_text(self, entity_type: str, data: Dict[str, Any]) -> str:
        """
        根据实体类型和行数据构建文档文本

        Args:
            entity_type: 实体类型
            data: 行数据

        Returns:
            文档文本
        """
        if entity_type == "project":
            # 项目: 使用项目名称和全名
            return f"{data.get('name', '')} {data.get('full_name', '')}"
        elif entity_type == "property":
            # 房产: 使用房产全名和所属项目名称
            return f"{data.get('name', '')} {data.get('project_name', '')}"
        elif entity_type == "target":
            # 指标: 使用指标名称和单位
            return f"{data.get('name', '')} {data.get('unit', '')}"
        return ""

    def search(
        self, query: str, entity_type: str = None, top_k: int = None
    ) -> Dict[str, Any]:
        """
        搜索实体

        Args:
            query: 查询文本
            entity_type: 实体类型 (project/property/target)，为None时搜索所有类型
            top_k: 返回结果数量

        Returns:
            包含搜索结果和状态的字典
        """
        try:
            if top_k is None:
                top_k = self.config["default_top_k"]

            # 1. 尝试精确匹配
            exact_results = self._exact_match(query)
            if exact_results["found"]:
                logger.info(f"找到精确匹配: {exact_results}")
                return {
                    "success": True,
                    "found": True,
                    "type": "exact",
                    "results": exact_results["results"][:top_k],
                    "message": f"找到{len(exact_results['results'])}个精确匹配结果",
                }

            # 2. 尝试模糊匹配
            fuzzy_results = self._fuzzy_match(query, entity_type, top_k)
            if fuzzy_results["found"]:
                logger.info(f"找到模糊匹配: {fuzzy_results}")
                return {
                    "success": True,
                    "found": True,
                    "type": "fuzzy",
                    "results": fuzzy_results["results"][:top_k],
                    "message": f"找到{len(fuzzy_results['results'])}个模糊匹配结果",
                }

            # 3. 尝试向量搜索
            vector_results = self._vector_search(query, entity_type, top_k)
            if vector_results["found"]:
                logger.info(f"向量搜索结果: {vector_results}")
                return {
                    "success": True,
                    "found": True,
                    "type": "vector",
                    "results": vector_results["results"][:top_k],
                    "message": f"找到{len(vector_results['results'])}个相关结果",
                }

            # 未找到结果
            return {
                "success": True,
                "found": False,
                "results": [],
                "message": "未找到匹配结果",
            }

        except Exception as e:
            error_msg = f"搜索失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "success": False,
                "found": False,
                "results": [],
                "message": error_msg,
            }

    def _exact_match(self, query: str) -> Dict[str, Any]:
        """精确匹配"""
        results = []

        # 在所有集合中搜索
        for entity_type, collection in self.collections.items():
            # 获取所有文档
            items = collection.get()

            # 检查每个文档
            for i, doc_id in enumerate(items["ids"]):
                metadata = items["metadatas"][i]
                name = metadata.get("name", "")

                # 检查名称是否完全匹配
                if name and name in query:
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

    def _fuzzy_match(
        self, query: str, entity_type: str = None, top_k: int = 5
    ) -> Dict[str, Any]:
        """模糊匹配"""
        results = []

        # 确定要搜索的集合
        collections_to_search = (
            [entity_type] if entity_type else self.collections.keys()
        )

        for entity_type in collections_to_search:
            if entity_type not in self.collections:
                continue

            collection = self.collections[entity_type]
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
                similarity = self._calculate_similarity(query, name)
                if similarity >= self.fuzzy_match_threshold:
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

    def _vector_search(
        self, query: str, entity_type: str = None, top_k: int = 5
    ) -> Dict[str, Any]:
        """向量搜索"""
        results = []

        # 确定要搜索的集合
        collections_to_search = (
            [entity_type] if entity_type else self.collections.keys()
        )

        for entity_type in collections_to_search:
            if entity_type not in self.collections:
                continue

            collection = self.collections[entity_type]

            try:
                # 执行向量搜索
                search_results = collection.query(query_texts=[query], n_results=top_k)

                # 处理结果
                if search_results["ids"] and search_results["ids"][0]:
                    for i, doc_id in enumerate(search_results["ids"][0]):
                        metadata = search_results["metadatas"][0][i]
                        distance = search_results["distances"][0][i]
                        # 将距离转换为相似度 (0-1之间，1表示最相似)
                        similarity = 1.0 - min(1.0, distance / 2.0)

                        if similarity >= self.vector_search_threshold:
                            results.append(
                                {
                                    "id": metadata.get("id"),
                                    "name": metadata.get("name", ""),
                                    "type": entity_type,
                                    "similarity": round(similarity, 2),
                                    "match_type": "vector",
                                    "metadata": metadata,
                                }
                            )
            except Exception as e:
                logger.error(f"向量搜索出错 ({entity_type}): {e}")

        # 按相似度排序
        results.sort(key=lambda x: x["similarity"], reverse=True)

        return {"found": len(results) > 0, "results": results[:top_k]}

    def _calculate_similarity(self, s1: str, s2: str) -> float:
        """计算两个字符串的相似度（基于编辑距离）"""
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 1.0
        return 1 - distance(s1, s2) / max_len

    def close(self):
        """释放资源"""
        if hasattr(self, "db") and self.db:
            self.db.close()
        logger.info("实体查找器已关闭")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# 使用示例
async def example_usage():
    # 配置数据库连接
    db_config = {
        "host": "localhost",
        "port": 3306,
        "user": "your_username",
        "password": "your_password",
        "database": "your_database",
    }

    # 创建实体查找器实例
    finder = EntityFinderMySQL(data_dir="./data")

    try:
        # 初始化并加载数据
        await finder.initialize(db_config)

        # 执行搜索
        query = "搜索关键词"
        results = finder.search(query, top_k=3)

        # 输出结果
        if results["found"]:
            print(f"找到 {len(results['results'])} 个结果:")
            for i, result in enumerate(results["results"], 1):
                print(
                    f"{i}. {result['name']} ({result['type']}) - 相似度: {result['similarity']}"
                )
        else:
            print("未找到匹配结果")

    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        finder.close()


if __name__ == "__main__":
    asyncio.run(example_usage())
