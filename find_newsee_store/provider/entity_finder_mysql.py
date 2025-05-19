import logging
from typing import Dict, List, Any
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
import asyncio
import logging

# 导入搜索工具
from utils.search import ExactMatcher, FuzzyMatcher, VectorSearcher, KeywordExtractor
import jieba
import jieba.analyse

# 导入数据库连接
from .database import DatabaseConnection, create_db_url

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 默认配置
DEFAULT_CONFIG = {
    "fuzzy_match_threshold": 0.6,  # 模糊匹配阈值 - 降低以捕获更多结果
    "vector_search_threshold": 0.3,  # 向量搜索阈值 - 降低以捕获更多结果
    "top_k": 3,  # 默认返回结果数量
    "enable_fuzzy": True,  # 是否启用模糊匹配
    "enable_vector_search": True,  # 是否启用向量搜索
    "default_top_k": 3,  # 默认返回结果数量
}


class EntityFinderMySQL:
    """实体查找器，用于从 MySQL 数据库中加载数据并进行实体查找"""

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
        # 使用内存模式初始化ChromaDB，避免持久化存储问题
        try:
            # 使用内存模式
            self.client = chromadb.Client()
            logger.info("使用内存模式初始化ChromaDB客户端")
        except Exception as e:
            logger.error(f"ChromaDB初始化失败: {e}")
            raise

        # 初始化搜索器
        self.exact_matcher = ExactMatcher()
        self.fuzzy_matcher = FuzzyMatcher(threshold=DEFAULT_CONFIG["fuzzy_match_threshold"])
        self.vector_searcher = VectorSearcher(threshold=DEFAULT_CONFIG["vector_search_threshold"])
        self.keyword_extractor = KeywordExtractor()

        # 初始化集合
        self.collections = {
            "project": self.client.get_or_create_collection(
                name="projects",
                embedding_function=self.embedding_function,
                metadata={"description": "项目信息"},
            ),
            "org": self.client.get_or_create_collection(
                name="orgs",
                embedding_function=self.embedding_function,
                metadata={"description": "组织信息"},
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

        # 加载组织数据
        orgs = await self._load_entities("org")
        if orgs:
            self._add_documents_to_collection("org", orgs)

        # 加载指标数据
        targets = await self._load_entities("target")
        if targets:
            self._add_documents_to_collection("target", targets)

        logger.info("数据加载完成")

    def _add_documents_to_collection(self, entity_type: str, documents: List[Dict]):
        """将文档添加到对应的集合中"""
        if not documents:
            logger.info(f"没有{entity_type}数据需要添加")
            return

        if entity_type not in self.collections:
            logger.warning(f"集合{entity_type}不存在，尝试创建...")
            try:
                self.collections[entity_type] = self.client.create_collection(
                    name=f"{entity_type}s",
                    embedding_function=self.embedding_function,
                    metadata={"description": f"{entity_type}信息"}
                )
            except Exception as e:
                logger.error(f"创建{entity_type}集合失败: {e}")
                return

        collection = self.collections[entity_type]

        try:
            # 清空现有数据
            try:
                collection.delete(where={"id": {"$ne": ""}})
                logger.info(f"已清空{entity_type}集合中的现有数据")
            except Exception as e:
                logger.warning(f"清空{entity_type}集合数据失败，将继续添加新数据: {e}")

            # 批量大小
            batch_size = 5000
            total_docs = len(documents)
            logger.info(
                f"开始添加{total_docs}条{entity_type}数据到集合，批量大小: {batch_size}"
            )

            # 分批处理
            success_count = 0
            for start_idx in range(0, total_docs, batch_size):
                try:
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
                    success_count += len(batch_docs)
                    logger.info(
                        f"已添加批次 {start_idx//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size}: "
                        f"{start_idx+1}-{end_idx} 条{entity_type}数据"
                    )
                except Exception as e:
                    logger.error(f"添加{entity_type}数据批次{start_idx//batch_size + 1}失败: {e}")

            logger.info(f"完成添加{success_count}/{total_docs}条{entity_type}数据到集合")
        except Exception as e:
            logger.error(f"添加{entity_type}数据到集合时发生错误: {e}")
            # 不抛出异常，确保我们仍然可以继续处理其他实体类型

    async def _load_entities(self, entity_type: str) -> List[Dict[str, Any]]:
        """
        从数据库加载指定类型的实体

        Args:
            entity_type: 实体类型 (project/org/target)

        Returns:
            实体文档列表，每个文档包含text和其他元数据
        """
        try:
            # 根据实体类型构建查询
            if entity_type == "project":
                query = """
                    SELECT
                        house_id as id,
                        pro_short_name as name
                    FROM `newsee-owner`.owner_house_precinct_info
                """
            elif entity_type == "org":
                query = """
                    SELECT
                        organization_id as id,
                        organization_name as name,
                        organization_short_name as short_name,
                        CASE organization_type
                            WHEN 0 THEN '集团'
                            WHEN 1 THEN '公司'
                            WHEN 2 THEN '部门'
                            ELSE '未知类型'
                        END AS organization_type_cn
                    FROM `newsee-system`.ns_system_organization
                    WHERE is_deleted = 0 and organization_enablestate not in (1, 3)
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
        elif entity_type == "org":
            # 组织: 使用组织
            return f"{data.get('name', '')} {data.get('short_name', '')}"
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
            entity_type: 实体类型 (project/org/target)，为None时搜索所有类型
            top_k: 返回结果数量

        Returns:
            包含搜索结果和状态的字典
        """
        try:
            if top_k is None:
                top_k = self.config["default_top_k"]

            logger.info(f"开始搜索: 查询='{query}', 实体类型={entity_type}, top_k={top_k}")

            # 1. 尝试精确匹配
            exact_results = self.exact_matcher.search(query, self.collections, entity_type)
            if exact_results["found"]:
                logger.info(f"找到精确匹配: {len(exact_results['results'])}个结果")
                return {
                    "success": True,
                    "found": True,
                    "type": "exact",
                    "results": exact_results["results"][:top_k],
                    "message": f"找到{len(exact_results['results'])}个精确匹配结果",
                }
            logger.info(f"未找到精确匹配结果: 查询='{query}', 实体类型={entity_type}")
            # 2. 尝试模糊匹配
            fuzzy_results = self.fuzzy_matcher.search(query, self.collections, entity_type, top_k)
            if fuzzy_results["found"]:
                logger.info(f"找到模糊匹配: {len(fuzzy_results['results'])}个结果")
                return {
                    "success": True,
                    "found": True,
                    "type": "fuzzy",
                    "results": fuzzy_results["results"][:top_k],
                    "message": f"找到{len(fuzzy_results['results'])}个模糊匹配结果",
                }
            logger.info(f"未找到模糊匹配结果: 查询='{query}', 实体类型={entity_type}")
            # 3. 尝试向量搜索
            vector_results = self.vector_searcher.search(query, self.collections, entity_type, top_k)
            if vector_results["found"]:
                logger.info(f"找到向量搜索结果: {len(vector_results['results'])}个结果")
                return {
                    "success": True,
                    "found": True,
                    "type": "vector",
                    "results": vector_results["results"][:top_k],
                    "message": f"找到{len(vector_results['results'])}个相关结果",
                }
            # 未找到结果
            logger.info(f"未找到匹配结果: 查询='{query}', 实体类型={entity_type}")
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

    def close(self):
        """释放资源"""
        if hasattr(self, "db") and self.db:
            self.db.close()
        logger.info("实体查找器已关闭")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
