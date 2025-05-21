import logging
from typing import Dict, List, Any
from pathlib import Path
import chromadb
import logging

# 导入搜索工具
from utils.search import ExactMatcher, FuzzyMatcher, VectorSearcher, KeywordExtractor

# 导入ChromaDB组件
from chromadb.utils import embedding_functions

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
    "batch_size": 25,  # 批量大小
}


class EntityFinderMySQL:
    """实体查找器，用于从 MySQL 数据库中加载数据并进行实体查找
    Args:
        data_dir: 数据存储目录
        embedding_api_key: 嵌入API密钥
    """

    def __init__(self, data_dir: str = "./data", embedding_api_key: str = None, reset_collections: bool = False):
        """
        初始化实体查找器

        Args:
            data_dir: 数据存储目录
            embedding_api_key: 嵌入API密钥
            reset_collections: 是否重置所有集合（当切换嵌入模型时需要设置为True）
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.reset_collections = reset_collections
        self.embedding_api_key = embedding_api_key
        
        # 确定使用的嵌入模型类型
        self.embedding_model_type = "tongyi" if embedding_api_key else "local"

        # 初始化组件
        self.db = None

        # 初始化ChromaDB客户端
        chroma_db_path = self.data_dir / "chroma_db"

        try:
            # 重置集合时，需要注意客户端初始化方式
            client_settings = {}
            
            # 配置非保护模式，确保不启动额外的进程
            client_settings = {
                "allow_reset": True,  # 允许重置
                "anonymized_telemetry": False  # 禁用遥测
            }
            
            # 强制使用可再生模式
            self.client = chromadb.PersistentClient(
                path=str(chroma_db_path),
                settings=chromadb.Settings(**client_settings)
            )
            logger.info("使用增强模式初始化ChromaDB客户端")
        except Exception as e:
            logger.error(f"ChromaDB初始化失败: {e}")
            raise

        # 设置嵌入函数
        if embedding_api_key:
            DEFAULT_CONFIG["batch_size"] = 25
            self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=embedding_api_key,
                api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
                api_type="tongyi",
                api_version="v2",
                model_name="text-embedding-v2",
            )
        else:
            DEFAULT_CONFIG["batch_size"] = 5000
            self.embedding_function = (
                embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="BAAI/bge-small-zh"  # 中文小型模型
                )
            )

        # 初始化搜索器
        self.exact_matcher = ExactMatcher()
        self.fuzzy_matcher = FuzzyMatcher(
            threshold=DEFAULT_CONFIG["fuzzy_match_threshold"]
        )
        self.vector_searcher = VectorSearcher(
            threshold=DEFAULT_CONFIG["vector_search_threshold"]
        )
        self.keyword_extractor = KeywordExtractor()

        # 初始化集合
        self.collections = {}
        
        # 如果需要重置集合，尝试清理已存在的全部集合
        if self.reset_collections:
            try:
                logger.info("正在重置所有集合...")
                try:
                    # 先获取所有集合列表
                    all_collections = self.client.list_collections()
                    
                    # 删除我们管理的集合
                    for collection_name in ["projects", "orgs", "targets"]:
                        try:
                            if collection_name in [c.name for c in all_collections]:
                                self.client.delete_collection(collection_name)
                                logger.info(f"已删除集合: {collection_name}")
                        except Exception as e:
                            logger.warning(f"删除集合 {collection_name} 时出错: {e}")
                            
                    # 也尝试删除任何孤立的集合（那些不在我们预期列表中的）
                    for collection in all_collections:
                        if collection.name not in ["projects", "orgs", "targets"]:
                            try:
                                self.client.delete_collection(collection.name)
                                logger.info(f"已删除孤立集合: {collection.name}")
                            except Exception as e:
                                logger.warning(f"删除孤立集合 {collection.name} 时出错: {e}")
                                
                    # 检查并清理UUID目录
                    self._clean_uuid_directories()
                except Exception as e:
                    logger.warning(f"清理集合失败: {e}")
            except Exception as e:
                logger.warning(f"重置集合过程中出错: {e}")
        
        # 创建或获取集合
        try:
            # 定义需要创建的集合及其配置
            collections_config = {
                "project": {"name": "projects", "description": "项目信息"},
                "org": {"name": "orgs", "description": "组织信息"},
                "target": {"name": "targets", "description": "指标信息"},
            }
            
            # 如果不需要重置集合，可以尝试直接使用现有的
            for entity_type, config in collections_config.items():
                collection_name = config["name"]
                try:
                    # 尝试获取现有集合
                    if not self.reset_collections and collection_name in [c.name for c in self.client.list_collections()]:
                        self.collections[entity_type] = self.client.get_collection(
                            name=collection_name,
                            embedding_function=self.embedding_function
                        )
                        logger.info(f"获取到现有集合: {collection_name}")
                    else:
                        # 如果不存在或需要重置，则创建新的
                        self.collections[entity_type] = self.client.get_or_create_collection(
                            name=collection_name,
                            embedding_function=self.embedding_function,
                            metadata={"description": config["description"]},
                        )
                        logger.info(f"创建或获取集合: {collection_name}")
                except Exception as e:
                    logger.error(f"初始化集合 {collection_name} 时出错: {e}")
                    raise
                    
            logger.info("所有集合已成功初始化")  
        except Exception as e:
            logger.error(f"初始化集合时出错: {e}")
            raise

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
        """将文档添加到对应的集合中、使用upsert模式"""
        if not documents:
            logger.info(f"没有{entity_type}数据需要添加")
            return

        collection_name = f"{entity_type}s"

        # 如果集合不存在，则获取或创建
        if entity_type not in self.collections:
            logger.info(f"集合 {collection_name} 引用不存在，获取或创建")
            try:
                # 使用跟其他地方相同的逻辑获取或创建集合
                if not self.reset_collections and collection_name in [c.name for c in self.client.list_collections()]:
                    self.collections[entity_type] = self.client.get_collection(
                        name=collection_name,
                        embedding_function=self.embedding_function
                    )
                    logger.info(f"使UPSERT模式获取到现有集合: {collection_name}")
                else:
                    # 如果需要重置，则先删除再创建
                    if collection_name in [c.name for c in self.client.list_collections()]:
                        self.client.delete_collection(collection_name)
                        logger.info(f"已删除现有集合{collection_name}以重新创建")
                    
                    # 创建新集合
                    self.collections[entity_type] = self.client.create_collection(
                        name=collection_name,
                        embedding_function=self.embedding_function,
                        metadata={"description": f"{entity_type}信息"}
                    )
                    logger.info(f"成功创建新集合{collection_name}")
            except Exception as e:
                logger.error(f"获取或创建集合{collection_name}失败: {e}")
                return

        collection = self.collections[entity_type]

        try:
            # 检查当前数据量
            try:
                count_result = collection.count()
                logger.info(f"当前{entity_type}集合数据量: {count_result}")
            except Exception as e:
                logger.warning(f"获取{entity_type}集合数据量失败: {e}")
                
            # 采用upsert模式，不需要先删除旧数据

            # 批量大小
            batch_size = DEFAULT_CONFIG["batch_size"]
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
                    collection.upsert(documents=texts, metadatas=metadatas, ids=ids)
                    success_count += len(batch_docs)
                    logger.info(
                        f"已添加批次 {start_idx//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size}: "
                        f"{start_idx+1}-{end_idx} 条{entity_type}数据"
                    )
                except Exception as e:
                    logger.error(
                        f"添加{entity_type}数据批次{start_idx//batch_size + 1}失败: {e}"
                    )

            logger.info(
                f"完成添加{success_count}/{total_docs}条{entity_type}数据到集合"
            )
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
            # 项目: 使用项目名称
            return f"{data.get('name', '')}"
        elif entity_type == "org":
            # 组织: 使用组织
            return f"{data.get('name', '')}"
        elif entity_type == "target":
            # 指标: 使用指标名称
            return f"{data.get('name', '')}"
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

            logger.info(
                f"开始搜索: 查询='{query}', 实体类型={entity_type}, top_k={top_k}"
            )

            # 1. 尝试精确匹配
            exact_results = self.exact_matcher.search(
                query, self.collections, entity_type
            )
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
            fuzzy_results = self.fuzzy_matcher.search(
                query, self.collections, entity_type, top_k
            )
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
            vector_results = self.vector_searcher.search(
                query, self.collections, entity_type, top_k
            )
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
    def _clean_uuid_directories(self):
        """清理ChromaDB目录中的UUID格式子目录"""
        try:
            import os
            # 确定chroma_db目录路径
            chroma_dir = os.path.join(self.data_dir, "chroma_db")
            if not os.path.exists(chroma_dir):
                logger.info(f"ChromaDB目录 {chroma_dir} 不存在，跳过清理")
                return
                
            # 查找所有UUID格式的子目录（大部分的UUID目录都有特定的长度和格式）
            uuid_dirs = []
            for item in os.listdir(chroma_dir):
                item_path = os.path.join(chroma_dir, item)
                # 排除重要的非UUID文件
                if item in ['chroma.sqlite3', 'chroma.sqlite3-shm', 'chroma.sqlite3-wal', '.uuid']:
                    continue
                    
                # 如果是目录且看起来像UUID（建立一个简单的检测）
                if os.path.isdir(item_path) and (
                    len(item) > 30 and '-' in item or  # 典型的UUID形式
                    (len(item) > 10 and all(c in '0123456789abcdef-' for c in item.lower()))  # 十六进制字符
                ):
                    uuid_dirs.append(item_path)
                    
            if uuid_dirs:
                logger.info(f"发现 {len(uuid_dirs)} 个可能的UUID格式目录需要清理")
                
                # 首先关闭当前客户端，确保资源释放
                if hasattr(self, "client") and self.client:
                    # 清除引用但不完全释放所有资源
                    self.collections = {}
                    
                # 尝试删除这些UUID目录
                success_count = 0
                for dir_path in uuid_dirs:
                    try:
                        import shutil
                        shutil.rmtree(dir_path, ignore_errors=True)
                        success_count += 1
                    except Exception as e:
                        logger.warning(f"删除UUID目录 {dir_path} 失败: {e}")
                        
                logger.info(f"成功清理了 {success_count}/{len(uuid_dirs)} 个UUID目录")
            else:
                logger.info("没有发现需要清理的UUID目录")
                
        except Exception as e:
            logger.warning(f"UUID目录清理失败: {e}")

    def close_chroma(self):
        """释放ChromaDB资源"""
        # 先清理UUID目录以防止资源累积
        try:
            self._clean_uuid_directories()
        except Exception as e:
            logger.warning(f"UUID目录清理失败: {e}")
            
        if hasattr(self, "client") and self.client:
            try:
                # 先清除集合引用
                if hasattr(self, "collections"):
                    for name in list(self.collections.keys()):
                        self.collections[name] = None
                    self.collections.clear()
                
                # 尝试关闭底层SQLite连接
                # ChromaDB没有显式的close方法，我们需要更积极地释放资源
                # 清理持久化客户端资源
                if hasattr(self.client, "_producer"):
                    if hasattr(self.client._producer, "storage_context"):
                        if hasattr(self.client._producer.storage_context, "sqlite_connection"):
                            try:
                                self.client._producer.storage_context.sqlite_connection.close()
                                logger.info("已关闭ChromaDB的SQLite连接")
                            except Exception as e:
                                logger.warning(f"关闭SQLite连接时出错: {e}")
                
                # 将客户端引用设为None
                self.client = None
                logger.info("ChromaDB客户端引用已清除")
            except Exception as e:
                logger.warning(f"清除ChromaDB客户端引用时出错: {e}")
                
        # 强制触发Python垃圾回收
        import gc
        gc.collect()
        
        logger.info("实体查找器资源已全部释放")

    def close(self):
        """释放资源，确保所有连接都被正确关闭"""
        # 关闭数据库连接
        if hasattr(self, "db") and self.db:
            try:
                self.db.close()
                logger.info("数据库连接已关闭")
            except Exception as e:
                logger.warning(f"关闭数据库连接时出错: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
