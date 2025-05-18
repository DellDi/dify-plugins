import logging
from typing import Dict, List, Any
from pathlib import Path
from Levenshtein import distance
import chromadb
from chromadb.utils import embedding_functions
import asyncio
import math
import logging
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
            return

        collection = self.collections[entity_type]

        # 清空现有数据
        collection.delete(where={"id": {"$ne": ""}})

        # 批量大小
        batch_size = 5000
        total_docs = len(documents)
        logger.info(
            f"开始添加{total_docs}条{entity_type}数据到集合，批量大小: {batch_size}"
        )

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
            logger.info(
                f"已添加批次 {start_idx//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size}: "
                f"{start_idx+1}-{end_idx} 条{entity_type}数据"
            )

        logger.info(f"完成添加{total_docs}条{entity_type}数据到集合")

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

            # 1. 尝试精确匹配，传递实体类型进行过滤
            exact_results = self._exact_match(query, entity_type)
            if exact_results["found"]:
                logger.info(f"找到精确匹配: {len(exact_results['results'])}个结果")
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
                logger.info(f"找到模糊匹配: {len(fuzzy_results['results'])}个结果")
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

    def _exact_match(self, query: str, entity_type: str = None) -> Dict[str, Any]:
        """精确匹配
        
        Args:
            query: 查询文本
            entity_type: 实体类型，如果指定则只搜索该类型
            
        Returns:
            包含搜索结果的字典
        """
        results = []
        query_lower = query.lower().strip()
        
        # 确定要搜索的集合
        collections_to_search = {}
        if entity_type and entity_type in self.collections:
            # 如果指定了实体类型，只搜索该类型
            collections_to_search[entity_type] = self.collections[entity_type]
        else:
            # 否则搜索所有类型
            collections_to_search = self.collections

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
                # 对于精确匹配，我们有两种策略：
                # 1. 严格精确匹配：name_lower == query_lower
                # 2. 包含匹配：将包含关系移至模糊匹配中处理
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

    def _fuzzy_match(
        self, query: str, entity_type: str = None, top_k: int = 5
    ) -> Dict[str, Any]:
        """模糊匹配"""
        results = []
        query_lower = query.lower().strip()

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
                    similarity1 = self._calculate_similarity(query_lower, name_lower)
                    similarity2 = self._calculate_similarity(name_lower, query_lower)
                    similarity = max(similarity1, similarity2)

                # 对于短文本，降低阈值
                threshold = self.fuzzy_match_threshold
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

    def _extract_keywords(self, text: str) -> str:
        """使用jieba分词库从长文本中提取关键词和实体名称"""
        # 常见的中文停用词
        stop_words = [
            '我', '想', '查询', '一下', '帮我', '找', '一下', '关于', 
            '的', '了', '请', '需要', '如何', '怎么样', '是', '吗', '吗？', 
            '呢', '呢？', '吗', '吗？', '吗', '吗？', '吗', '吗？',
            '个', '和', '有', '不', '在', '也', '为', '么', '到', '得', '这', '那',
            '都', '而', '之', '已', '与', '还', '就', '可', '但', '却', '使', '由',
            '于', '所', '以', '都', '就', '很', '很多', '这个', '那个'
        ]
        
        # 1. 使用jieba分词
        seg_list = jieba.cut(text)
        words = [w for w in seg_list if w not in stop_words and len(w) > 1]
        
        # 2. 使用jieba的TF-IDF算法提取关键词
        # 对于短文本，返回最多5个关键词
        keywords = jieba.analyse.extract_tags(text, topK=5, withWeight=False)
        
        # 3. 使用jieba的TextRank算法提取关键词
        # TextRank算法更适合提取长文本中的关键词
        textrank_keywords = jieba.analyse.textrank(text, topK=3, withWeight=False)
        
        # 4. 提取可能的实体名称
        # 尝试提取连续的名词短语（使用jieba的词性标注功能）
        import jieba.posseg as pseg
        words_with_pos = pseg.cut(text)
        entity_candidates = []
        
        # 提取名词、地名和机构名称
        for word, flag in words_with_pos:
            # n表示名词，ns表示地名，nt表示机构名称
            if flag.startswith('n') and len(word) >= 2 and word not in stop_words:
                entity_candidates.append(word)
        
        # 5. 组合所有提取的关键词和实体
        all_keywords = list(set(words + keywords + textrank_keywords + entity_candidates))
        
        # 如果提取到的关键词过多，只保留前10个
        if len(all_keywords) > 10:
            all_keywords = all_keywords[:10]
        
        # 将关键词组合成一个字符串
        combined_text = ' '.join(all_keywords)
        
        # 记录提取的关键词，便于调试
        logger.debug(f"关键词提取: 原文='{text}', 关键词='{combined_text}'")
        
        return combined_text if combined_text else text  # 如果提取失败，返回原文本

    def _vector_search(
        self, query: str, entity_type: str = None, top_k: int = 5
    ) -> Dict[str, Any]:
        """向量搜索"""
        results = []
        query = query.strip()
        
        # 对长句进行关键词提取
        original_query = query
        if len(query) > 15:  # 对长句进行关键词提取
            query_for_search = self._extract_keywords(query)
            logger.info(f"长句关键词提取: '{original_query}' -> '{query_for_search}'")
        else:
            query_for_search = query

        # 确定要搜索的集合
        collections_to_search = (
            [entity_type] if entity_type else self.collections.keys()
        )

        for entity_type in collections_to_search:
            if entity_type not in self.collections:
                continue

            collection = self.collections[entity_type]

            try:
                # 执行向量搜索 - 增加结果数量以提高召回率
                search_results = collection.query(
                    query_texts=[query_for_search], 
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
                        threshold = self.vector_search_threshold
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

    def _calculate_similarity(self, s1: str, s2: str) -> float:
        """计算两个字符串的相似度（基于编辑距离）"""
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
