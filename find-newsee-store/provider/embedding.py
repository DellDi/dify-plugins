from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import jieba
import logging
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

class TextEmbedder:
    """文本嵌入和向量化处理"""
    
    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        """
        初始化文本嵌入器
        
        Args:
            model_name: 预训练模型名称
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        logger.info(f"加载嵌入模型: {model_name}")
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        将文本列表转换为向量
        
        Args:
            texts: 文本列表
            
        Returns:
            文本向量数组，形状为 (n_texts, embedding_dim)
        """
        return self.model.encode(texts, convert_to_numpy=True)
    
    def tokenize_chinese(self, text: str) -> str:
        """
        中文分词
        
        Args:
            text: 输入文本
            
        Returns:
            分词后的文本，用空格分隔
        """
        return ' '.join(jieba.cut(text))


class VectorStore:
    """向量存储和检索"""
    
    def __init__(self, persist_directory: str = "./data/chroma"):
        """
        初始化向量存储
        
        Args:
            persist_directory: 持久化存储目录
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # 初始化ChromaDB客户端
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        # 初始化集合
        self.collections = {}
        for entity_type in ['project', 'property', 'target']:
            self.collections[entity_type] = self.client.get_or_create_collection(
                name=entity_type,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
        
        logger.info(f"向量存储初始化完成，目录: {self.persist_directory}")
    
    def add_documents(self, entity_type: str, documents: List[Dict[str, Any]]):
        """
        添加文档到向量存储
        
        Args:
            entity_type: 实体类型 (project/property/target)
            documents: 文档列表，每个文档是包含id和text的字典
        """
        if entity_type not in self.collections:
            raise ValueError(f"不支持的实体类型: {entity_type}")
        
        collection = self.collections[entity_type]
        
        # 准备数据
        ids = []
        texts = []
        metadatas = []
        
        for doc in documents:
            doc_id = doc.get('id')
            if not doc_id:
                logger.warning(f"文档缺少id: {doc}")
                continue
                
            text = doc.get('text')
            if not text:
                logger.warning(f"文档 {doc_id} 缺少文本内容")
                continue
                
            ids.append(str(doc_id))
            texts.append(text)
            metadatas.append({k: str(v) for k, v in doc.items() if k != 'text'})
        
        # 添加到集合
        if ids:
            collection.upsert(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"成功添加 {len(ids)} 个 {entity_type} 文档到向量存储")
    
    def search(self, query: str, entity_type: str = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        相似性搜索
        
        Args:
            query: 查询文本
            entity_type: 实体类型，为None时搜索所有类型
            top_k: 返回结果数量
            
        Returns:
            匹配的文档列表，按相似度降序排列
        """
        results = []
        
        # 确定要搜索的集合
        collections_to_search = []
        if entity_type and entity_type in self.collections:
            collections_to_search.append((entity_type, self.collections[entity_type]))
        elif not entity_type:
            collections_to_search = list(self.collections.items())
        else:
            raise ValueError(f"不支持的实体类型: {entity_type}")
        
        # 对每个集合执行搜索
        for etype, collection in collections_to_search:
            try:
                search_results = collection.query(
                    query_texts=[query],
                    n_results=top_k
                )
                
                # 处理结果
                for i in range(len(search_results['ids'][0])):
                    doc_id = search_results['ids'][0][i]
                    score = 1.0 - search_results['distances'][0][i]  # 转换为相似度分数
                    metadata = search_results['metadatas'][0][i]
                    
                    results.append({
                        'id': doc_id,
                        'type': etype,
                        'score': float(score),  # 转换为Python原生float类型
                        **metadata
                    })
                
            except Exception as e:
                logger.error(f"搜索 {etype} 时出错: {e}")
        
        # 按分数排序并限制返回数量
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def clear_collection(self, entity_type: str):
        """清空指定类型的集合"""
        if entity_type in self.collections:
            self.collections[entity_type].delete()
            logger.info(f"已清空 {entity_type} 集合")