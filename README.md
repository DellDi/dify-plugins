# Dify 插件集合

本仓库包含为 Dify 平台开发的各种插件，用于扩展 Dify 的功能和能力。

## 插件列表

### 1. 实体查找器 (find_newsee_store)

**功能**: 从用户输入的文本中识别并提取项目和房产实体，并返回对应的 ID 和相关信息。

**技术实现**:
- 使用 ChromaDB 作为向量数据库，支持本地部署
- 采用混合检索策略：精确匹配 + 向量检索
- 使用 jieba 分词处理中文文本
- 支持模糊匹配和语义相似度搜索

**主要特点**:
- 支持项目和房产两种实体类型的识别
- 返回结果包含实体 ID、名称、置信度和匹配类型
- 完全本地化部署，保证数据安全
- 提供数据同步接口，方便与企业数据库集成

**使用方法**:
```bash
cd find_newsee_store
pip install -r requirements.txt
python -m main
```

详细文档请参考 [find_newsee_store/README.md](./find_newsee_store/README.md)
