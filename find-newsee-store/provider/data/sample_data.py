"""
示例数据配置

此文件包含实体查找器使用的示例数据。
在实际应用中，这些数据应该从数据库或其他数据源动态加载。
"""

# 示例指标数据
SAMPLE_TARGETS = [
    {"id": "T001", "name": "销售额"},
    {"id": "T002", "name": "净利润"},
    {"id": "T003", "name": "毛利率"},
    {"id": "T004", "name": "毛利率"},
    {"id": "T005", "name": "毛利率"},
]


# 示例项目数据
SAMPLE_PROJECTS = [
    {"id": "P001", "name": "星河湾", "location": "北京朝阳区"},
    {"id": "P002", "name": "翡翠城", "location": "上海浦东新区"},
    {"id": "P003", "name": "金色家园", "location": "广州天河区"},
    {"id": "P004", "name": "阳光小区", "location": "深圳南山区"},
    {"id": "P005", "name": "绿地国际花都", "location": "杭州西湖区"},
]

# 示例房产数据
SAMPLE_PROPERTIES = [
    {"id": "R001", "name": "星河湾1号楼", "project_id": "P001", "rooms": 3},
    {"id": "R002", "name": "星河湾2号楼", "project_id": "P001", "rooms": 4},
    {"id": "R003", "name": "翡翠城A区", "project_id": "P002", "rooms": 2},
    {"id": "R004", "name": "金色家园1期", "project_id": "P003", "rooms": 3},
    {"id": "R005", "name": "阳光小区B栋", "project_id": "P004", "rooms": 2},
]

# 停用词列表
STOP_WORDS = {
    "的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一", "一个",
    "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这"
}

# 默认配置
DEFAULT_CONFIG = {
    "fuzzy_match_threshold": 0.8,  # 模糊匹配阈值
    "vector_search_threshold": 0.6,  # 向量搜索阈值
    "top_k": 3,  # 默认返回结果数量
    "enable_fuzzy": True,  # 是否启用模糊匹配
    "enable_vector_search": True,  # 是否启用向量搜索
}
