"""
关键词提取模块
使用jieba分词库从文本中提取关键词和实体名称
"""

import logging
import jieba
import jieba.analyse
import jieba.posseg as pseg

logger = logging.getLogger(__name__)

class KeywordExtractor:
    """关键词提取器，使用jieba分词库从文本中提取关键词和实体名称"""

    def __init__(self):
        """初始化关键词提取器"""
        # 常见的中文停用词
        self.stop_words = [
            '我', '想', '查询','信息','相关','时间', '一下', '帮我', '找','小区','楼栋','单元','室','住宅','大厦','关于',
            '的', '了', '请', '需要', '如何', '怎么样', '是', '吗', '吗？',
            '呢', '呢？', '吗', '吗？', '吗', '吗？', '吗', '吗？',
            '个', '和', '有', '不', '在', '也', '为', '么', '到', '得', '这', '那',
            '都', '而', '之', '已', '与', '还', '就', '可', '但', '却', '使', '由',
            '于', '所', '以', '都', '就', '很', '很多', '这个', '那个'
        ]

    def extract(self, text: str, max_keywords: int = 10) -> list[str]:
        """
        从文本中提取关键词

        Args:
            text: 输入文本
            max_keywords: 最大关键词数量

        Returns:
            提取的关键词列表
        """
        if not text or len(text.strip()) == 0:
            return [text]

        # 1. 使用jieba分词
        seg_list = jieba.cut(text)
        words = [w for w in seg_list if w not in self.stop_words and len(w) > 1]

        # 2. 使用jieba的TF-IDF算法提取关键词
        # 对于短文本，返回最多5个关键词
        keywords = jieba.analyse.extract_tags(text, topK=5, withWeight=False)
        # 过滤停用词
        keywords = [w for w in keywords if w not in self.stop_words and len(w) > 1]

        # 3. 使用jieba的TextRank算法提取关键词
        # TextRank算法更适合提取长文本中的关键词
        textrank_keywords = jieba.analyse.textrank(text, topK=3, withWeight=False)
        # 过滤停用词
        textrank_keywords = [w for w in textrank_keywords if w not in self.stop_words and len(w) > 1]

        # 4. 提取可能的实体名称
        # 尝试提取连续的名词短语（使用jieba的词性标注功能）
        words_with_pos = pseg.cut(text)
        entity_candidates = []
        
        # 专门处理复合实体名称（如“金佳园”）
        compound_entities = []
        pos_words = list(words_with_pos)  # 转换为列表以便多次遍历
        
        # 尝试提取连续的名词组合
        for i in range(len(pos_words) - 1):
            word1, flag1 = pos_words[i]
            word2, flag2 = pos_words[i + 1]
            # 如果两个连续的词都是名词类型，尝试合并
            if (flag1.startswith('n') and flag2.startswith('n')) and \
               word1 not in self.stop_words and word2 not in self.stop_words and \
               len(word1 + word2) >= 2:
                compound_entities.append(word1 + word2)
        
        # 提取名词、地名和机构名称
        for word, flag in pos_words:
            # n表示名词，ns表示地名，nt表示机构名称
            if flag.startswith('n') and len(word) >= 2 and word not in self.stop_words:
                entity_candidates.append(word)

        # 5. 组合所有提取的关键词和实体
        # 特别处理：将复合实体放在最前面
        all_keywords = compound_entities + list(set(entity_candidates + keywords + textrank_keywords + words))
        
        # 去除停用词和重复项
        filtered_keywords = []
        seen = set()
        for keyword in all_keywords:
            if keyword not in self.stop_words and keyword not in seen and len(keyword) > 1:
                filtered_keywords.append(keyword)
                seen.add(keyword)

        # 6. 对关键词进行排序，按照实体名称、地点、动词的顺序
        # 重新对所有关键词进行词性标注
        keywords_with_pos = []
        for keyword in filtered_keywords:
            # 对每个关键词进行词性标注
            pos_tags = [(word, flag) for word, flag in pseg.cut(keyword)]
            if pos_tags:
                # 使用第一个词的词性作为关键词的词性
                main_pos = pos_tags[0][1]
                keywords_with_pos.append((keyword, main_pos))
            else:
                # 如果无法标注，则使用默认词性
                keywords_with_pos.append((keyword, 'x'))

        # 定义词性权重函数，实体名称权重高，动词权重低
        def get_pos_weight(pos):
            # nr人名、ns地名、nt机构名称优先级最高
            if pos.startswith('nr') or pos.startswith('ns') or pos.startswith('nt'):
                return 0
            # n名词优先级次之
            elif pos.startswith('n'):
                return 1
            # 地名和时间词优先级中等
            elif pos.startswith('t') or pos.startswith('s'):
                return 2
            # 动词和形容词优先级较低
            elif pos.startswith('v') or pos.startswith('a'):
                return 3
            # 其他词性优先级最低
            else:
                return 4

        # 按词性权重排序
        keywords_with_pos.sort(key=lambda x: get_pos_weight(x[1]))

        # 提取排序后的关键词
        sorted_keywords = [keyword for keyword, _ in keywords_with_pos]
        
        # 特别处理：确保复合实体始终在前面
        # 将复合实体从已排序关键词中移除，然后添加到列表头部
        for compound in compound_entities:
            if compound in sorted_keywords:
                sorted_keywords.remove(compound)
        
        # 将复合实体添加到列表头部
        sorted_keywords = compound_entities + sorted_keywords
        
        # 去除重复项
        final_keywords = []
        seen = set()
        for keyword in sorted_keywords:
            if keyword not in seen:
                final_keywords.append(keyword)
                seen.add(keyword)

        # 如果关键词过多，只保留前N个
        if len(final_keywords) > max_keywords:
            final_keywords = final_keywords[:max_keywords]

        # 如果没有提取到关键词，返回原文本
        if not final_keywords:
            final_keywords = [text]

        # 记录提取的关键词，便于调试
        logger.debug(f"关键词提取: 原文='{text}', 关键词='{final_keywords}'")

        return final_keywords
