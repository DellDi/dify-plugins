import asyncio
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.absolute()))

from provider.entity_finder_mysql import EntityFinderMySQL
from provider.database import DatabaseConnection, create_db_url

async def test_plugin():
    """测试Dify插件的功能"""
    # 数据库配置 - 请替换为您的实际数据库信息
    db_config = {
        "host": "192.168.1.52",
        "port": 3306,
        "user": "root",
        "password": "Newsee888",
        "database": "newsee-view",
    }

    # 初始化实体查找器
    finder = EntityFinderMySQL(data_dir="./test_data")

    try:
        # 初始化数据库连接
        print("正在初始化数据库连接...")
        await finder.initialize(db_config)
        print("数据库连接初始化成功")

        # 测试搜索功能
        test_queries = ["金色蓝庭", "龙游", "实收"]

        for query in test_queries:
            print(f"\n测试搜索: {query}")
            results = finder.search(query, top_k=3)

            if results["found"]:
                print(f"找到 {len(results['results'])} 个结果:")
                for i, result in enumerate(results["results"][:3], 1):
                    print(f"  {i}. {result['name']} ({result['type']}) - 相似度: {result['similarity']:.2f}")
            else:
                print("未找到匹配结果")

    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        finder.close()
        print("\n测试完成，已清理资源")

if __name__ == "__main__":
    asyncio.run(test_plugin())
