import asyncio
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.absolute()))

from provider.entity_finder_mysql import EntityFinderMySQL

async def test_plugin():
    """测试Dify插件的功能"""

    # 数据库配置 - 请替换为您的实际数据库信息
    db_config = {
        "host": "localhost",
        "port": 3306,
        "user": "root",
        "password": "zd808611",
        "database": "newsee-view",
    }

    # 初始化实体查找器
    finder = EntityFinderMySQL(data_dir="./test_data")

    try:
        # 初始化数据库连接
        print("正在初始化数据库连接...")
        # await finder.initialize(db_config)
        print("数据库连接初始化成功")

        # 测试搜索功能
        # 1. 精确匹配测试案例
        exact_queries = [
            ("金色佳园", "project"),
            ("鹿港大厦服务中心", "org"),
            ("物业费实收", "target"),
            ("阳光100国际新城", "project"),
            ("万达广场物业管理处", "org"),
            ("水费收缴率", "target"),
        ]

        # 2. 错别字测试案例
        typo_queries = [
            # 项目名称错别字
            ("金色佳苑", "project"),  # "园"错写成"苑"
            ("金色家园", "project"),  # "佳"错写成"家"
            ("金色家园", "project"),  # 全拼错误
            # 组织名称错别字
            ("鹿港大夏服务中心", "org"),  # "厦"错写成"夏"
            ("鹿港大厦服务中兴", "org"),  # "心"错写成"兴"
            ("鹿港大厦服务中芯", "org"),  # 多字错误
            # 指标名称错别字
            ("物业费市收", "target"),  # "实"错写成"市"
            ("物业费实受", "target"),  # "收"错写成"受"
            ("物业费实收率", "target"),  # 多字
        ]

        # 3. 短句和模糊匹配测试案例
        short_queries = [
            # 项目相关
            ("金佳", "project"),  # 部分名称
            ("阳光100", "project"),  # 部分名称（带数字）
            ("国际新城", "project"),  # 部分名称（后半部分）
            # 组织相关
            ("鹿港", "org"),  # 部分名称
            ("物业", "org"),  # 通用词
            ("服务中心", "org"),  # 通用词
            # 指标相关
            ("实收", "target"),  # 部分名称
            ("收缴", "target"),  # 近义词
            ("费用", "target"),  # 通用词
        ]

        # 4. 长句和自然语言测试案例
        long_queries = [
            # 项目相关
            ("我想查询一下金佳园小区的相关信息", "project"),
            # 组织相关
            ("请帮我找一下鹿港大厦服务点的工作时间", "org"),
            # 指标相关
            ("我需要了解一下关于物业费收取的统计数据", "target"),
            ("上个月的水电费收缴情况如何", "target"),
            ("今年的物业费实收率达到了多少", "target"),
        ]


        # 按类别测试并输出结果
        def run_test_group(queries, group_name):
            print(f"\n{'-'*20} {group_name} {'-'*20}")
            success_count = 0
            total_count = len(queries)

            for (query, entity_type) in queries:
                print(f"\n测试搜索: {query} (类型: {entity_type})")
                results = finder.search(query, entity_type=entity_type, top_k=3)

                if results["found"]:
                    success_count += 1
                    print(f"找到 {len(results['results'])} 个结果:")
                    for i, result in enumerate(results["results"][:3], 1):
                        print(f"  {i}. {result['name']} ({result['type']}) - 相似度: {result['similarity']:.2f} - 匹配类型: {result['match_type']}")
                else:
                    print("未找到匹配结果")

            success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
            print(f"\n{group_name}测试结果: {success_count}/{total_count} 成功率: {success_rate:.1f}%")
            return success_count, total_count

        # 按类别执行测试
        print("\n开始执行实体查找测试...")

        # 执行各类测试
        test_results = []
        test_results.append(run_test_group(exact_queries, "精确匹配测试"))
        test_results.append(run_test_group(typo_queries, "错别字测试"))
        test_results.append(run_test_group(short_queries, "短句测试"))
        test_results.append(run_test_group(long_queries, "长句测试"))

        # 计算总体统计信息
        total_success = sum(success for success, _ in test_results)
        total_tests = sum(total for _, total in test_results)
        overall_success_rate = (total_success / total_tests) * 100 if total_tests > 0 else 0

        # 输出测试报告
        print(f"\n{'='*20} 测试总结报告 {'='*20}")
        print(f"{'测试类型':<15} | {'通过数':<8} | {'总数':<8} | {'成功率':<8}")
        print("-" * 50)

        test_types = ["精确匹配", "错别字", "短句", "长句"]
        for (success, total), test_type in zip(test_results, test_types):
            success_rate = (success / total) * 100 if total > 0 else 0
            print(f"{test_type:<15} | {success:<8} | {total:<8} | {success_rate:>7.1f}%")

        print("-" * 50)
        print(f"{'总计':<15} | {total_success:<8} | {total_tests:<8} | {overall_success_rate:>7.1f}%")
        print("=" * 50)

        # 输出测试建议
        print("\n测试建议：")
        if overall_success_rate < 80:
            print("⚠️  整体成功率较低，建议检查匹配算法和阈值设置")
        if test_results[1][0] / test_results[1][1] < 0.7:
            print("⚠️  错别字识别率较低，建议优化模糊匹配算法")
        if test_results[2][0] / test_results[2][1] < 0.8:
            print("⚠️  短句匹配效果不理想，建议改进部分匹配逻辑")
        if test_results[3][0] / test_results[3][1] < 0.7:
            print("⚠️  长句理解能力有限，建议增强自然语言处理能力")

        print("\n提示：可以通过调整相似度阈值和优化匹配算法来提升搜索效果")

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
