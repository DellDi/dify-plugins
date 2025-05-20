import asyncio
import sys
import time
from pathlib import Path
import logging
import colorlog

# 设置彩色日志

def setup_logger():
    """初始化彩色日志配置"""
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            fmt='%(log_color)s%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
            }
        )
    )

    logger = logging.getLogger('test_plugin')
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False
    return logger

logger = setup_logger()

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.absolute()))

from provider.entity_finder_mysql import EntityFinderMySQL

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
        # 修改后
        if not hasattr(finder, 'db') or not finder.db or not finder.db.is_connected():
            await finder.initialize(db_config)

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
            ("金佳园", "project"),  # 部分名称
            ("阳光100", "project"),  # 部分名称（带数字）
            ("国际新城", "project"),  # 部分名称（后半部分）
            # 组织相关
            ("鹿港服五", "org"),  # 部分名称
            ("广物业", "org"),  # 通用词
            ("德瑞服务中心", "org"),  # 通用词
            # 指标相关
            ("旧欠实收", "target"),  # 部分名称
            ("收缴滤", "target"),  # 近义词
            ("费用占比", "target"),  # 通用词
        ]

        # 4. 长句和自然语言测试案例
        long_queries = [
            # 项目相关
            ("我想查询一下金色佳园小区的相关信息", "project"),
            # 组织相关
            ("请帮我找一下鹿港大厦服务中心的工作时间", "org"),
            # 指标相关
            ("我需要了解一下关于物业费收取的统计数据", "target"),
            # 指标相关
            ("上个月的水电费物业费收缴情况如何", "target"),
            # 指标相关
            ("今年的物业费收缴率达到了多少", "target"),
        ]

        # 定义测试类型图标
        test_icons = {
            "精确匹配测试": "🎯",  # 目标
            "错别字测试": "📖",  # 书本
            "短句测试": "🔍",  # 放大镜
            "长句测试": "💬",  # 对话框
            "边界测试": "🔮"   # 水晶球
        }

        # 定义实体类型图标
        entity_icons = {
            "project": "🏘️",  # 建筑
            "org": "🏛️",     # 组织
            "target": "💰"      # 指标
        }

        # 定义匹配类型颜色和图标
        match_type_format = {
            "exact": ("[1;32m精确匹配[0m", "✅"),  # 绿色勾
            "fuzzy": ("[1;33m模糊匹配[0m", "⚠️"),  # 黄色感叹号
            "vector": ("[1;36m向量匹配[0m", "🧠")   # 蓝色大脑
        }

        # 按类别测试并输出结果
        def run_test_group(queries, group_name):
            icon = test_icons.get(group_name, "📊")
            logger.info(f"\n{'═'*20} {icon} {group_name} {'═'*20}")
            success_count = 0
            total_count = len(queries)
            start_time = time.time()

            for (query, entity_type) in queries:
                e_icon = entity_icons.get(entity_type, "")
                logger.info(f"\n{e_icon} 测试搜索: [1;34m{query}[0m (类型: [1;35m{entity_type}[0m)")
                results = finder.search(query, entity_type=entity_type, top_k=3)

                if results["found"]:
                    success_count += 1
                    match_type = results["results"][0]["match_type"]
                    color_type, type_icon = match_type_format.get(match_type, ("未知匹配", "?"))
                    logger.info(f"{type_icon} 找到 {len(results['results'])} 个结果 ({color_type}):")

                    for i, result in enumerate(results["results"][:3], 1):
                        # 根据相似度调整颜色
                        if result["similarity"] >= 0.9:
                            sim_color = "[1;32m"  # 绿色（高相似度）
                        elif result["similarity"] >= 0.7:
                            sim_color = "[1;33m"  # 黄色（中相似度）
                        else:
                            sim_color = "[1;31m"  # 红色（低相似度）

                        e_type_icon = entity_icons.get(result["type"], "")
                        logger.info(f"  {i}. [1;37m{result['name']}[0m ({e_type_icon} {result['type']}) - 相似度: {sim_color}{result['similarity']:.2f}[0m")
                else:
                    logger.warning(f"❌ 未找到匹配结果")

            # 计算成功率和耗时
            success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
            elapsed_time = time.time() - start_time

            if success_rate >= 80:
                rate_color = "[1;32m"  # 绿色（高成功率）
            elif success_rate >= 60:
                rate_color = "[1;33m"  # 黄色（中成功率）
            else:
                rate_color = "[1;31m"  # 红色（低成功率）

            logger.info(f"\n{icon} {group_name}测试结果: {success_count}/{total_count} 成功率: {rate_color}{success_rate:.1f}%[0m (耗时: {elapsed_time:.2f}秒)")
            return success_count, total_count

        # 按类别执行测试
        logger.info("\n🔔 开始执行实体查找测试...🔔")
        logger.info("✨ 测试环境已准备就绪，开始运行测试用例✨")

        # 执行各类测试
        test_results = []

        # 显示测试进度条
        total_test_groups = 5  # 包含边界测试
        logger.info(f"\n📊 测试进度: [{'='*0}{' '*(total_test_groups-0)}] 0/{total_test_groups}")

        # 执行测试并更新进度
        test_results.append(run_test_group(exact_queries, "精确匹配测试"))
        logger.info(f"\n📊 测试进度: [{'='*1}{' '*(total_test_groups-1)}] 1/{total_test_groups}")

        test_results.append(run_test_group(typo_queries, "错别字测试"))
        logger.info(f"\n📊 测试进度: [{'='*2}{' '*(total_test_groups-2)}] 2/{total_test_groups}")

        test_results.append(run_test_group(short_queries, "短句测试"))
        logger.info(f"\n📊 测试进度: [{'='*3}{' '*(total_test_groups-3)}] 3/{total_test_groups}")

        test_results.append(run_test_group(long_queries, "长句测试"))
        logger.info(f"\n📊 测试进度: [{'='*4}{' '*(total_test_groups-4)}] 4/{total_test_groups}")

        logger.info(f"\n📊 测试进度: [{'='*5}{' '*(total_test_groups-5)}] 5/{total_test_groups} ✅")

        # 计算总体统计信息
        total_success = sum(success for success, _ in test_results)
        total_tests = sum(total for _, total in test_results)
        overall_success_rate = (total_success / total_tests) * 100 if total_tests > 0 else 0

        # 输出测试报告
        logger.info(f"\n{'═'*20} 📈 测试总结报告 📈 {'═'*20}")

        # 创建表格头
        header = f"\n[1;37m{'测试类型':<15} | {'通过数':<10} | {'总数':<8} | {'成功率':<10} | {'评价':<8}[0m"
        logger.info(header)
        logger.info("─" * 65)

        # 定义评价图标
        rating_icons = ["💥", "👎", "👍", "🚀", "🌟"]

        test_types = ["精确匹配", "错别字", "短句", "长句", "边界测试"]
        for i, ((success, total), test_type) in enumerate(zip(test_results, test_types)):
            success_rate = (success / total) * 100 if total > 0 else 0

            # 根据成功率设置颜色和评价
            if success_rate >= 90:
                color = "[1;32m"  # 绿色
                rating = rating_icons[4]  # 最高评价
            elif success_rate >= 80:
                color = "[32m"  # 浅绿色
                rating = rating_icons[3]
            elif success_rate >= 70:
                color = "[1;33m"  # 黄色
                rating = rating_icons[2]
            elif success_rate >= 50:
                color = "[33m"  # 浅黄色
                rating = rating_icons[1]
            else:
                color = "[1;31m"  # 红色
                rating = rating_icons[0]

            # 添加测试类型图标
            icon = test_icons.get(f"{test_type}测试", "📋")

            # 输出格式化的结果行
            logger.info(f"{icon} {test_type:<12} | {success:<10} | {total:<8} | {color}{success_rate:>7.1f}%[0m | {rating}")

        # 输出总计行
        logger.info("─" * 65)

        # 设置总体成功率颜色
        if overall_success_rate >= 80:
            total_color = "[1;32m"  # 绿色
            total_rating = "🎉"  # 庆祝
        elif overall_success_rate >= 60:
            total_color = "[1;33m"  # 黄色
            total_rating = "👍"  # 赞
        else:
            total_color = "[1;31m"  # 红色
            total_rating = "⚠️"  # 警告

        logger.info(f"📊 总计        | {total_success:<10} | {total_tests:<8} | {total_color}{overall_success_rate:>7.1f}%[0m | {total_rating}")
        logger.info("═" * 65)

        # 输出测试建议
        logger.info("\n💡 测试分析与建议：")

        # 根据测试结果给出具体建议
        suggestions = []
        if overall_success_rate < 80:
            suggestions.append("⚠️  整体成功率较低，建议检查匹配算法和阈值设置")
        if test_results[1][0] / test_results[1][1] < 0.7:
            suggestions.append("⚠️  错别字识别率较低，建议优化模糊匹配算法")
        if test_results[2][0] / test_results[2][1] < 0.8:
            suggestions.append("⚠️  短句匹配效果不理想，建议改进部分匹配逻辑")
        if test_results[3][0] / test_results[3][1] < 0.7:
            suggestions.append("⚠️  长句理解能力有限，建议增强自然语言处理能力")

        # 如果没有具体问题，给出积极反馈
        if not suggestions:
            logger.info("🌟 测试结果令人满意！实体查找器在各类测试中表现良好。")
        else:
            for suggestion in suggestions:
                logger.info(suggestion)

        logger.info("\n🔸 优化建议：")
        logger.info("🔍 可以通过调整相似度阈值和优化匹配算法来提升搜索效果")
        logger.info("📃 建议添加更多测试案例，特别是真实用户查询场景")
        logger.info("💡 考虑集成高级的自然语言处理技术，提升长句理解能力")

    except Exception as e:
        logger.error(f"⛔️ 测试过程中发生错误: {e}")
        import traceback
        error_msg = traceback.format_exc()
        logger.error(f"[1;31m{error_msg}[0m")
    finally:
        # 清理资源
        finder.close()
        logger.info(f"\n{'═'*30}")
        logger.info("🔔 测试完成，所有资源已清理 ✅")
        logger.info(f"💾 测试数据已保存在 './test_data' 目录")
        logger.info(f"📈 测试报告生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'═'*30}")

if __name__ == "__main__":
    asyncio.run(test_plugin())
