import logging
import os
import sys
import pandas as pd

from inducing_factor.config.loader import load_config
from inducing_factor.core.attribution_system import EcoDisturbanceAttributionSystem

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("DisplayResults")

def display_results():
    config = load_config('configs/production_fresh.yaml')
    
    # 检查缓存目录
    cache_dir = config.paths.cache_dir
    import glob
    chunks = glob.glob(os.path.join(cache_dir, "*.npz"))
    logger.info(f"找到 {len(chunks)} 个数据块缓存文件。")
    
    if len(chunks) == 0:
        logger.error("尚未生成数据缓存，请先运行 pipeline。")
        return

    logger.info("开始加载缓存并进行归因分析 (这在全量真实数据上可能需要几十分钟)...")
    
    # 扩大样本抽取率以防内存不足或提速
    config.attribution.background_sample_rate = 0.01 
    
    system = EcoDisturbanceAttributionSystem(config=config)
    try:
        result = system.run()
        
        print("\n" + "="*80)
        print("🎉 诱导因子挖掘与生态扰动归因完成 🎉")
        print("="*80)
        
        print(f"\n[数据统计]")
        print(f"训练样本总数: {len(result['training_samples'])}")
        print(f"提取特征数: {len(system.feature_columns)}")
        
        print("\n[排名最高的核心诱导因子 (Top 10)]")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        factors_df = result['factors']
        print(factors_df.head(10)[['conditions', 'relative_risk', 'support', 'stability']])
        
        print("\n[单点归因预测示例 (前 5 条)]")
        preview_df = result['attribution_preview']
        print(preview_df.head())
        
        print(f"\n详细结果已保存至: {config.paths.out_dir}")
        print(" - training_samples.csv")
        print(" - factor_library.csv")
        print(" - attribution_preview.csv")
        print(" - bayesian_network.bif")
        
    except Exception as e:
        logger.error(f"归因运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    display_results()