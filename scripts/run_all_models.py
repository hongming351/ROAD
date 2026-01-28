#!/usr/bin/env python3
"""
运行所有模型的主脚本
"""

import os
import sys
import subprocess
import time

def run_script(script_path, description):
    """运行指定脚本"""
    print(f"\n{'='*60}")
    print(f"运行: {description}")
    print(f"{'='*60}")
    
    try:
        # 获取脚本的绝对路径
        script_abs_path = os.path.abspath(script_path)
        script_dir = os.path.dirname(script_abs_path)
        script_name = os.path.basename(script_abs_path)
        
        # 切换到脚本所在目录
        original_dir = os.getcwd()
        os.chdir(script_dir)
        
        # 运行脚本
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, check=True)
        
        print(f"{description} 完成!")
        print(result.stdout)
        
        # 切换回原目录
        os.chdir(original_dir)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"{description} 运行出错: {e}")
        print(f"错误输出: {e.stderr}")
        return False
    except Exception as e:
        print(f"运行 {description} 时发生异常: {e}")
        return False

def main():
    """主函数"""
    print("=" * 80)
    print("ROAD交通流分析项目 - 全部模型运行脚本")
    print("=" * 80)
    
    # 获取脚本目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 定义要运行的脚本
    scripts_to_run = [
        (os.path.join(script_dir, 'run_lwr_analysis.py'), "LWR宏观交通流分析"),
        (os.path.join(script_dir, 'run_cellular_automaton.py'), "元胞自动机微观模型"),
        (os.path.join(script_dir, 'run_network_equilibrium.py'), "网络均衡模型"),
    ]
    
    # 询问用户是否要运行所有模型
    print("\n即将运行所有交通流分析模型:")
    for i, (_, description) in enumerate(scripts_to_run, 1):
        print(f"  {i}. {description}")
    
    print(f"\n注意: 这将运行 {len(scripts_to_run)} 个模型，可能需要较长时间。")
    
    choice = input("\n是否继续? (y/n): ").strip().lower()
    
    if choice not in ['y', 'yes', '是', '继续']:
        print("已取消运行。")
        return
    
    # 记录开始时间
    start_time = time.time()
    
    # 运行所有脚本
    success_count = 0
    total_count = len(scripts_to_run)
    
    for script_path, description in scripts_to_run:
        if os.path.exists(script_path):
            if run_script(script_path, description):
                success_count += 1
        else:
            print(f"警告: 脚本文件不存在 - {script_path}")
    
    # 计算运行时间
    end_time = time.time()
    total_time = end_time - start_time
    
    # 输出总结
    print("\n" + "=" * 80)
    print("运行总结")
    print("=" * 80)
    print(f"总运行时间: {total_time:.2f} 秒")
    print(f"成功运行: {success_count}/{total_count} 个模型")
    
    if success_count == total_count:
        print("\n✅ 所有模型运行成功!")
        print("\n输出文件说明:")
        print("- LWR分析结果: src/lwr/ 目录下")
        print("- 元胞自动机结果: src/cellular_automaton/ 目录下")
        print("- 网络均衡结果: src/network_equilibrium/ 目录下")
        print("- 可视化图表: results/visualizations/ 目录下")
        print("- 分析结果: results/processed/ 目录下")
    else:
        print(f"\n❌ 有 {total_count - success_count} 个模型运行失败")
        print("请检查错误信息并重试")
        sys.exit(1)

if __name__ == "__main__":
    main()