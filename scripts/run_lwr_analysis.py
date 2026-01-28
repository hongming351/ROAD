#!/usr/bin/env python3
"""
LWR宏观交通流分析运行脚本
"""

import os
import sys
import subprocess

def run_lwr_analysis():
    """运行LWR宏观交通流分析"""
    print("=" * 60)
    print("运行LWR宏观交通流分析")
    print("=" * 60)
    
    # 切换到LWR目录
    lwr_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'lwr')
    os.chdir(lwr_dir)
    
    # 运行LWR分析
    try:
        result = subprocess.run([sys.executable, 'lwr_analysis.py'], 
                              capture_output=True, text=True, check=True)
        print("LWR分析完成!")
        print(result.stdout)
        
        # 运行拥堵分类器
        result = subprocess.run([sys.executable, 'congestion_classifier.py'], 
                              capture_output=True, text=True, check=True)
        print("拥堵分类器完成!")
        print(result.stdout)
        
    except subprocess.CalledProcessError as e:
        print(f"运行出错: {e}")
        print(f"错误输出: {e.stderr}")
        return False
    
    return True

if __name__ == "__main__":
    success = run_lwr_analysis()
    if success:
        print("\nLWR分析脚本运行成功!")
    else:
        print("\nLWR分析脚本运行失败!")
        sys.exit(1)