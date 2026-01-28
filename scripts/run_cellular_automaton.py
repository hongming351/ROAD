#!/usr/bin/env python3
"""
元胞自动机微观模型运行脚本
"""

import os
import sys
import subprocess

def run_cellular_automaton():
    """运行元胞自动机微观模型"""
    print("=" * 60)
    print("运行元胞自动机微观模型")
    print("=" * 60)
    
    # 切换到元胞自动机目录
    ca_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'cellular_automaton')
    os.chdir(ca_dir)
    
    # 运行元胞自动机仿真
    try:
        result = subprocess.run([sys.executable, 'CA.py'], 
                              capture_output=True, text=True, check=True)
        print("元胞自动机仿真完成!")
        print(result.stdout)
        
    except subprocess.CalledProcessError as e:
        print(f"运行出错: {e}")
        print(f"错误输出: {e.stderr}")
        return False
    
    return True

def run_bayesian_optimization():
    """运行贝叶斯优化"""
    print("=" * 60)
    print("运行贝叶斯优化参数校准")
    print("=" * 60)
    
    # 切换到元胞自动机目录
    ca_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'cellular_automaton')
    os.chdir(ca_dir)
    
    # 运行贝叶斯优化
    try:
        result = subprocess.run([sys.executable, 'bayesian_optimization.py'], 
                              capture_output=True, text=True, check=True)
        print("贝叶斯优化完成!")
        print(result.stdout)
        
    except subprocess.CalledProcessError as e:
        print(f"运行出错: {e}")
        print(f"错误输出: {e.stderr}")
        return False
    
    return True

def run_integrated_model():
    """运行集成模型"""
    print("=" * 60)
    print("运行集成模型")
    print("=" * 60)
    
    # 切换到元胞自动机目录
    ca_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'cellular_automaton')
    os.chdir(ca_dir)
    
    # 运行集成模型
    try:
        result = subprocess.run([sys.executable, 'integrate_ml2.py'], 
                              capture_output=True, text=True, check=True)
        print("集成模型完成!")
        print(result.stdout)
        
    except subprocess.CalledProcessError as e:
        print(f"运行出错: {e}")
        print(f"错误输出: {e.stderr}")
        return False
    
    return True

if __name__ == "__main__":
    print("选择要运行的元胞自动机相关脚本:")
    print("1. 元胞自动机仿真")
    print("2. 贝叶斯优化")
    print("3. 集成模型")
    print("4. 运行全部")
    
    choice = input("请输入选择 (1-4): ").strip()
    
    success = True
    
    if choice == "1":
        success = run_cellular_automaton()
    elif choice == "2":
        success = run_bayesian_optimization()
    elif choice == "3":
        success = run_integrated_model()
    elif choice == "4":
        success = run_cellular_automaton()
        if success:
            success = run_bayesian_optimization()
        if success:
            success = run_integrated_model()
    else:
        print("无效选择!")
        sys.exit(1)
    
    if success:
        print("\n元胞自动机脚本运行成功!")
    else:
        print("\n元胞自动机脚本运行失败!")
        sys.exit(1)