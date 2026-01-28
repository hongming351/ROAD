#!/usr/bin/env python3
"""
网络均衡模型运行脚本
"""

import os
import sys
import subprocess

def run_network_equilibrium():
    """运行网络均衡模型"""
    print("=" * 60)
    print("运行网络均衡模型")
    print("=" * 60)
    
    # 切换到网络均衡目录
    net_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'network_equilibrium')
    os.chdir(net_dir)
    
    # 运行网络均衡分析
    try:
        result = subprocess.run([sys.executable, 'network_equilibrium_model.py'], 
                              capture_output=True, text=True, check=True)
        print("网络均衡分析完成!")
        print(result.stdout)
        
    except subprocess.CalledProcessError as e:
        print(f"运行出错: {e}")
        print(f"错误输出: {e.stderr}")
        return False
    
    return True

def run_od_estimation():
    """运行OD需求估计"""
    print("=" * 60)
    print("运行OD需求估计")
    print("=" * 60)
    
    # 切换到网络均衡目录
    net_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'network_equilibrium')
    os.chdir(net_dir)
    
    # 运行OD估计
    try:
        result = subprocess.run([sys.executable, 'python od_estimation.py'], 
                              capture_output=True, text=True, check=True)
        print("OD需求估计完成!")
        print(result.stdout)
        
    except subprocess.CalledProcessError as e:
        print(f"运行出错: {e}")
        print(f"错误输出: {e.stderr}")
        return False
    
    return True

if __name__ == "__main__":
    print("选择要运行的网络均衡相关脚本:")
    print("1. 网络均衡模型")
    print("2. OD需求估计")
    print("3. 运行全部")
    
    choice = input("请输入选择 (1-3): ").strip()
    
    success = True
    
    if choice == "1":
        success = run_network_equilibrium()
    elif choice == "2":
        success = run_od_estimation()
    elif choice == "3":
        success = run_network_equilibrium()
        if success:
            success = run_od_estimation()
    else:
        print("无效选择!")
        sys.exit(1)
    
    if success:
        print("\n网络均衡脚本运行成功!")
    else:
        print("\n网络均衡脚本运行失败!")
        sys.exit(1)