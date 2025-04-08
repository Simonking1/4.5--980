import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 读取数据
def read_data():
    data_xls = pd.read_excel('data.xls', index_col=0)
    charging_records = pd.read_csv('ChargingRecords_random_20.csv')
    return data_xls, charging_records

# 随机选择 20 个充电桩
def select_random_chargers(charging_records):
    return charging_records.sample(20)

# 计算初始负荷和碳排放量
def calculate_initial_load_and_emissions(data_xls, chargers):
    # 假设初始碳排放量系数
    carbon_emission_factor = 0.5
    # 初始化 24 小时负荷
    total_load = np.zeros(24)
    for _, row in data_xls.iterrows():
        total_load += row.values[:24]
    # 加入充电桩负荷
    for _, charger in chargers.iterrows():
        start_time = int(charger['StartTime'].split(':')[0])
        end_time = int(charger['EndTime'].split(':')[0])
        demand = charger['Demand']
        duration = charger['Duration'] / 60
        load_per_hour = demand / duration
        for t in range(start_time, end_time):
            total_load[t] += load_per_hour
    # 计算碳排放量
    carbon_emissions = total_load * carbon_emission_factor
    return total_load, carbon_emissions

# 优化算法：在负荷低的时候充电
def optimize_charging(data_xls, chargers):
    # 假设初始碳排放量系数
    carbon_emission_factor = 0.5
    # 初始化 24 小时负荷
    total_load = np.zeros(24)
    for _, row in data_xls.iterrows():
        total_load += row.values[:24]
    # 优化后的负荷
    optimized_load = total_load.copy()
    # 对每个充电桩进行优化
    for _, charger in chargers.iterrows():
        demand = charger['Demand']
        duration = charger['Duration'] / 60
        load_per_hour = demand / duration
        # 找到负荷最低的时间段
        min_load_index = np.argmin(optimized_load[:24 - int(duration)])
        for t in range(min_load_index, min_load_index + int(duration)):
            optimized_load[t] += load_per_hour
    # 计算优化后的碳排放量
    optimized_carbon_emissions = optimized_load * carbon_emission_factor
    return optimized_load, optimized_carbon_emissions

# 绘制图表
def plot_results(initial_load, initial_emissions, optimized_load, optimized_emissions):
    # 绘制 24 小时负荷对比图
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(24), initial_load, label='Initial Load')
    plt.plot(np.arange(24), optimized_load, label='Optimized Load')
    plt.xlabel('Time (hours)')
    plt.ylabel('Load')
    plt.title('24-Hour Load Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 绘制 24 小时碳排放量对比图
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(24), initial_emissions, label='Initial Carbon Emissions')
    plt.plot(np.arange(24), optimized_emissions, label='Optimized Carbon Emissions')
    plt.xlabel('Time (hours)')
    plt.ylabel('Carbon Emissions')
    plt.title('24-Hour Carbon Emissions Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

# 绘制横向柱状图对比不同时间点优化前后充电负荷
def plot_horizontal_bar(initial_load, optimized_load):
    times = [4, 7, 13, 20]
    objects = ['DC1', 'DC2', 'DC3', 'DC4']
    bar_width = 0.35
    index = np.arange(len(times))

    initial_load_at_times = [initial_load[t] for t in times]
    optimized_load_at_times = [optimized_load[t] for t in times]

    fig, ax = plt.subplots()
    for i, obj in enumerate(objects):
        ax.barh(index - bar_width/2 + i * bar_width/len(objects), initial_load_at_times,
                bar_width/len(objects), label=f'{obj} Initial', color=f'C{i}')
        ax.barh(index + bar_width/2 + i * bar_width/len(objects), optimized_load_at_times,
                bar_width/len(objects), label=f'{obj} Optimized', color=f'C{i}', alpha=0.5)

    ax.set_xlabel('Charging Load (MW)')
    ax.set_ylabel('Time')
    ax.set_title('Comparison of Charging Load Before and After Optimization at Different Times')
    ax.set_yticks(index)
    ax.set_yticklabels([f'{t}:00' for t in times])
    ax.legend()

    plt.show()

# 主函数
def main():
    data_xls, charging_records = read_data()
    chargers = select_random_chargers(charging_records)
    initial_load, initial_emissions = calculate_initial_load_and_emissions(data_xls, chargers)
    optimized_load, optimized_emissions = optimize_charging(data_xls, chargers)
    plot_results(initial_load, initial_emissions, optimized_load, optimized_emissions)
    plot_horizontal_bar(initial_load, optimized_load)

if __name__ == "__main__":
    main()
    
