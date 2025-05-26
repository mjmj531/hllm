import pandas as pd
import os

def split_csv(input_file, output_file1, output_file2):
    """
    将一个 CSV 文件拆分为前后两半，并分别保存到两个文件中。

    Args:
        input_file (str): 输入的 CSV 文件路径。
        output_file1 (str): 保存前半部分的文件路径。
        output_file2 (str): 保存后半部分的文件路径。
    """
    # 读取 CSV 文件
    df = pd.read_csv(input_file)

    # 计算1/4索引
    quarter_index = len(df) // 16

    # 拆分为前1/4和后3/4
    df_first_half = df.iloc[:quarter_index]
    df_second_half = df.iloc[quarter_index:]

    # 保存到两个文件
    df_first_half.to_csv(output_file1, index=False)
    df_second_half.to_csv(output_file2, index=False)

    print(f"文件已拆分并保存到：\n前半部分：{output_file1}\n后半部分：{output_file2}")

# 示例用法
input_csv = "/home/stu2/HLLM/dataset/amazon_books_first_half.csv"  # 输入的 CSV 文件路径
output_csv1 = "/home/stu2/HLLM/dataset/amazon_books_first_first_half.csv"  # 前半部分保存路径
output_csv2 = "/home/stu2/HLLM/dataset/amazon_books_first_second_half.csv"  # 后半部分保存路径

os.makedirs(os.path.dirname(output_csv1), exist_ok=True)  # 创建输出目录
os.makedirs(os.path.dirname(output_csv2), exist_ok=True)
split_csv(input_csv, output_csv1, output_csv2)