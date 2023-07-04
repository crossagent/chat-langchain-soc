import csv

def read_csv(filename):
    with open(filename, "r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            print(row)

# 指定要读取的CSV文件路径
filename = "docs\SOC-jiekouren.csv"

# 调用函数读取CSV文件
def read_csv(filename):
    modules = {}
    with open(filename, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        next(reader)  # 跳过首行
        for row in reader:
            module_info = {"name": row['细分'], "module": row['大模块'], "contact_person":row['策划接口'], "progammer":row['技术接口'], "quality_assurance":row['QA接口'], "description":row['模块功能']}
            modules[row['细分']] = module_info
    return modules

ALL_MODULE = read_csv(filename)