# 目录结构
```ini
|-- README.md
|-- jdqd                          # 项目名
|   |-- common                    # 该工程范围内的公共方法归属包
|   |-- a01                       # 事件提取
|   |   |-- [module name]         # 模块名
|   |-- a03                       # 事件预测
|   |-- a04                       # 事理图谱
|-- resources                     # 放全局路径和数据库的一些参数
|   |-- fdconfig                  # 所有在不同运行环境下需要动态修改的配置信息
|   |   |-- appinfo.conf          # 全局文件路径 
|   |   |-- dbinfo.conf           # 全局数据库的参数
|   |-- module                    # 训练好的模型文件的存放根目录。下面用模块名创建子目录
|   |   |-- [module name]         # 存放该模块训练出来的模型文件
|   |-- pretrain                  # 依赖的预训练模型
|-- testsuite                     # 测试用例代码。
|   |-- [module name]             # 与被测试模块一一对应
|   |   |-- algor                 # 
|   |   |-- services              #
```