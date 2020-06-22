# 该模块功能

    - 端口：38081
    
    - 本机调试：127.0.0.1:38081/ematch?title=日本派遣宙斯盾军舰赶赴东部海域.&content=
    
    - 对输入的中文文章标题以及内容进行事件匹配，输出匹配结果
    
    - 输入:{"title":"",
    
            "content":"",
            
            "sample_type:""(也可以不用传，底层默认parts)[parts abstract triples]
            
            }
    
    
    -输出: {"code": 0,
    
            "data":[{"title_pred":[
                    
                        {"event_id": "3",
                        
                         "ratio": 0.753}]},
                         
                     {"content_pred":[
                        
                        {"event_id": "3",
                        
                         "ratio": 0.753}]}],
                     
            "message": "success"}
            
## 使用前准备

    将对应的模型存放到对应的文件夹下，然后运行main.py文件即可
    
## algor
    
    存放模型测试与预测所有的代码
    
    ### common

        - utils.py 训练模型以及预测时构建数据生成器代码    

    ### predict
    
        - execute.py 预测过程中加载事件列表、加载模型、预测以及预测结果的整理代码
        
        - get_abstract.py 获取摘要的代码
            
    ### train

        - match_model_train.py 匹配模型训练代码
        
## config

    存放模块参数类
    
## restapp
    
    - main.py flask接口
        