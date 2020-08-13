# 该模块功能：
    
    - 接口：38083    
    
    - 本机调试：事件相似度匹配 127.0.0.1:38083/event_match
               事件向量化保存 127.0.0.1:38083/vec_save
               事件向量删除 127.0.0.1:38083/vec_delete
               事件检索     127.0.0.1::38083/search
    
    - 事件相似度匹配：使用传入的事件匹配事件表中向量化后的事件；
    
                接口输入的数据格式：{"short_sentence":""， "cameo":"", "threshold":0.6}
                
                关键字说明：short_sentence：事件短句，由主语、否定词、谓语、宾语组成的字符串--str
                           cameo：事件cameo编号--str, 如果不传cameo则默认检索所有事件。
                           threshold：阈值--float 默认为0.5
                
                接口输出的数据格式：{
                                      "result": [
                                                  {
                                                    "event_id": "",
                                                    "score": 0.6354377269744873
                                                   }
                                                 ],
                                      "status": "success"
                                    }
                                    
                关键字说明：status:状态
                
                           result:匹配得到的结果
                           
                           event_id:事件编号
                           
                           score:相似度

    
    - 事件向量化保存：将事件短句进行向量化并保存到文件中
    
               接口输入的数据格式：{"short_sentence":""， "cameo":"", "event_id":""}
               
               关键字说明：short_sentence：事件短句，由主语、否定词、谓语、宾语组成的字符串--str
               
                          cameo：事件cameo编号--str
                           
                          event_id：需要保存的事件编号--str
            
               接口输出的数据格式：{"status":"success", "message":""}
               
               关键字说明：status: 状态
               
                          message: 信息
                          
    - 事件向量删除：根据传入的事件id，将对应的向量删除
               
               接口输入数据格式：{"event_id":""}
               
               接口输出数据格式：{"status":"", "message":""}
               
    - 事件检索：传入事件短句，检索相似的事件
    
               接口输入数据格式：{"short_sentence":""}
               
               接口输出数据格式: {"status":"","message":""}
               

## algor
    
    存放模型测试与预测所有的代码

    ### predict
            
        - delete_vec.py 删除向量模块
        
        - load_all_model.py 加载所有的模型以及调用模型进行预测
        
        - load_vec.py 加载向量模块
        
        - save_vec.py 保存向量模块
            
    ### train
    
        - utils 模型训练数据处理以及生成模块
        
        - event_match_train.py 匹配模型训练代码
        
## config

    存放模块参数
    
## restapp
    
    - main.py flask接口
        
