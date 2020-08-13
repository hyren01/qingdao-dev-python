# 该模块功能：
    
    - 接口：38083    
    
    - 本机调试：事件相似度匹配 127.0.0.1:38083/event_match
               事件向量化保存 127.0.0.1:38083/vec_save
               事件向量删除 127.0.0.1:38083/vec_delete
    
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
               

## predict

    模型预测需要使用的所有代码。
    
    1、load_all_model.py  加载bert模型和相似度匹配模型，将数据转化为向量并进行相似度计算
    
    2、load_vec.py 暂定从文件中读取向量传送给flask模块，后期改为从数据库中读取
    
    3、save_vec.py 暂定将向量保存到文件中，后期改为将向量保存到数据库中
    
    4、data_utils.py 数据相关的公共模块
    
    5、delete_vec.py 向量删除模块


## resourses
    
    1、chinese_roberta_wwm_ext_L-12_H-768_A-12--存放bert模型的参数以及字典
    
    2、model--存放训练好的模型文件
    
    3、vec_data--暂定将转化后的向量保存到该文件夹中
    
    4、代码执行后会生成cameo2id.json文件，用来保存cameo编号与事件id对应字典
    
## train
    
    1、model--存放训练后的模型
              
    2、resourses--存放训练数据以及第三方包
    
    3、event_match_train.py--事件相似度模型训练
    
    备注：训练代码执行前需要安装  tqdm==4.41.0  scikit-learn==0.19.1


## restapp 

    main.py--flask主控接口
    