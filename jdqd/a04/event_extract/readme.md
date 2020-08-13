# 该模块功能：
    
    - 端口：38082    
    
    - 本机调试：事件抽取 127.0.0.1:38082/event_extract
    
               指代消解 127.0.0.1:38082/coref_with_content
               
               模型训练 127.0.0.1:38082/event_train
               
    - 事件抽取：抽取中文文本中的事件；
    
                接口输入的数据格式：{"sentence":""}
                
                关键字说明：sentence：需要提取事件的内容，为中文字符串，句子或者文章
                
                接口输出的数据格式：{ "status":"success", 
                                     "data":[{"sentence":sentence, 
                                              "sentence_id":"0",
                                              "events":[
                                                      { "triggerloc_index":[12, 15],
                                                        "event_id":"0-0",
                                                        "subject":"", 
                                                        "verb":"", 
                                                        "object":"",
                                                        "event_datetime":"", 
                                                        "event_location":"", 
                                                        "negative_word":"", 
                                                        "state":"", 
                                                        "cameo":"",
                                                        "namedentity":{"person":[], 
                                                                       "location":[], 
                                                                       "organization":[]}  
                                                       } ]
                                              }
                                                              
                                             ]                
                                    }
                                    
                关键字说明：status:状态
                           data:预测得到的数据
                           sentence：抽取事件的单个句子
                           sentence_id:句子编号
                           events:改句子中对应的所有事件
                           triggerloc_index:事件动词在句子中的下标
                           event_id:事件编号，格式为------句子编号-第n个事件
                           subject:事件主语
                           verb:事件动词
                           object:宾语
                           event_datatime:事件发生的时间
                           event_location:事件发生的地点
                           nagative_word:否定词
                           state:事件状态
                           cameo:事件cameo编号
                           namedentity:命名实体
                           person:事件中所有的人物
                           location:事件中所有的地点
                           organization:事件中所有的组织机构
    
    - 指代消解：将传入的英文进行指代消解，消解后返回翻译后的中文字符串；
    
               接口输入的数据格式：{"content":""}
               
               关键字说明：content: 需要指代消解的英文字符串内容
            
               接口输出的数据格式：{"status":"success", "coref":""}
               
               关键字说明：status: 状态
                          coref:经过指代消解并翻译成中文的字符串
    
## 使用前准备

    - 手动下载安装：

        jdk==1.8.0.0 
        
        hanlp:data-for-1.7.4.zip 解压将data文件夹放到hanlp包static文件夹下
        
        spacy:en_core_web_sm-2.1.0.tar.gz 需要pip 安装
        
        .neuralcoref_cache 第一次运行指代消解需要联网下载，如果下载速度会很慢，
                           可以到resources文件夹中获取并映射到对应的文件夹。

        pip3 install git+http://139.9.126.19:38111/hyren/feedwork-py.git
        
        将EventExtract设置为根目录export PYTHONPATH=$"D:\work\QingDao_Graph\QingDao_EventExtract\event_extract_program\EventExtract"
        
        运行 main.py文件
        
        
## algor
    
    存放模型测试与预测所有的代码

    ### predict
    
        - coref_spacy.py 指代消解模块
                
        - load_all_model.py 加载所有的模型以及调用模型进行预测
        
        - xiaoniu_translate.py 小牛翻译所有的代码
    
    ### train
    
        - utils 模型训练数据处理以及生成模块
        
        - event_cameo_train.py cameo训练代码
        
        - event_extract_train.py 事件抽取模型训练代码
        
        - event_state_train.py 事件状态训练代码
        
        - extract_data_analysis.py
    
## config

    存放模块参数类
    
## restapp
    
    - main.py flask接口
    