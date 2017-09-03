# xiaoming 小明
a seq2seq + attention chatbot

基于PyTorch实现的智能聊天机器人，采用seq2seq+attention模型

## environment 运行环境
* centos
* python3.5+
* PyTorch

## Synopsis 演示
```
me > 你是

我是__robotname__请问有什么可以帮您吗<EOS>

me > 我叫什么

您是__username__<EOS>

me > 天气怎样

__actWeather__<EOS>

me > 气温如何

__actWeather__<EOS>

me > 中国人

中国是以华夏文明为源泉中华文化为基础并以汉族为主体民族的多民族国家通用汉语汉字汉族与少数民族被为中华民族又自称为炎黄子孙龙的传人split我爱中国<EOS>

me > 明天 下午三点开会

__actAddPlan__<EOS>


```

### usage
* train model 模型的训练
```
python3 train.py
```
* test model 模型测试
```
python3 test.py
```

## 问题反馈
欢迎和我交流！共同学习！

* 邮件(lxm_0828#163.com, 把#换成@)
* QQ: 929325776
* weibo: [@捏明](http://weibo.com/littlelxm)


##关于作者

```
学生一枚，就读于山大（shanxi university）, 对deep learning(深度学习), NLP(自然语言处理)，NLU(自然语言理解)，Big Data(大数据)有狂热的学习欲望，精通WEB。
```
