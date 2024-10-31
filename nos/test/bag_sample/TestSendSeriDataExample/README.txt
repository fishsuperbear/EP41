从包中读取数据，并将数据发布出去，订阅者可以静态的方式接收数据

1.test_data中的binary.dat保存要发送的消息的序列化数据
2.TestWorld是数据类，从binary.dat中读取的一个消息的数据存在这里，后续用于发布
3.TestWorld.getMaxCdrSerializedSize()会在初始化writer缓存的时候用到(表示要发送的数据类型的序列化数据的最大长度)，当实际发送的数据大于此值，会重新开辟缓存，跟效率有关
4.TestWorld.getCdrSerializedSize()代表当前消息的序列化数据的长度，需要在从binary.dat中读取的一个massage后，判断其长度，然后进行设置
5.HelloWorldPubSubTypes是序列化TestWorld数据的辅助类，主要是serialize()函数
6.HelloWorldPubSubTypes.serialize()主要用来获取要发送消息的序列化数据，此处返回TestWorld对象中保存的序列化数据

***TestWorld.isKeyDefined()永远为false，与Topic kind有关，如果Topic kind=WITH_KEY,则isKeyDefined()应返回true，且HelloWorldPubSubTypes.getkey()返回所需的key，key具体的作用参考https://fast-dds.docs.eprosima.com/en/latest/fastdds/dds_layer/topic/typeSupport/typeSupport.html


