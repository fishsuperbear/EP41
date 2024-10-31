### 自制流feature简单说明

##### 概述

> 在nv提供的demo基础上新增一个element（ELEMENT_TYPE_CUST_RAW），自定义buffer格式与大小，
> 新增自定义producer（CCustProducer）与consumer（CCustConsumer），专门发送与处理ELEMENT_TYPE_CUST_RAW

##### nvstream event触发顺序以及对应处理函数

##### 1. NvSciStreamEventType_Connected				===》init  ==》handleElemSupport

				//nv提供的demo里单独处理了connect事件，并通过init函数处理buffer初始化(buffer格式等相关信息)等

##### 2. NvSciStreamEventType_Elements               ===》handleElemSetting

				//Set waiter attrs

##### 3. NvSciStreamEventType_PacketCreate           ===》handlePacketCreate

				//根据buffer格式创建对应的packet

##### 4. NvSciStreamEventType_PacketsComplete        ===》NvSciStreamBlockSetupStatusSet

				//PacketsComplete  import packet

##### 5. NvSciStreamEventType_WaiterAttr             ===》handleSyncExport							||exception

				//Reconcile

##### 6. NvSciStreamEventType_SignalObj              ===》HandleSyncImport


				//register waiter

##### 7. NvSciStreamEventType_SetupComplete          ===》handleSetupComplete

				//通知所有 block，SetupComplete 

##### 8. NvSciStreamEventType_PacketReady            ===》handlePayload

				//若前面的事件没有异常，在SetupComplete之后，producer就会先接收到PacketReady，填充数据send出去后consumer收到PacketReady开始处理。
					consumer处理完后释放buffer，producer就会重新接收到PacketReady，开启循环，这便完成了自制流的流程

#### 当前主要问题：

1.Reconcile 失败，自定义buffer格式配置异常或与waiter配合异常
