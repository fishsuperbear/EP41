目前跑通了单consumer的模式，通过修改hw_nvmedia_eventhandler_common_impl.h文件中的NUM_IPC_CONSUMERS为1U
启动hw_nvmedia_ipc_producer和启动hw_nvmedia_ipc_consumer_cuda既可完成通信