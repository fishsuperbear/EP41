#include <limits>
#include <thread>
#include <iostream>
#include <atomic>
#include <algorithm>
#include <cmath>
#include <unistd.h>
#include <string>
#include <unordered_map>

#include <Common.hpp>
#include "CPoolManager.hpp"
#include <CCustProducer.hpp>
#include <CCustConsumer.hpp>
#include <CEventHandler.hpp>

// 原子变量，用于控制程序是否退出
std::atomic<bool> g_bExit(false);

// 监听键盘输入的函数
void keyboardListener()
{
    std::cout << "Press 'q' to exit the program." << std::endl;

    // 循环读取键盘输入
    while (!g_bExit)
    {
        char ch = std::cin.get(); // 读取一个字符

        // 判断是否是 'q' 字符
        if (ch == 'q')
        {
            g_bExit = true; // 设置退出标志
            break; // 退出循环
        }
    }
}
static void EventThreadFunc(CEventHandler *pEventHandler)
{
    uint32_t timeouts = 0U;
    EventStatus eventStatus = EVENT_STATUS_OK;

    string threadName = pEventHandler->GetName();
    pthread_setname_np(pthread_self(), threadName.c_str());

    /* Simple loop, waiting for events on the block until the block is done */
    while (!g_bExit) {
        eventStatus = pEventHandler->HandleEvents();
        if (eventStatus == EVENT_STATUS_TIMED_OUT) {
            // if query timeouts - keep waiting for event until wait threshold is reached
            if (timeouts < MAX_QUERY_TIMEOUTS) {
                timeouts++;
                continue;
            }
            printf((pEventHandler->GetName() + ": HandleEvents() seems to be taking forever!\n").c_str());
        } else if (eventStatus == EVENT_STATUS_OK) {
            timeouts = 0U;
            continue;
        } else if (eventStatus == EVENT_STATUS_COMPLETE) {
            break;
        } else {
            g_bExit = true;
            break;
        }
    }
}

int main(int argc, char *argv[])
{
    bool bUseMailbox = false;
    // Open NvSciSync/NvSciBuf modules
    NvSciBufModule bufModule{ nullptr };
    NvSciBufModuleOpen(&bufModule);
    NvSciSyncModule syncModule{ nullptr };
    NvSciSyncModuleOpen(&syncModule);
    std::vector<CEventHandler *> vEventThreadHandlers;
    NvSciStreamBlock                producerHandle{ 0U };
    NvSciStreamBlock                poolHandle{ 0U };

    NvSciStreamBlock                multicastiHandle{ 0U };

    NvSciStreamBlock                queueHandle{ 0U };
    NvSciStreamBlock                consumerHandle{ 0U };
    //--------------------Create Stream
    //create pool and producer
    std::vector<ElementInfo> elemsInfo;
    elemsInfo.resize(MAX_NUM_ELEMENTS);
    elemsInfo[ELEMENT_TYPE_ICP_RAW] = { ELEMENT_TYPE_ICP_RAW, true };
    /* elemsInfo[ELEMENT_TYPE_NV12_BL] = { ELEMENT_TYPE_NV12_BL, true }; */
    /* elemsInfo[ELEMENT_TYPE_METADATA] = { ELEMENT_TYPE_METADATA, true }; */
    auto sciErr = NvSciStreamStaticPoolCreate(MAX_NUM_PACKETS, &poolHandle);
    std::unique_ptr<CPoolManager> m_upPoolManager(new CPoolManager(poolHandle, 0/*sensor ID*/, MAX_NUM_PACKETS));
    vEventThreadHandlers.push_back(m_upPoolManager.get());

    NvSciStreamProducerCreate(poolHandle, &producerHandle);
    std::unique_ptr<CCustProducer> m_upProducer(new CCustProducer("CCustProDucer",producerHandle,0/*sensor ID*/));
    m_upProducer->SetPacketElementsInfo(elemsInfo);
    vEventThreadHandlers.push_back(m_upProducer.get());

    //create queue and consumer
    if (bUseMailbox) {
        NvSciStreamMailboxQueueCreate(&queueHandle);
    } else {
        NvSciStreamFifoQueueCreate(&queueHandle);
    }
    NvSciStreamConsumerCreate(queueHandle, &consumerHandle);
    std::unique_ptr<CCustConsumer> upCons(new CCustConsumer("CCustConsumer",consumerHandle,0/*sensor ID*/,queueHandle));
    upCons->SetPacketElementsInfo(elemsInfo);
    vEventThreadHandlers.push_back(upCons.get());
    //connect stream
    NvSciStreamBlockConnect(producerHandle, consumerHandle);

    //start new thread to handle Events
    std::vector<std::unique_ptr<std::thread>> vupThreads;
    for (const auto &pEventHandler : vEventThreadHandlers) {
        vupThreads.push_back(std::make_unique<std::thread>(EventThreadFunc, pEventHandler));
    }
    //init blocks
    m_upPoolManager->Init();
    m_upProducer->Init(bufModule,syncModule);
    upCons->Init(bufModule,syncModule);

    // 新建一个线程来运行键盘监听函数
    std::thread t(keyboardListener);

    // 等待用户输入 'q' 来退出程序
    while (!g_bExit)
    {
        // 让主线程空转一段时间
        uint32_t pIndex = m_upProducer->getFromPacketIndexs();
        if(pIndex == -1){
            printf("get pIndex = %d\n", pIndex);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        m_upProducer->post(pIndex);
    }

    for (auto &upThread : vupThreads) {
        if (upThread != nullptr) {
            upThread->join();
            upThread.reset();
            LOG_DBG("upThread->join.\n");
        }
    }
    // 等待键盘监听线程结束
    t.join();
    return 0;
}
