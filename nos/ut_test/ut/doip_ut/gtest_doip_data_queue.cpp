

#include "gtest/gtest.h"
#include "base/doip_data_queue.h"
TEST(DoipDataQueue, DoipQueueEmpty) {
    hozon::netaos::diag::DoipDataQueue queue;
    EXPECT_TRUE(queue.DoipQueueEmpty());
}
TEST(DoipDataQueue, DoipQueueSize) {
    hozon::netaos::diag::DoipDataQueue queue;
    EXPECT_EQ((uint32_t)0, queue.DoipQueueSize());
    hozon::netaos::diag::doip_cache_data_t* cache_data1 = new hozon::netaos::diag::doip_cache_data_t;
    cache_data1->data = new char[4];
    queue.DoipInsertQueue(cache_data1);
    EXPECT_EQ((uint32_t)1, queue.DoipQueueSize());
    hozon::netaos::diag::doip_cache_data_t* cache_data2 = new hozon::netaos::diag::doip_cache_data_t;
    cache_data2->data = new char[4];
    queue.DoipInsertQueue(cache_data2);
    EXPECT_EQ((uint32_t)2, queue.DoipQueueSize());
}
TEST(DoipDataQueue, DoipInsertQueue) {
    hozon::netaos::diag::DoipDataQueue queue;
    EXPECT_EQ((uint32_t)0, queue.DoipQueueSize());
    hozon::netaos::diag::doip_cache_data_t* cache_data1 = new hozon::netaos::diag::doip_cache_data_t;
    cache_data1->data = new char[4];
    queue.DoipInsertQueue(cache_data1);
    EXPECT_EQ((uint32_t)1, queue.DoipQueueSize());
    hozon::netaos::diag::doip_cache_data_t* cache_data2 = new hozon::netaos::diag::doip_cache_data_t;
    cache_data2->data = new char[4];
    queue.DoipInsertQueue(cache_data2);
    EXPECT_EQ((uint32_t)2, queue.DoipQueueSize());
    hozon::netaos::diag::doip_cache_data_t* cache_data3 = new hozon::netaos::diag::doip_cache_data_t;
    cache_data3->data = new char[4];
    queue.DoipInsertQueue(cache_data3);
    EXPECT_EQ((uint32_t)3, queue.DoipQueueSize());
}
TEST(DoipDataQueue, DoipPopFrontQueue) {
    hozon::netaos::diag::DoipDataQueue queue;
    EXPECT_EQ((uint32_t)0, queue.DoipQueueSize());
    EXPECT_EQ(nullptr, queue.DoipPopFrontQueue());

    hozon::netaos::diag::doip_cache_data_t* cache_data1 = new hozon::netaos::diag::doip_cache_data_t;
    cache_data1->data = new char[4];
    queue.DoipInsertQueue(cache_data1);
    hozon::netaos::diag::doip_cache_data_t* cache;
    EXPECT_EQ(cache_data1, (cache = queue.DoipPopFrontQueue()));
    delete[] cache->data;
    delete cache;


    cache_data1 = new hozon::netaos::diag::doip_cache_data_t;
    cache_data1->data = new char[4];
    queue.DoipInsertQueue(cache_data1);
    hozon::netaos::diag::doip_cache_data_t* cache_data2 = new hozon::netaos::diag::doip_cache_data_t;
    cache_data2->data = new char[4];
    queue.DoipInsertQueue(cache_data2);
    EXPECT_EQ((uint32_t)2, queue.DoipQueueSize());

    EXPECT_EQ(cache_data1, (cache = queue.DoipPopFrontQueue()));
    delete[] cache->data;
    delete cache;
}
TEST(DoipDataQueue, DoipClearQueue) {
    hozon::netaos::diag::DoipDataQueue queue;
    EXPECT_EQ((uint32_t)0, queue.DoipQueueSize());
    
    hozon::netaos::diag::doip_cache_data_t* cache_data1 = new hozon::netaos::diag::doip_cache_data_t;
    cache_data1->data = new char[4];
    queue.DoipInsertQueue(cache_data1);
    EXPECT_EQ((uint32_t)1, queue.DoipQueueSize());

    hozon::netaos::diag::doip_cache_data_t* cache_data2 = new hozon::netaos::diag::doip_cache_data_t;
    cache_data2->data = new char[4];
    queue.DoipInsertQueue(cache_data2);
    EXPECT_EQ((uint32_t)2, queue.DoipQueueSize());

    queue.DoipClearQueue();
    EXPECT_EQ((uint32_t)0, queue.DoipQueueSize());

    cache_data1 = new hozon::netaos::diag::doip_cache_data_t;
    cache_data1->data = new char[4];
    queue.DoipInsertQueue(cache_data1);
    EXPECT_EQ((uint32_t)1, queue.DoipQueueSize());
    
    queue.DoipClearQueue();
    EXPECT_EQ((uint32_t)0, queue.DoipQueueSize());
}
