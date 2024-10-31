#include <unistd.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include "gtest/gtest.h"

#include "idl/generated/avm_methodPubSubTypes.h"
#include "idl/generated/chassis_methodPubSubTypes.h"
#include "idl/generated/sensor_reattachPubSubTypes.h"

#include "cm/include/method.h"
#include "log/include/default_logger.h"

#include "sensor/nvs_adapter/nvs_block_cuda_consumer.h"
#include "sensor/nvs_adapter/nvs_helper.h"
#include "sensor/nvs_consumer/CCudaConsumer.hpp"
#include "sensor/nvs_consumer/CIpcConsumerChannel.hpp"

using namespace hozon::netaos;
using namespace hozon::netaos::cm;
#define TEST_NVSTREAM_CHANNEL_ID 5
#define MAX_RETRY 50
#define MAX_SENSOR_NUM 11
// static int sensorList[MAX_SENSOR_NUM] = {0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11};
static int sensorList[MAX_SENSOR_NUM] = {0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11};

class NVS_ProducerTest : public ::testing::Test {
   protected:
    void SetUp() override {
        std::shared_ptr<sensor_reattachPubSubType> req_data_type = std::make_shared<sensor_reattachPubSubType>();
        std::shared_ptr<sensor_reattach_respPubSubType> resp_data_type = std::make_shared<sensor_reattach_respPubSubType>();
        req_data = std::make_shared<sensor_reattach>();
        resq_data = std::make_shared<sensor_reattach_resp>();

        _reattach_clint.reset(new hozon::netaos::cm::Client<sensor_reattach, sensor_reattach_resp>(req_data_type, resp_data_type));
        _reattach_clint->Init(0, "sensor_reattach");
    }

    void TearDown() override {
        printf("NVS_ProducerTest  TearDown\n");
        if (needDettach && current_sensor_id != -1) {
            req_data->isattach(false);
            req_data->sensor_id(current_sensor_id);
            req_data->index(TEST_NVSTREAM_CHANNEL_ID);
            _reattach_clint->RequestAndForget(req_data);
        }
        _reattach_clint->Deinit();
    }

    void testProducerConnect(int sensor_id) { testProducerConnect(sensor_id, nullptr); }

    void testProducerConnect(int sensor_id, desay::CCudaConsumer::PacketReadyCallback callback) {
        current_sensor_id = sensor_id;
        needDettach = true;
        int isReattachServerOnline;
        for (int i = 0; i < MAX_RETRY; i++) {
            isReattachServerOnline = _reattach_clint->WaitServiceOnline(500);
        }
        ASSERT_EQ(isReattachServerOnline, 0) << "Nvs_Producer is offline!!! Test Failed!!";

        SensorInfo sensor_info;
        sensor_info.id = sensor_id;
        std::shared_ptr<hozon::netaos::desay::CIpcConsumerChannel> _consumer = std::make_shared<hozon::netaos::desay::CIpcConsumerChannel>(
            nv::NVSHelper::GetInstance().sci_buf_module, nv::NVSHelper::GetInstance().sci_sync_module, &sensor_info, hozon::netaos::desay::CUDA_CONSUMER, sensor_id, TEST_NVSTREAM_CHANNEL_ID);
        ASSERT_NE(_consumer, nullptr);

        ASSERT_EQ(_consumer->CreateBlocks(nullptr), NVSIPL_STATUS_OK);

        desay::ConsumerConfig consumer_config{false, true};
        _consumer->SetConsumerConfig(consumer_config);

        req_data->isattach(true);
        req_data->sensor_id(sensor_id);
        req_data->index(TEST_NVSTREAM_CHANNEL_ID);
        _reattach_clint->RequestAndForget(req_data);
        //TODO add ConnectWithTimeOut
        ASSERT_EQ(_consumer->Connect(), NVSIPL_STATUS_OK);

        ASSERT_EQ(_consumer->InitBlocks(), NVSIPL_STATUS_OK);

        ASSERT_EQ(_consumer->Reconcile(), NVSIPL_STATUS_OK);
        if (callback != nullptr) {
            static_cast<desay::CCudaConsumer*>(_consumer->m_upConsumer.get())->SetOnPacketCallback(callback);
        }
        _consumer->Start();
        if (callback != nullptr) {
            std::this_thread::sleep_for(std::chrono::seconds(2));
            if (sensor_id < 8) {
                ASSERT_GE(framecount[sensor_id], 15);//7v
            } else {
                ASSERT_GE(framecount[sensor_id], 45);//4v
            }
        }
        _consumer->Stop();

        req_data->isattach(false);
        req_data->sensor_id(sensor_id);
        req_data->index(TEST_NVSTREAM_CHANNEL_ID);
        _reattach_clint->RequestAndForget(req_data);
        needDettach = false;
    }

   public:
    void NVSReadyCallback(int sensor_id, std::shared_ptr<desay::DesayCUDAPacket> packet) {
        if (packet->data_size > 0) {
            framecount[sensor_id]++;
        }
    }

   public:
    int framecount[MAX_SENSOR_NUM];
    bool needDettach = false;
    int current_sensor_id = -1;
    std::unique_ptr<hozon::netaos::cm::Client<sensor_reattach, sensor_reattach_resp>> _reattach_clint;

    std::shared_ptr<sensor_reattach> req_data;
    std::shared_ptr<sensor_reattach_resp> resq_data;
};

TEST_F(NVS_ProducerTest, NVS_ProducerOnline) {
    needDettach = false;
    int isReattachServerOnline;
    for (int i = 0; i < MAX_RETRY; i++) {
        isReattachServerOnline = _reattach_clint->WaitServiceOnline(500);
    }
    ASSERT_EQ(isReattachServerOnline, 0) << "Nvs_Producer is offline!!! Test Failed!!";
}

#define NVSTREAM_CONNECT_TEST_CASE(CameraNumber)                      \
    TEST_F(NVS_ProducerTest, NVSTREAM_CONNECT_CAMERA##CameraNumber) { \
        testProducerConnect(CameraNumber);                            \
    }

#define NVSTREAM_DATA_TEST_CASE(CameraNumber)                                                                                         \
    TEST_F(NVS_ProducerTest, NVSTREAM_DATA_CAMERA##CameraNumber) {                                                                    \
        testProducerConnect(CameraNumber, std::bind(&NVS_ProducerTest::NVSReadyCallback, this, CameraNumber, std::placeholders::_1)); \
    }

NVSTREAM_CONNECT_TEST_CASE(0)
NVSTREAM_CONNECT_TEST_CASE(1)
NVSTREAM_CONNECT_TEST_CASE(2)

NVSTREAM_CONNECT_TEST_CASE(4)
NVSTREAM_CONNECT_TEST_CASE(5)
NVSTREAM_CONNECT_TEST_CASE(6)
NVSTREAM_CONNECT_TEST_CASE(7)

NVSTREAM_CONNECT_TEST_CASE(8)
NVSTREAM_CONNECT_TEST_CASE(9)
NVSTREAM_CONNECT_TEST_CASE(10)
NVSTREAM_CONNECT_TEST_CASE(11)

NVSTREAM_DATA_TEST_CASE(0)
NVSTREAM_DATA_TEST_CASE(1)
NVSTREAM_DATA_TEST_CASE(2)

NVSTREAM_DATA_TEST_CASE(4)
NVSTREAM_DATA_TEST_CASE(5)
NVSTREAM_DATA_TEST_CASE(6)
NVSTREAM_DATA_TEST_CASE(7)

NVSTREAM_DATA_TEST_CASE(8)
NVSTREAM_DATA_TEST_CASE(9)
NVSTREAM_DATA_TEST_CASE(10)
NVSTREAM_DATA_TEST_CASE(11)

// TEST_F(NVS_ProducerTest, NVSTREAM_CONNECT) {

//     int isReattachServerOnline;
//     for (int i = 0; i < MAX_RETRY; i++) {
//         isReattachServerOnline = _reattach_clint->WaitServiceOnline(500);
//     }
//     ASSERT_EQ(isReattachServerOnline, 0);
//     // for(int i =0)
//     for (int i = 0; i < MAX_SENSOR_NUM; i++) {  //do nvstream connect test
//         printf("Sensor[%d] start connect\n", sensorList[i]);
//         SensorInfo sensor_info;
//         sensor_info.id = sensorList[i];
//         // GTEST_LOG_ ("test");
//         std::shared_ptr<hozon::netaos::desay::CIpcConsumerChannel> _consumer = std::make_shared<hozon::netaos::desay::CIpcConsumerChannel>(
//             nv::NVSHelper::GetInstance().sci_buf_module, nv::NVSHelper::GetInstance().sci_sync_module, &sensor_info, hozon::netaos::desay::CUDA_CONSUMER, sensorList[i], TEST_NVSTREAM_CHANNEL_ID);
//         ASSERT_NE(_consumer, nullptr);
//         printf("Sensor[%d] start connect2\n", sensorList[i]);
//         ASSERT_EQ(_consumer->CreateBlocks(nullptr), NVSIPL_STATUS_OK);
//         printf("Sensor[%d] start connect3\n", sensorList[i]);
//         desay::ConsumerConfig consumer_config{false, true};
//         _consumer->SetConsumerConfig(consumer_config);
//         printf("Sensor[%d] start connect4\n", sensorList[i]);
//         req_data->isattach(true);
//         req_data->sensor_id(sensorList[i]);
//         req_data->index(TEST_NVSTREAM_CHANNEL_ID);
//         _reattach_clint->RequestAndForget(req_data);
//         //TODO add ConnectWithTimeOut
//         printf("Sensor[%d] is connecting\n", sensorList[i]);
//         ASSERT_EQ(_consumer->ConnectWithTimeOut(500), NVSIPL_STATUS_OK) << "Connect to Producer Failed!";

//         ASSERT_EQ(_consumer->InitBlocks(), NVSIPL_STATUS_OK);

//         ASSERT_EQ(_consumer->Reconcile(), NVSIPL_STATUS_OK);

//         _consumer->Start();

//         _consumer->Stop();

//         req_data->isattach(false);
//         req_data->sensor_id(sensorList[i]);
//         req_data->index(TEST_NVSTREAM_CHANNEL_ID);
//         _reattach_clint->RequestAndForget(req_data);
//         printf("Sensor[%d] connect success\n", sensorList[i]);
//     }
// }

int main(int argc, char* argv[]) {
    nv::NVSHelper::GetInstance().Init();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
