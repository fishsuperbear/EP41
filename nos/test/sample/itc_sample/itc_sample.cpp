#include "adf/include/itc/itc.h"
#include <vector>
#include <mutex>
#include <thread>
#include <iostream>
#include <unistd.h>

struct TestData {
    std::string sender;
    uint32_t seq;
};

std::mutex g_print_mtx;
void DataProcess(uint32_t index, hozon::netaos::adf::ITCDataType data) {
    std::shared_ptr<TestData> test_data = std::static_pointer_cast<TestData>(data);

    g_print_mtx.lock();
    std::cout << "Reader" << index <<  " recv data " << test_data->sender << ":" << test_data->seq << std::endl;
    g_print_mtx.unlock();
}

void PollReaderRoutine(std::shared_ptr<hozon::netaos::adf::ITCReader> reader, uint32_t index) {
    while (1) {
        hozon::netaos::adf::ITCDataType data = reader->Take(1000);
        if (data) {
            DataProcess(index, data);
        }
    }
} 

int main() {
    std::vector<std::shared_ptr<hozon::netaos::adf::ITCWriter>> _writers;
    std::vector<std::shared_ptr<hozon::netaos::adf::ITCReader>> _readers;

    // 1. init writers
    for (uint32_t i = 0; i < 5; ++i) {
        _writers.emplace_back(std::make_shared<hozon::netaos::adf::ITCWriter>());
        _writers[i]->Init("test_topic");
    }

    // 2. init readers
    // 2.1 poll mode
    for (uint32_t i = 0; i < 5; ++i) {
        _readers.emplace_back(std::make_shared<hozon::netaos::adf::ITCReader>());
        _readers[i]->Init("test_topic", 100);
        std::thread t(&PollReaderRoutine, _readers[i], i);
        t.detach();
    }

    // 2.2 callback mode
    for (uint32_t i = 5; i < 10; ++i) {
        _readers.emplace_back(std::make_shared<hozon::netaos::adf::ITCReader>());
        _readers[i]->Init("test_topic", std::bind(DataProcess, i, std::placeholders::_1), 100);
    }

    // 3. start to write
    while (1) {
        static int idx;
        ++idx;

        for (uint32_t i = 0; i < 5; ++i) {
            std::shared_ptr<TestData> test_data(new TestData);
            test_data->seq = idx;
            test_data->sender = std::string("Writer") + std::to_string(i);
            _writers[i]->Write(test_data);
        }

        usleep(10 * 1000);
    }
}