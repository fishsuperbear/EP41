/**

* Copyright @ 2020 - 2027 iAuto Software(Shanghai) Co., Ltd.

* All Rights Reserved.

*

* Redistribution and use in source and binary forms, with or without

* modification, are NOT permitted except as agreed by

* iAuto Software(Shanghai) Co., Ltd.

*

* Unless required by applicable law or agreed to in writing, software

* distributed under the License is distributed on an "AS IS" BASIS,

* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

*/
#include <iostream>
#include <vector>
#include <chrono>
#include <stdlib.h>
#include <string.h>


#include "buffer.h"

#define TEST_PERFORMENCT_COUNT 102400

struct buffer {
    bool is_init;
    size_t read_pos;
    size_t write_pos;
    size_t size;
    size_t capacity;
    memory_t* memory;
};



int main()
{
    {
        uint8_t element = 1;
        std::vector<uint8_t> vector;
        auto start_time = std::chrono::steady_clock::now();
        for (int i = 0; i< TEST_PERFORMENCT_COUNT; i++) {
            vector.push_back(1);
        }
        auto end_time = std::chrono::steady_clock::now();

        std::chrono::duration<double, std::micro> elapsed_time = end_time-start_time;
        std::chrono::duration<double, std::nano> elapsed_time_ns = end_time-start_time;
        std::cout <<"vector elapsed time: " << elapsed_time.count() << ": " << elapsed_time_ns.count() << std::endl;
    }

    {
        void* element = calloc(1, sizeof(uint8_t));

        buffer_t* buffer = buffer_new();
        memory_t* memory = memory_alloc();
        buffer_attach_memory(buffer, memory);

        auto start_time = std::chrono::steady_clock::now();
        for (int i = 0; i< TEST_PERFORMENCT_COUNT; i++) {
            buffer_write_at(buffer, buffer->write_pos, element, 1);
        }
        auto end_time = std::chrono::steady_clock::now();

        std::chrono::duration<double, std::micro> elapsed_time = end_time-start_time;
        std::chrono::duration<double, std::nano> elapsed_time_ns = end_time-start_time;
        std::cout <<"buffer elapsed time: " << elapsed_time.count() << ": " << elapsed_time_ns.count() << std::endl;

        free(element);
    }

    return 0;
}
