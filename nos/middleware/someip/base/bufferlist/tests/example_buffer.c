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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "buffer.h"

int main(int argc, char* argv[])
{
    int error = 0;
    size_t write_pos = 0;
    size_t read_pos = 0;

    buffer_t* buffer = NULL;
    memory_t* memory = NULL;
    mem_map_t* mem_map = NULL;

    mem_seg_t* segment = NULL;
    uint8_t data_write[6] = {1,2,3,4,5,6};
    void* data_read = calloc(1, 64);


// buffer write.
{
    // #1. new a buffer.
    buffer = buffer_new();

    // #2. calloc a memory for write.
    memory = memory_alloc();

    // #3.attach this memory to the buffer for write.
    error = buffer_attach_memory(buffer, memory);
    if (error) {
        printf("error: buffer_attach_memory()\n");
        return error;
    }

    // #4. write data to the buffer.
    error = buffer_write(buffer, (const void*)data_write, sizeof(data_write));
    if (error) {
        printf("error: buffer_write()\n");
        return error;
    }

    // #5. write data to the buffer according to the write_pos.
    write_pos = buffer_get_write_pos(buffer);
    error = buffer_write_at(buffer, write_pos, (const void*)data_write, sizeof(data_write));
    if (error) {
        printf("error: buffer_write_at()\n");
        return error;
    }

    // #6. detach the memory from buffer after the data operation is completed.
    error = buffer_detach_memory(buffer);
    if (error) {
        printf("error: buffer_detach_memory()\n");
        return error;
    }

    // #7. get the memory with iovec_t type.
    // caution: once the memory_map interface is called, we can no longer manipulate this memory.
    mem_map = memory_map(memory);
    if (!mem_map) {
        printf("error: buffer_write_at()\n");
        error = 1;
        return error;
    }

#ifndef BUFFER_DEBUG
        for (size_t i = 0; i< mem_map->iov_cnt; i++) {
            size_t size =  mem_map->iov_raw[i].iov_len;
            for (size_t j = 0; j < (mem_map->iov_raw[i].iov_len); j++) {
                    printf("[%u]", ((uint8_t*)(mem_map->iov_raw[i].iov_base))[j]);
            }
            printf("\n\n");
        }
#endif

        // #8. once the memory_unmap interface is called, we can read and write to this memory.
        memory_unmap(mem_map, memory);

        // #9. destory this memory for free memory.
        memory_unref(memory);

        // #10. free buffer.
        buffer_free(&buffer);
}

printf("\nbuffer read. \n");
// buffer read.
{
    // construct a memory for test read.
    {
        uint8_t data_segment[64];
        memory = memory_alloc();
        segment = mem_seg_new_form_alloc(64);

        memset(data_segment, 1, sizeof(data_segment));
        memcpy(mem_seg_data(segment), data_segment, mem_seg_size(segment));
        memory_insert_segment(memory, 0, segment);
    }


    // #1. new a buffer.
    buffer = buffer_new();

    // #2.attach this memory to the buffer for read.
    error = buffer_attach_memory(buffer, memory);
    if (error) {
        printf("error: buffer_attach_memory()\n");
        return error;
    }

    // #3. read data from the buffer.
    error = buffer_read(buffer, data_read, 1);
    if (error) {
        printf("error: buffer_read()\n");
        return error;
    }
#ifndef BUFFER_DEBUG
        for (int i = 0; i<64; i++) {
            printf("[%u]", ((uint8_t*)data_read)[i]);
        }
        printf("\n\n");
#endif

        // #4. read data from the buffer according the read position.
        read_pos = buffer_get_read_pos(buffer);
        error = buffer_read_at(buffer, read_pos, (uint8_t*)data_read + 1, 10);
        if (error) {
            printf("error: buffer_read_at()\n");
            return error;
        }
#ifndef BUFFER_DEBUG
        for (int i = 0; i<64; i++) {
            printf("[%u]", ((uint8_t*)data_read)[i]);
        }
        printf("\n\n");
#endif

        // #5. detach the memory from the buffer.
        error = buffer_detach_memory(buffer);
        if (error) {
            printf("error: buffer_detach_memory()\n");
            return error;
        }

        // #6. destory this memory for free memory.
        memory_unref(memory);

        // #7. free buffer.
        buffer_free(&buffer);

        // #8. memory for owner.
        mem_seg_unref(segment);
        free(data_read);
}

    return error;
}
