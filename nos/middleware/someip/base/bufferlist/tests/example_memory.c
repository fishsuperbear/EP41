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

#include "memory.h"
#include "list.h"


int main(int argc, char* argv[])
{
    int error = 0;
    mem_map_t* mem_map = NULL;
    mem_seg_t* segment_1 = NULL;
    mem_seg_t* segment_2 = NULL;
    mem_seg_t* segment_3 = NULL;
    memory_t* memory = memory_alloc();

    // #1. you must first obtain the permission before using the memory.
    {
        error = memory_acquire(memory);
        if (error) {
            printf("error: memory_acquire()\n");
            return error;
        }
    }

    // #2. insert a segment at the offset 0.  memory:[0123456789]
    {
        const char* str = "0123456789";

        // #2.1 construct a memory segment according size.
        segment_1 = mem_seg_new_form_alloc(strlen(str));
        memcpy(mem_seg_data(segment_1), str, mem_seg_size(segment_1));

        // #2.2 insert this memory segment into the offset 0 of the memory.
        error = memory_insert_segment(memory, 0, segment_1);
        if (error) {
            printf("error: memory_insert_segment()\n");
            return error;
        }
    }

    // #3. insert another segment in the middle of a segment. memory:[012] [abc] [3456789]
    {
        const char* str = "abc";

        // #3.1 construct a memory segment according size.
        segment_2 = mem_seg_new_form_alloc(strlen(str));
        memcpy(mem_seg_data(segment_2), str, mem_seg_size(segment_2));

        // #3.2 insert this memory segment into the offset 3 of the memory.
        error = memory_insert_segment(memory, 3, segment_2);
        if (error) {
            printf("error: memory_insert_segment()\n");
            return error;
        }
    }

    // #4. append a segment to the memory.  memory:[012] [abc] [3456789] [xyz]
    {
        const char* str = "xyz";

        // #4.1 construct a memory segment according size.
        segment_3 = mem_seg_new_form_alloc(strlen(str));
        memcpy(mem_seg_data(segment_3), str, mem_seg_size(segment_3));

        // #4.2 append this memory segment into the memory.
        error = memory_append_segment(memory, segment_3);
        if (error) {
            printf("error: memory_append_segment()\n");
            return error;
        }
    }

    // #5. remove a segment.  memory:[bc] [3456789] [xyz]
    {
        // #5.1 remove a memory segment of size 4 from memory offset 0
        error = memory_remove_segment(memory, 0, 4);
        if (error) {
            printf("error: memory_remove_segment()\n");
            return error;
        }
    }

    // #6. map memory.
    {
        // #6.1. get the memory with iovec_t type.
        // caution: once the memory_map interface is called, we can no longer manipulate this memory.
        mem_map = memory_map(memory);
        if (!mem_map) {
            printf("error: memory_map()\n");
            error = 1;
            return error;
        }

#ifndef BUFFER_DEBUG
        for (size_t i = 0; i< mem_map->iov_cnt; i++) {
            size_t size =  mem_map->iov_raw[i].iov_len;
            for (size_t j = 0; j < (mem_map->iov_raw[i].iov_len); j++) {
                    printf("[%c]", ((uint8_t*)(mem_map->iov_raw[i].iov_base))[j]);
            }
            printf("\n\n");
        }
#endif
    }

    // #7. unmap memory.
    {
        // #7.1. once the memory_unmap interface is called, we can read and write to this memory.
        memory_unmap(mem_map, memory);
    }

    // #8. release control of memory.
    {
        memory_release(memory);
    }

    // #9. destory memory.
    {
        memory_unref(memory);
    }

    // #10. memory for owner.
    {
        mem_seg_unref(segment_1);
        mem_seg_unref(segment_2);
        mem_seg_unref(segment_3);
    }

    return error;
}
