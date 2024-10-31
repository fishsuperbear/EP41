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
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include "memory.h"
#include "list.h"

static int UT_MEMORY_ALLOC = 1;
static int UT_MEMORY_ACQUIRE = 2;
static int UT_MEMORY_INSERT_SEGMENT_MIDDLE = 3;
static int UT_MEMORY_REMOVE = 4;
static int UT_MEMORY_DESTORY = 5;
static int UT_MEMORY_SEGMENT_DESTORY = 6;

struct mem_seg {
    atomic_int ref_cnt;
    size_t size;
    free_func_t free_func;
    void* data_ctx;
    void* data;
    memory_list_t link;
};

struct memory {
    atomic_int ref_cnt;
    atomic_bool mem_is_acquire;
    bool mem_is_map;
    memory_list_t* link;
};


int main(int argc, char* argv[])
{

    memory_t* memory = NULL;

    mem_seg_t* segment_1 = NULL;
    mem_seg_t* segment_2 = NULL;
    mem_seg_t* segment_3 = NULL;
    mem_seg_t* segment_4 = NULL;
    mem_seg_t* segment_5 = NULL;
    mem_seg_t* segment_6 = NULL;
    mem_seg_t* segment_7 = NULL;
    mem_seg_t* segment_8 = NULL;

    // case: memory_alloc/memory_destory
    printf("case 1: \n");
    {
        int ret = 0;

        memory_t* memory = memory_alloc();
        if (memory && memory->link) {
            if (memory->ref_cnt == 1 && memory->mem_is_map == false) {
                printf("\033[1;40;32mUT[%d]: memory_alloc() success\n\033[0m", UT_MEMORY_ALLOC);
            }
        }
        else {
            printf("\033[1;40;31mUT[%d]: memory_alloc() error. \n\033[0m", UT_MEMORY_ALLOC);
        }


        {
            ret = memory_acquire(memory);
            if (ret) {
                goto EXIT_ACQUIRE;
            }


            ret = memory_acquire(memory);
            if (!ret) {
                goto EXIT_ACQUIRE;
            }

            memory_release(memory);

            ret = memory_acquire(memory);
            if (ret) {
                goto EXIT_ACQUIRE;
            }

            ret = 0;
            EXIT_ACQUIRE:
                if (ret) {
                    printf("\033[1;40;31mUT[%d]: memory_acquire() error. \n\033[0m", UT_MEMORY_ACQUIRE);
                }
                else {
                    printf("\033[1;40;32mUT[%d]: memory_acquire() success\n\033[0m", UT_MEMORY_ACQUIRE);
                }
        }

        {
            memory_unref(memory);
            printf("\033[1;40;32mUT[%d]: memory_destory() success\n\033[0m", UT_MEMORY_DESTORY);
        }

    }

    printf("\ncase 2: \n");
    // memory_insert_segment
    {
        int ret;
        int i;
        size_t size;
        int list_length;

        const char* arr_0 = "012345";
        const char* arr_1 = "abcdefgh";
        const char* arr_2 = "6789";
        const char* arr_3 = "hello world";
        const char* arr_4 = "xyz";
        const char* arr_5 = "aoe";
        const char* arr_6 = "yuU";
        const char* arr_7 = "bpm";
        const char* arr_8 = "fdt";



        memory = memory_alloc();

        // check mem_seg_new_form_alloc().
        {
            ret = 0;
            size = 10;

            segment_1 = mem_seg_new_form_alloc(size);
            if (!segment_1) {
                ret = 1;
                goto EXIT_ALLOC;
            }

            if (1 != segment_1->ref_cnt) {
                ret = 1;
                goto EXIT_ALLOC;
            }

            if (size != segment_1->size) {
                ret = 1;
                goto EXIT_ALLOC;
            }

            if (NULL == segment_1->data) {
                ret = 1;
                goto EXIT_ALLOC;
            }

            const char* arr = "0123456789";
            memcpy(segment_1->data, arr, segment_1->size);

        EXIT_ALLOC:
            if (ret) {
                printf("\033[1;40;31mUT[%d]: mem_seg_new_form_alloc() error. \n\033[0m", UT_MEMORY_ALLOC);
            }
            else {
                printf("\033[1;40;32mUT[%d]: mem_seg_new_form_alloc() success\n\033[0m", UT_MEMORY_ALLOC);
            }
        }

{
        // [0123456789]
        // memory_append_segment or insert a segment in empty list.
        {
             // ret = memory_append_segment(memory, segment_1);
            ret = memory_insert_segment(memory, 0, segment_1);
            if (ret) {
                goto EXIT_INSERT;
            }

            list_length = memory_list_length(memory->link);
            if (1 != list_length) {
                goto EXIT_INSERT;
            }
        }


        // [012345] [abcdefgh] [6789]
        // insert the memory segment in the middle of other memory segment
        {
            segment_2 = mem_seg_new_form_alloc(8);
            memcpy(segment_2->data, arr_1, segment_2->size);

            ret = memory_insert_segment(memory, 6, segment_2);
            if (ret) {
                goto EXIT_INSERT;
            }

            list_length = memory_list_length(memory->link);
            if (3 != list_length) {
                goto EXIT_INSERT;
            }

            mem_seg_t* node = NULL;
            memory_list_for_each(node, memory->link, link) {
                static int count = 0;
                if (count == 0) {
                    ret = strncmp(arr_0, (char*)(node->data), node->size);
                    if (ret) {
                        goto EXIT_INSERT;
                    }
                }
                else if (count == 1) {
                    ret = strncmp(arr_1, (char*)(node->data), node->size);
                    if (ret) {
                        goto EXIT_INSERT;
                    }
                }
                else if (count == 2) {
                    ret = strncmp(arr_2, (char*)(node->data), node->size);
                    if (ret) {
                        goto EXIT_INSERT;
                    }
                }
                count++;

#ifdef MEM_DEBUG
                for (i = 0; i < node->size; i++) {
                    printf("[%c]", ((char*)(node->data))[i]);
                }
                printf("\n\n");
#endif
            }

        }

        // [012345] [abcdefgh] [6789] [hello world]
        // insert the memory segment after the given element.
        {
            segment_3 = mem_seg_new_form_alloc(11);
            memcpy(segment_3->data, arr_3, segment_3->size);

            ret = memory_insert_segment(memory, 18, segment_3);
            if (ret) {
                goto EXIT_INSERT;
            }

            list_length = memory_list_length(memory->link);
            if (4 != list_length) {
                ret = 1;
                goto EXIT_INSERT;
            }

            mem_seg_t* node = NULL;
            memory_list_for_each(node, memory->link, link) {
                static int count = 0;
                if (count == 0) {
                    ret = strncmp(arr_0, (char*)(node->data), node->size);
                    if (ret) {
                        goto EXIT_INSERT;
                    }
                }
                else if (count == 1) {
                    ret = strncmp(arr_1, (char*)(node->data), node->size);
                    if (ret) {
                        goto EXIT_INSERT;
                    }
                }
                else if (count == 2) {
                    ret = strncmp(arr_2, (char*)(node->data), node->size);
                    if (ret) {
                        goto EXIT_INSERT;
                    }
                }
                else if (count == 3) {
                    ret = strncmp(arr_3, (char*)(node->data), node->size);
                    if (ret) {
                        goto EXIT_INSERT;
                    }
                }
                count++;

#ifdef MEM_DEBUG
                for (i = 0; i < node->size; i++) {
                        printf("[%c]]", ((char*)(node->data))[i]);
                }
                printf("\n\n");
#endif
            }
        }
        // [xyz] [012345] [abcdefgh] [6789] [hello world]
        // insert the memory segment at the head.
        {
            segment_4 = mem_seg_new_form_alloc(3);
            memcpy(segment_4->data, arr_4, segment_4->size);

            ret = memory_insert_segment(memory, 0, segment_4);
            if (ret) {
                goto EXIT_INSERT;
            }

            list_length = memory_list_length(memory->link);
            if (5 != list_length) {
                ret = 1;
                goto EXIT_INSERT;
            }

            mem_seg_t* node = NULL;
            memory_list_for_each(node, memory->link, link) {
                static int count = 0;
                if (count == 1) {
                    ret = strncmp(arr_0, (char*)(node->data), node->size);
                    if (ret) {
                        goto EXIT_INSERT;
                    }
                }
                else if (count == 2) {
                    ret = strncmp(arr_1, (char*)(node->data), node->size);
                    if (ret) {
                        goto EXIT_INSERT;
                    }
                }
                else if (count == 3) {
                    ret = strncmp(arr_2, (char*)(node->data), node->size);
                    if (ret) {
                        goto EXIT_INSERT;
                    }
                }
                else if (count == 4) {
                    ret = strncmp(arr_3, (char*)(node->data), node->size);
                    if (ret) {
                        goto EXIT_INSERT;
                    }
                }
                else if (count == 0) {
                    ret = strncmp(arr_4, (char*)(node->data), node->size);
                    if (ret) {
                        goto EXIT_INSERT;
                    }
                }
                count++;

#ifdef MEM_DEBUG
                for (i = 0; i < node->size; i++) {
                        printf("[%c]]", ((char*)(node->data))[i]);
                }
                printf("\n\n");
#endif
            }
        }

        // [xyz] [012345] [abcdefgh] [6789] [hello world] [aoe] [yuU] [bpm] [fdt]
        {

            segment_5 = mem_seg_new_form_alloc(3);
            memcpy(segment_5->data, arr_5, segment_5->size);

            segment_6 = mem_seg_new_form_alloc(3);
            memcpy(segment_6->data, arr_6, segment_6->size);

            segment_7 = mem_seg_new_form_alloc(3);
            memcpy(segment_7->data, arr_7, segment_7->size);

            segment_8 = mem_seg_new_form_alloc(3);
            memcpy(segment_8->data, arr_8, segment_8->size);

            ret = memory_append_segment(memory, segment_5);
            ret |= memory_append_segment(memory, segment_6);
            ret |= memory_append_segment(memory, segment_7);
            ret |= memory_append_segment(memory, segment_8);
            if (ret) {
                goto EXIT_INSERT;
            }


            mem_seg_t* node = NULL;
            memory_list_for_each(node, memory->link, link) {
#ifdef MEM_DEBUG
                for (i = 0; i < node->size; i++) {
                        printf("[%c]", ((char*)(node->data))[i]);
                }
                printf("\n\n");
#endif
            }
        }

        // [xyz] [0] [89] [hello world] [aoe] [yuU] [bpm] [fdt]
        {
            memory_remove_segment(memory, 4, 15);

            mem_seg_t* node = NULL;
            memory_list_for_each(node, memory->link, link) {
#ifdef MEM_DEBUG
                for (i = 0; i < node->size; i++) {
                        printf("[%c]", ((char*)(node->data))[i]);
                }
                printf("\n\n");
#endif
            }
        }

        // [xy] [9] [hello world] [aoe] [yuU] [bpm] [fdt]
        {
            memory_remove_segment(memory, 2, 3);

            mem_seg_t* node = NULL;
            memory_list_for_each(node, memory->link, link) {
#ifdef MEM_DEBUG
                for (i = 0; i < node->size; i++) {
                        printf("[%c]", ((char*)(node->data))[i]);
                }
                printf("\n\n");
#endif
            }
        }

        // [9] [hello world] [aoe] [yuU] [bpm] [fdt]
        {
            memory_remove_segment(memory, 0, 2);

            mem_seg_t* node = NULL;
            memory_list_for_each(node, memory->link, link) {
#ifdef MEM_DEBUG
                for (i = 0; i < node->size; i++) {
                        printf("[%c]", ((char*)(node->data))[i]);
                }
                printf("\n\n");
#endif
            }
        }

        // [9] [hello world] [aoe] [yuU] [bpm]
        {
            memory_remove_segment(memory, 21, 3);

            mem_seg_t* node = NULL;
            memory_list_for_each(node, memory->link, link) {
#ifdef MEM_DEBUG
                for (i = 0; i < node->size; i++) {
                        printf("[%c]", ((char*)(node->data))[i]);
                }
                printf("\n\n");
#endif
            }
        }

        // [hello world] [aoe] [yuU] [bpm]
        {
            memory_remove_segment(memory, 0, 1);

            mem_seg_t* node = NULL;
            memory_list_for_each(node, memory->link, link) {
#ifdef MEM_DEBUG
                for (i = 0; i < node->size; i++) {
                        printf("[%c]", ((char*)(node->data))[i]);
                }
                printf("\n\n");
#endif
            }
        }

        {
            mem_seg_t* node = NULL;
            memory_list_for_each(node, memory->link, link) {
                static int count = 0;
                if (count == 1) {
                    ret = strncmp(arr_5, (char*)(node->data), node->size);
                    if (ret) {
                        goto EXIT_REMOVE;
                    }
                }
                else if (count == 2) {
                    ret = strncmp(arr_6, (char*)(node->data), node->size);
                    if (ret) {
                        goto EXIT_REMOVE;
                    }
                }
                else if (count == 3) {
                    ret = strncmp(arr_7, (char*)(node->data), node->size);
                    if (ret) {
                        goto EXIT_REMOVE;
                    }
                }
                else if (count == 0) {
                    ret = strncmp(arr_3, (char*)(node->data), node->size);
                    if (ret) {
                        goto EXIT_REMOVE;
                    }
                }
                count++;

#ifdef MEM_DEBUG
                for (i = 0; i < node->size; i++) {
                        printf("[%c]]", ((char*)(node->data))[i]);
                }
                printf("\n\n");
#endif
            }
        }

        EXIT_INSERT:
            if (ret) {
                printf("\033[1;40;31mUT[%d]: memory_insert_segment() error. \n\033[0m", UT_MEMORY_INSERT_SEGMENT_MIDDLE);
            }
            else {
                printf("\033[1;40;32mUT[%d]: memory_insert_segment() success\n\033[0m", UT_MEMORY_INSERT_SEGMENT_MIDDLE);
            }

        EXIT_REMOVE:
            if (ret) {
                printf("\033[1;40;31mUT[%d]: memory_remove_segment() error. \n\033[0m", UT_MEMORY_REMOVE);
            }
            else {
                printf("\033[1;40;32mUT[%d]: memory_remove_segment() success\n\033[0m", UT_MEMORY_REMOVE);
            }
}


    }

    // check memory_destory().
    {
        memory_unref(memory);
        printf("\033[1;40;32mUT[%d]: memory_unref() success\n\033[0m", UT_MEMORY_DESTORY);
    }


    printf("\ncase 3:\n");
    // free memory segment.
    {
        mem_seg_unref(segment_1);
        mem_seg_unref(segment_2);
        mem_seg_unref(segment_3);
        mem_seg_unref(segment_4);
        mem_seg_unref(segment_5);
        mem_seg_unref(segment_6);
        mem_seg_unref(segment_7);
        mem_seg_unref(segment_8);
        printf("\033[1;40;32mUT[%d]: mem_seg_unref() success\n\033[0m", UT_MEMORY_SEGMENT_DESTORY);
    }

    return 0;
}
