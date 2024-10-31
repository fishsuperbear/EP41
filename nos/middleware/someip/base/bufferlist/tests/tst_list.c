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
#include "list.h"

#define MLOGD(para, args...) printf(para, ##args)
#define MLOGE(para, args...) printf(para, ##args)


typedef struct data
{
    int aid;
    memory_list_t link;
}data_t;

typedef struct data_list
{
    int aid;
    memory_list_t* link;
}data_list_t;


void
print_list(data_list_t* list)
{
    data_t* node = NULL;
    data_t* tmp = NULL;
    memory_list_for_each_safe(node, tmp, list->link, link) {
        MLOGD("node: [%d]\n", node->aid);
   }
    MLOGD("\n\n");
}

void
get_node(data_list_t* list, data_t** data)
{
    data_t* node = NULL;
    data_t* tmp = NULL;
    memory_list_for_each_safe(node, tmp, list->link, link) {
        MLOGD("node: [%d]\n", node->aid);
   }
    MLOGD("\n\n");
}


int main() {
    data_t data1;
    data_t data2;
    data_t data3;
    data_t data4;
    data_t data5;
    data_t data6;
    data_t data7;
    data_t data8;

    data1.aid = 1;
    data2.aid = 2;
    data3.aid = 3;
    data4.aid = 4;
    data5.aid = 5;
    data6.aid = 6;
    data7.aid = 7;
    data8.aid = 8;

    data_list_t* memlist = (data_list_t*)calloc(1, sizeof(data_list_t));
    memlist->link = (memory_list_t*)calloc(1, sizeof(memory_list_t));
    memory_list_init(memlist->link);

    // insert a node.
    {
        memory_list_insert_tail(memlist->link, &data1.link);
        memory_list_insert_tail(memlist->link, &data2.link);
        memory_list_insert_tail(memlist->link, &data3.link);
        memory_list_insert_before(memlist->link, &data4.link);
        memory_list_insert_after(memlist->link, &data5.link);
        memory_list_insert_tail(memlist->link, &data7.link);
        print_list(memlist);
    }

    // insert a new node based on the old node.
    {
        data_t*node = NULL;
        data_t* tmp = NULL;
        memory_list_for_each_safe(node, tmp, memlist->link, link) {
            if (node->aid == 7) {
                memory_list_insert_after(&node->link, &data6.link);
            }
        }
        print_list(memlist);
    }

    // remove a node.
    {
        data_t*node = NULL;
        data_t* tmp = NULL;
        memory_list_for_each_safe(node, tmp, memlist->link, link) {
               if (node->aid == 3) {
                   memory_list_remove(&node->link);
               }
           }
        print_list(memlist);
    }


    memory_list_insert_head(memlist->link, &data8.link);
    print_list(memlist);

    {
        data_t* node_prev = NULL;
        data_t* node_next = NULL;
        node_prev = memory_container_of(memlist->link->prev->prev, node_prev, link);
        node_next = memory_container_of(memlist->link->prev, node_next, link);
        printf("aid:[%d][%d]\n", node_prev->aid, node_next->aid);
    }
    return 0;
}
