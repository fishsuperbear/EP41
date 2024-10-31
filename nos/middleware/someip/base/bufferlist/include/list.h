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
#ifndef __AP_MEMORY_LIST__
#define __AP_MEMORY_LIST__


#ifdef  __cplusplus
extern "C" {
#endif


typedef struct memory_list {
    struct memory_list *prev;
    struct memory_list *next;
} memory_list_t;


/* get the offset address of member in type  */
#define memory_offset(type, member) ((size_t) &((type *)0)->member)
/* get the actual pointer address */
#define memory_container_of(ptr, type, member) \
    (typeof(type))((char *)(ptr) - \
            memory_offset(typeof(*type), member))


/* traversing the lis */
#define memory_list_for_each(pos, head, member) \
    for (pos = memory_container_of((head)->next, pos, member); \
         &pos->member != (head); \
         pos = memory_container_of(pos->member.next, pos, member))

#define memory_list_for_each_safe(pos, tmp, head, member) \
    for (pos = memory_container_of((head)->next, pos, member), \
         tmp = memory_container_of((pos)->member.next, tmp, member); \
         &pos->member != (head);                    \
         pos = tmp,                         \
         tmp = memory_container_of(pos->member.next, tmp, member))

/* init list */
static inline void
memory_list_init(memory_list_t *list)
{
    list->next = list;
    list->prev = list;
}

/* insert a new node to list */
static inline void
memory_list_insert(memory_list_t *list, memory_list_t *node)
{
    node->prev = list;
    node->next = list->next;
    list->next = node;
    node->next->prev = node;
}

/* insert a new node to list */
static inline void
memory_list_insert_before(memory_list_t *list, memory_list_t *node)
{
    node->prev = list->prev;
    node->next = list;
    node->next->prev = node;
    node->prev->next = node;
}

/* insert a new node to list */
static inline void
memory_list_insert_after(memory_list_t *list, memory_list_t *node)
{
    node->prev = list;
    node->next = list->next;
    node->next->prev = node;
    node->prev->next = node;
}

/* insert a new node to head */
static inline void
memory_list_insert_head(memory_list_t *list, memory_list_t *node)
{
    memory_list_insert_after(list, node);
}

/* insert a new node to tail */
static inline void
memory_list_insert_tail(memory_list_t *list, memory_list_t *node)
{
    memory_list_insert_before(list, node);
}

/* remove a node from list */
static inline void
memory_list_remove(memory_list_t *node)
{
    node->prev->next = node->next;
    node->next->prev = node->prev;
    node->next = NULL;
    node->prev = NULL;
}

/* get the length of list */
static inline int
memory_list_length(const memory_list_t *list)
{
    int count;
    memory_list_t *n;

    count = 0;
    n = list->next;
    while (n != list) {
        n = n->next;
        count++;
    }

    return count;
}


/* whether the list is empty */
static inline int
memory_list_empty(const memory_list_t *list)
{
    return list->next == list;
}



#ifdef  __cplusplus
} // extern "C"
#endif // __cplusplus
#endif // __AP_MEMORY_LIST__
/* EOF */
