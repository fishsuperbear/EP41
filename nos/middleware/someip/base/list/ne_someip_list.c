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


# include "ne_someip_list.h"
# include "stdlib.h"
# include "string.h"
# include "ne_someip_log.h"

// define someip list
struct ne_someip_list
{
    struct ne_someip_list_element* head;  /* list head pointer */
    struct ne_someip_list_element* tail;  /* list tail pointer */
    int32_t size;                        /* list size */
    struct ne_someip_list_element* cursor;  /* pointer to elemenet operated recently */
    int32_t cursor_index;                  /* index of elemenet operated recently */
};

struct ne_someip_list_iterator
{
    ne_someip_list_t* list;
    ne_someip_list_element_t* cur_item;
    ne_someip_list_element_t* next_item;
};

static ne_someip_list_element_t* _ne_someip_list_sort_merge(ne_someip_list_element_t* l1, ne_someip_list_element_t* l2, ne_someip_base_compare_func func, void* user_data);

static ne_someip_list_t* _ne_someip_list_list_alloc();

static ne_someip_list_element_t* _ne_someip_list_element_alloc();

static ne_someip_list_iterator_t* _ne_someip_list_iterator_alloc();

static inline void _ne_someip_list_element_free(ne_someip_list_element_t* element);

static inline void _ne_someip_list_list_free(ne_someip_list_t* list);

static ne_someip_list_element_t* _ne_someip_list_cut(ne_someip_list_element_t* list, int32_t size);

static ne_someip_list_element_t* _ne_someip_list_sorted_by_merge_sort(ne_someip_list_element_t* list, ne_someip_base_compare_func cmp_func, void* user_data);

static inline ne_someip_list_t* _ne_someip_list_reset_cursor(ne_someip_list_t* list);

static ne_someip_list_t* _ne_someip_list_insert_before_element(ne_someip_list_t* list, ne_someip_list_element_t* element, void* data);

static ne_someip_list_element_t* _ne_someip_list_get_element(ne_someip_list_t* list, int32_t index);

static void _ne_someip_list_iterator_set_invalid(ne_someip_list_iterator_t* iterator);

ne_someip_list_t* ne_someip_list_create()
{
    ne_someip_list_t* list = _ne_someip_list_list_alloc();
    if (NULL == list) {
        ne_someip_log_error("[List] list alloc error");
        return NULL;
    }

    list->size = 0;
    list->head = NULL;
    list->tail = NULL;
    list->cursor = NULL;
    list->cursor_index = 0;

    return list;
}

ne_someip_list_t* ne_someip_list_append(ne_someip_list_t* list, void* data)
{
    if (NULL == list) {
        list = ne_someip_list_create();
        if (!list) {
            ne_someip_log_error("[List] list alloc error");
            return NULL;
        }
    }

    ne_someip_list_element_t* new_element = _ne_someip_list_element_alloc();
    if (NULL == new_element) {
        ne_someip_log_error("[List] elem alloc error");
        return NULL;
    }

    new_element->data = data;

    if (0 == list->size) {
        new_element->prev = NULL;
        new_element->next = NULL;
        list->head = new_element;
        list->tail = new_element;
    }
    else {
        new_element->prev = list->tail;
        new_element->next = NULL;
        list->tail->next = new_element;
        list->tail = new_element;
    }

    ++(list->size);
    list->cursor = list->head;
    list->cursor_index = 0;
    return list;
}

ne_someip_list_t* ne_someip_list_prepend(ne_someip_list_t* list, void* data)
{
    if (NULL == list) {
        ne_someip_log_error("[List] list is NULL");
        return NULL;
    }

    ne_someip_list_element_t* new_element = _ne_someip_list_element_alloc();
    if (NULL == new_element) {
        ne_someip_log_error("[List] elem alloc error");
        return NULL;
    }

    new_element->data = data;

    if (0 == list->size) {
        new_element->prev = NULL;
        new_element->next = NULL;
        list->head = new_element;
        list->tail = new_element;
    }
    else {
        new_element->prev = NULL;
        new_element->next = list->head;
        list->head->prev = new_element;
        list->head = new_element;
    }

    ++(list->size);
    list->cursor = list->head;
    list->cursor_index = 0;
    return list;
}

int32_t ne_someip_list_length(ne_someip_list_t* list) {
    if (!list) {
        ne_someip_log_error("[List] list is NULL");
        return -1;
    }
    return list->size;
}

void ne_someip_list_remove_all(ne_someip_list_t* list, ne_someip_base_free_func free_func) {
    if (!list) {
        ne_someip_log_error("[List] list is NULL");
        return ;
    }

    while (ne_someip_list_length(list) && list->head) {
        ne_someip_list_remove_by_elem(list, list->head, free_func);
    }

    list->head = NULL;
    list->tail = NULL;
    list->size = 0;
    _ne_someip_list_reset_cursor(list);
    return;
}

void ne_someip_list_destroy(ne_someip_list_t* list, ne_someip_base_free_func free_func)
{
    if (!list) {
        ne_someip_log_error("[List] list is NULL");
        return;
    }
    ne_someip_list_remove_all(list, free_func);
    _ne_someip_list_list_free(list);
}

ne_someip_list_t* ne_someip_list_remove_by_data(ne_someip_list_t* list, void* data, ne_someip_base_free_func free_func)
{
    if (!list) {
        ne_someip_log_error("[List] list is NULL");
        return NULL;
    }
    ne_someip_list_element_t* element = list->head;
    while (element) {
        if (element->data == data) {
            break;
        }
        element = element->next;
    }

    if (!element) {
        ne_someip_log_error("[List] elem is NULL");
        return NULL;
    }

    return ne_someip_list_remove_by_elem(list, element, free_func);
}

ne_someip_list_t* ne_someip_list_remove_by_elem(ne_someip_list_t* list, ne_someip_list_element_t* elem, ne_someip_base_free_func free_func)
{
    if (!list || !elem) {
        ne_someip_log_error("[List] list:%p, elem:%p is NULL", list, elem);
        return NULL;
    }

    if (list->size <= 0) {
        ne_someip_log_error("[List] list size:%d", list->size);
        return NULL;
    }

    if (elem->prev) {
        if (elem->prev->next == elem) {
            elem->prev->next = elem->next;
        }
        else {
            ne_someip_log_error("[List] elem state error");
            return NULL;
        }
    }

    if (elem->next) {
        if (elem->next->prev == elem) {
            elem->next->prev = elem->prev;
        }
        else {
            ne_someip_log_error("[List] elem state error");
            return NULL;
        }
    }

    if (list->head == elem) {
        list->head = elem->next;
    }

    if (list->tail == elem) {
        list->tail = elem->prev;
    }

    --(list->size);
    _ne_someip_list_reset_cursor(list);

    if (free_func) {
        (*free_func)(elem->data);
    }
    _ne_someip_list_element_free(elem);

    return list;
}

ne_someip_list_t* ne_someip_list_merge_list(ne_someip_list_t* dest_list, ne_someip_list_t* src_list)
{
    if (!dest_list || !src_list) {
        ne_someip_log_error("[List] dest_list:%p, src_list:%p", dest_list, src_list);
        return NULL;
    }

    ne_someip_list_element_t* src_list_element = ne_someip_list_first(src_list);
    while (src_list_element) {
        ne_someip_list_append(dest_list, src_list_element->data);
        src_list_element = src_list_element->next;
    }

    _ne_someip_list_reset_cursor(dest_list);

    return dest_list;
}

ne_someip_list_t* ne_someip_list_merge_element(ne_someip_list_t* first_list, ne_someip_list_element_t* second_element)
{
    if (!first_list || !second_element) {
        ne_someip_log_error("[List] first_list:%p, second_element:%p", first_list, second_element);
        return NULL;
    }

    ne_someip_list_append(first_list, second_element->data);

    _ne_someip_list_reset_cursor(first_list);

    return first_list;
}

ne_someip_list_t* ne_someip_list_move_list(ne_someip_list_t* dest_list, ne_someip_list_t* src_list)
{
    if (!dest_list || !src_list) {
        ne_someip_log_error("[List] dest_list:%p, src_list:%p", dest_list, src_list);
        return NULL;
    }

    if (0 == dest_list->size) {
        // 目标list为空list的时候
        dest_list->head = src_list->head;
        dest_list->tail = src_list->tail;
        dest_list->size = src_list->size;
    }
    else {
        // 目标list不为空list的时候
        // 将src list的element拼接到dest的尾部
        dest_list->tail->next = src_list->head;
        src_list->head->prev = dest_list->tail;
        // 更新tail和size
        dest_list->tail = src_list->tail;
        dest_list->size = dest_list->size + src_list->size;
    }

    // 清空src list
    src_list->head = NULL;
    src_list->tail = NULL;
    src_list->size = 0;

    // 重置游标
    _ne_someip_list_reset_cursor(dest_list);
    _ne_someip_list_reset_cursor(src_list);

    return dest_list;
}


int32_t ne_someip_list_get_position(ne_someip_list_t* list, ne_someip_list_element_t* elem)
{
    if (NULL == list || NULL == elem) {
        ne_someip_log_error("[List] list:%p, elem:%p", list, elem);
    }
    int32_t index = 0;
    ne_someip_list_element_t* list_element = list->head;
    while (list_element) {
        if (list_element == elem) {
            return index;
        }
        list_element = list_element->next;
        index++;
    }
    return -1;
}

ne_someip_list_element_t* ne_someip_list_get_element(ne_someip_list_t* list, int32_t index)
{
    if (NULL == list) {
        ne_someip_log_error("[List] list is empty");
        return NULL;
    }
    ne_someip_list_element_t* element = _ne_someip_list_get_element(list, index);
    return element;
}

void* ne_someip_list_get_data(ne_someip_list_t* list, int32_t index)
{
    if (NULL == list) {
        ne_someip_log_error("[List] list is NULL");
        return NULL;
    }
    ne_someip_list_element_t* element = _ne_someip_list_get_element(list, index);
    if (!element) {
        ne_someip_log_error("[List] elem is NULL");
        return NULL;
    }
    return element->data;
}

ne_someip_list_element_t* _ne_someip_list_get_element(ne_someip_list_t* list, int32_t index)
{
    if (NULL == list) {
        ne_someip_log_error("[List] list is NULL");
        return NULL;
    }

    if (0 >= list->size) {
        ne_someip_log_error("[List] list size error");
        return NULL;
    }

    if (index >= list->size) {
        ne_someip_log_error("[List] index out of range");
        return NULL;
    }

    // index between cursor_index and tail_index
    if (index >= list->cursor_index) {
        uint32_t tail_index = list->size - 1;
        if ((tail_index - index) >= (index - list->cursor_index)) {
            // from cursor_index to tail_index
            ne_someip_list_element_t* e = NULL;
            uint32_t loop_index = list->cursor_index;
            for (e = list->cursor; e != NULL; e = e->next) {
                if (loop_index == index) {
                    list->cursor_index = index;
                    list->cursor = e;
                    break;
                }
                ++loop_index;
            }
            return e;
        }
        else {
            // from tail_index back to cursor_index
            ne_someip_list_element_t* e = NULL;
            uint32_t loop_index = tail_index;
            for (e = list->tail; e != NULL; e = e->prev) {
                if (loop_index == index) {
                    list->cursor_index = index;
                    list->cursor = e;
                    break;
                }
                --loop_index;
            }
            return e;
        }
    }
    else {
        // index between head_index and cursor_index
        uint32_t head_index = 0;
        if ((index - head_index) <= (list->cursor_index - index)) {
            // from head_index to cursor_index
            ne_someip_list_element_t* e = NULL;
            uint32_t loop_index = head_index;
            for (e = list->head; e != NULL; e = e->next) {
                if (loop_index == index) {
                    list->cursor_index = index;
                    list->cursor = e;
                    break;
                }
                ++loop_index;
            }
            return e;
        }
        else {
            // from cursor_index back to head_index
            ne_someip_list_element_t* e = NULL;
            uint32_t loop_index = list->cursor_index;
            for (e = list->cursor; e != NULL; e = e->prev) {
                if (loop_index == index) {
                    list->cursor_index = index;
                    list->cursor = e;
                    break;
                }
                --loop_index;
            }
            return e;
        }
    }

    return NULL;
}

int32_t ne_someip_list_get_index(ne_someip_list_t* list, void* data)
{
    if (!list) {
        ne_someip_log_error("[List] list is NULL");
        return -1;
    }

    ne_someip_list_element_t* element = list->head;

    int32_t index = 0;
    while (element) {
        if (element->data == data) {
            return index;
        }
        element = element->next;
        index++;
    }

    return -1;
}

ne_someip_list_element_t* ne_someip_list_first(ne_someip_list_t* list)
{
    if (!list) {
        ne_someip_log_error("[List] list is NULL");
        return NULL;
    }

    return list->head;
}

ne_someip_list_element_t* ne_someip_list_last(ne_someip_list_t* list)
{
    if (!list) {
        ne_someip_log_error("[List] list is NULL");
        return NULL;
    }

    return list->tail;
}

ne_someip_list_t* ne_someip_list_insert(ne_someip_list_t* list, void* data, int32_t pos)
{
    if (!list) {
        ne_someip_log_error("[List] list is NULL");
        return NULL;
    }
    if (0 > pos || (list->size - 1 <= pos)) {
        return ne_someip_list_append(list, data);
    }
    else if (0 == pos) {
        return ne_someip_list_prepend(list, data);
    }

    ne_someip_list_element_t *temp_element;
    temp_element = ne_someip_list_get_element(list, pos);

    if (!temp_element) {
        ne_someip_log_error("[List] not find %d data", pos);
        return NULL;
    }

    return _ne_someip_list_insert_before_element(list, temp_element, data);
}

ne_someip_list_t* _ne_someip_list_insert_before_element(ne_someip_list_t* list, ne_someip_list_element_t* element, void* data)
{
    if (!list) {
        ne_someip_log_error("[List] list is NULL");
        return NULL;
    }

    if (!element) {
        return ne_someip_list_append(list, data);
    }

    if (element == list->head) {
        return ne_someip_list_prepend(list, data);
    }

    ne_someip_list_element_t* new_item = _ne_someip_list_element_alloc();
    if (!new_item) {
        ne_someip_log_error("[List] elem alloc error");
        return NULL;
    }
    new_item->data = data;
    new_item->prev = element->prev;
    element->prev->next = new_item;
    new_item->next = element;
    element->prev = new_item;

    ++list->size;
    _ne_someip_list_reset_cursor(list);
    return list;
}

ne_someip_list_t* ne_someip_list_insert_sorted(ne_someip_list_t* list, void* data, ne_someip_base_compare_func cmp_func, void* user_data)
{
    if (!list || !cmp_func) {
        ne_someip_log_error("[List] list:%p, cmp_func:%p", list, cmp_func);
        return NULL;
    }

    ne_someip_list_element_t* element = list->head;
    int32_t cmp_result = 0;
    while (element) {
        cmp_result = (*cmp_func)(data, element->data);
        if (cmp_result <= 0) {
            break;
        }
        element = element->next;
    }

    list = _ne_someip_list_insert_before_element(list, element, data);

    _ne_someip_list_reset_cursor(list);
    return list;
}

ne_someip_list_t* ne_someip_list_sorted(ne_someip_list_t* list, ne_someip_base_compare_func cmp_func, void* user_data)
{
    if (NULL == list || NULL == cmp_func) {
        ne_someip_log_error("[List] list or cmp_func is NULL");
        return NULL;
    }
    // sort by merge sort
    ne_someip_list_element_t* element = list->head;
    element = _ne_someip_list_sorted_by_merge_sort(element, cmp_func, user_data);
    list->head = element;
    if (NULL == element) {
        list->tail = element;
    }
    else {
        while (element->next) {
            element = element->next;
        }
        list->tail = element;
    }

    _ne_someip_list_reset_cursor(list);
    return list;
}

ne_someip_list_element_t* ne_someip_list_find(ne_someip_list_t* list, void* data, ne_someip_base_compare_func cmp_func, void* user_data)
{
    if (NULL == list || NULL == cmp_func) {
        ne_someip_log_error("[List] list or cmp_func is NULL");
        return NULL;
    }
    ne_someip_list_element_t* elem = list->head;
    while (elem) {
        if (cmp_func) {
            if (0 == (*cmp_func)(elem->data, data)) {
                break;
            }
        }
        else {
            if (elem->data == data) {
                break;
            }
        }
        elem = elem->next;
    }
    return elem;
}

void ne_someip_list_foreach(ne_someip_list_t* list, ne_someip_list_foreach_func foreach_func, void* user_data)
{
    if (!foreach_func || !list) {
        ne_someip_log_error("[List] foreach_func:%p, list:%p", foreach_func, list);
        return ;
    }
    ne_someip_list_element_t* element = list->head;
    while (element) {
        if (!(*foreach_func)(element->data, user_data)) {
            break;
        }
        element = element->next;
    }

    return;
}

ne_someip_list_element_t* _ne_someip_list_sort_merge(ne_someip_list_element_t* l1, ne_someip_list_element_t* l2, ne_someip_base_compare_func func, void* user_data)
{
    if (NULL == l2) {
        return l1;
    }

    if (NULL == l1) {
        return l2;
    }

    ne_someip_list_element_t* temp_head = NULL;
    ne_someip_list_element_t* ret_head = NULL;

    int32_t result = 0;
    if (func) {
        result = (*func)(l1->data, l2->data);
    }
    else {
        result = l1->data - l2->data;
    }
    if (0 < result) {
        ret_head = l2;
        temp_head = l2;
        l2 = l2->next;
    }
    else {
        ret_head = l1;
        temp_head = l1;
        l1 = l1->next;
    }


    while ((NULL != l1) && (NULL != l2)) {
        result = (*func)(l1->data, l2->data);
        if (0 < result) {
            temp_head->next = l2;
            l2->prev = temp_head;
            l2 = l2->next;
        }
        else {
            temp_head->next = l1;
            l1->prev = temp_head;
            l1 = l1->next;
        }
        temp_head = temp_head->next;
    }

    if (NULL != l2) {
        temp_head->next = l2;
        l2->prev = temp_head;
    }
    if (NULL != l1) {
        temp_head->next = l1;
        l1->prev = temp_head;
    }

    return ret_head;
}

ne_someip_list_t* _ne_someip_list_list_alloc()
{
    ne_someip_list_t* new_list = malloc(sizeof(ne_someip_list_t));
    if (!new_list) {
        ne_someip_log_error("[List] malloc error");
        return NULL;
    }
    memset(new_list, 0, sizeof(ne_someip_list_t));
    return new_list;
}

ne_someip_list_element_t* _ne_someip_list_element_alloc()
{
    ne_someip_list_element_t* new_list = (ne_someip_list_element_t*)malloc(sizeof(ne_someip_list_element_t));
    if (!new_list) {
        ne_someip_log_error("[List] malloc error");
        return NULL;
    }
    memset(new_list, 0, sizeof(ne_someip_list_element_t));
    return new_list;
}

ne_someip_list_iterator_t* _ne_someip_list_iterator_alloc()
{
    ne_someip_list_iterator_t* iter = malloc(sizeof(ne_someip_list_iterator_t));
    if (!iter) {
        ne_someip_log_error("[List] malloc error");
        return NULL;
    }
    memset(iter, 0, sizeof(ne_someip_list_iterator_t));
    return iter;
}

void _ne_someip_list_element_free(ne_someip_list_element_t* element)
{
    if (element) {
        free(element);
        element = NULL;
    }
}

void _ne_someip_list_list_free(ne_someip_list_t* list)
{
    if (list) {
        free(list);
        list = NULL;
    }
    return;
}

ne_someip_list_element_t* _ne_someip_list_cut(ne_someip_list_element_t* list, int32_t size) {
    while ((list) && (--size)) {
        list = list->next;
    }

    if (!list) {
        return list;
    }

    ne_someip_list_element_t* res = list->next;
    if (res) {
        res->prev = NULL;
    }

    list->next = NULL;

    return res;
}

ne_someip_list_element_t* _ne_someip_list_sorted_by_merge_sort(ne_someip_list_element_t* element_head, ne_someip_base_compare_func cmp_func, void* user_data)
{
    if (!element_head) {
        return element_head;
    }

    ne_someip_list_element_t* l = element_head, *r, *tmp, *res;
    int32_t length = 0;
    while (l) {
        length++;
        l = l->next;
    }

    ne_someip_list_element_t temp_root;
    temp_root.next = element_head;

    for (int32_t size = 1; size < length; size <<= 1) {
        res = &temp_root;
        tmp = temp_root.next;
        while (tmp)
        {
            l = tmp;
            r = _ne_someip_list_cut(l, size);
            tmp = _ne_someip_list_cut(r, size);
            res->next = _ne_someip_list_sort_merge(l, r, cmp_func, NULL);
            if (!(res->next)) {
                ne_someip_log_error("link error");
                return NULL;
            }
            while (res->next) {
                res = res->next;
            }
        }
    }

    return temp_root.next;
}

ne_someip_list_t* _ne_someip_list_reset_cursor(ne_someip_list_t* list)
{
    if (NULL == list) {
        ne_someip_log_error("[List] list is NULL");
        return NULL;
    }
    list->cursor = list->head;
    list->cursor_index = 0;
    return list;
}


ne_someip_list_iterator_t* ne_someip_list_iterator_create(ne_someip_list_t* list)
{
    if (NULL == list) {
        ne_someip_log_error("[List] list is NULL");
        return NULL;
    }
    ne_someip_list_iterator_t* iter = _ne_someip_list_iterator_alloc();
    if (!iter) {
        ne_someip_log_error("[List] alloc error");
        return NULL;
    }
    iter->list = list;
    iter->cur_item = list->head;
    if (iter->cur_item) {
        iter->next_item = iter->cur_item->next;
    }

    return iter;
}

void ne_someip_list_iterator_destroy(ne_someip_list_iterator_t* iterator)
{
    if (iterator) {
        free(iterator);
        iterator = NULL;
    }
    return;
}

void ne_someip_list_iterator_next(ne_someip_list_iterator_t* iterator)
{
    if (!iterator || !(iterator->list)) {
        iterator->next_item = NULL;
        iterator->cur_item = 0;
        return;
    }

    if (0 >= iterator->list->size) {
        _ne_someip_list_iterator_set_invalid(iterator);
        return ;
    }


    if (!iterator->cur_item && iterator->next_item) {
        // cur_item:null, next item:not null(shoud at list->head)
        iterator->cur_item = iterator->next_item;
        iterator->next_item = iterator->next_item->next;
    }
    else if (iterator->cur_item && !iterator->next_item) {
        // cur_item: not null(shoud at list->tail), next_item:null
        iterator->cur_item = iterator->next_item;
    }
    else if (iterator->cur_item && iterator->next_item) {
        // cur_item: not null, next_item:not null
        iterator->cur_item = iterator->next_item;
        iterator->next_item = iterator->next_item->next;
    }
    else if (!iterator->cur_item && !iterator->next_item) {
        // cur_item: null, next_item:null
        // at end position, do nothing
        iterator->next_item = NULL;
        iterator->cur_item = 0;
    }
    else {
        iterator->next_item = NULL;
        iterator->cur_item = 0;
    }

    return;
}

void ne_someip_list_iterator_next_step(ne_someip_list_iterator_t* iterator, int32_t step)
{
    if (!iterator) {
        ne_someip_log_error("[List] iter is NULL");
        return;
    }

    while (step--) {
        ne_someip_list_iterator_next(iterator);
        if (NULL == iterator->cur_item) {
            break;
        }
    }

    return;
}

void ne_someip_list_iterator_prev(ne_someip_list_iterator_t* iterator)
{
    if (!iterator) {
        ne_someip_log_error("[List] iter is NULL");
        return;
    }

    if (0 >= iterator->list->size) {
        _ne_someip_list_iterator_set_invalid(iterator);
        return ;
    }

    if (!iterator->cur_item && iterator->next_item) {
        // cur_item:null, next item:not null(shoud at list->head)
        // at end pos, do nothing
        ;
    }
    else if (iterator->cur_item && !iterator->next_item) {
        // cur_item: not null(shoud at list->tail), next_item:null
        iterator->next_item = iterator->cur_item;
        iterator->cur_item = iterator->cur_item->prev;
    }
    else if (iterator->cur_item && iterator->next_item) {
        // cur_item: not null, next_item:not null
        iterator->next_item = iterator->cur_item;
        iterator->cur_item = iterator->cur_item->prev;
    }
    else if (!iterator->cur_item && !iterator->next_item) {
        // cur_item: null, next_item:null
        iterator->cur_item = iterator->list->tail;
    }
    return;
}

void ne_someip_list_iterator_prev_step(ne_someip_list_iterator_t* iterator, int32_t step)
{
    if (!iterator) {
        ne_someip_log_error("[List] iter is NULL");
        return;
    }

    while (step--) {
        ne_someip_list_iterator_prev(iterator);
        if (NULL == iterator->cur_item) {
            break;
        }
    }

    return;
}

void* ne_someip_list_iterator_data(ne_someip_list_iterator_t* iterator)
{
    if (!iterator || !(iterator->cur_item)) {
        ne_someip_log_error("[List] iter is NULL or cur item is NULL");
        return NULL;
    }
    return iterator->cur_item->data;
}

void ne_someip_list_iterator_remove(ne_someip_list_iterator_t* iterator, ne_someip_base_free_func free_func)
{
    if (!iterator || !iterator->cur_item) {
        ne_someip_log_error("[List] iter is NULL or cur item is NULL");
        return;
    }
    ne_someip_list_element_t* prev = iterator->cur_item->prev;
    ne_someip_list_remove_by_elem(iterator->list, iterator->cur_item, free_func);

    iterator->cur_item = prev;
    return;
}

bool ne_someip_list_iterator_valid(ne_someip_list_iterator_t* iterator)
{
    if (iterator && iterator->cur_item) {
        return true;
    }

    return false;
}

void _ne_someip_list_iterator_set_invalid(ne_someip_list_iterator_t* iterator)
{
    if (NULL == iterator) {
        ne_someip_log_error("[List] iterator is NULL");
        return ;
    }
    iterator->cur_item = NULL;
    iterator->next_item = NULL;
    return ;
}
