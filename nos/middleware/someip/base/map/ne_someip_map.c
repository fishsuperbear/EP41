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
# include "ne_someip_map.h"
# include "ne_someip_object.h"
# include "string.h"
# include "stdlib.h"
# include "ne_someip_log.h"

# define HASH_MAP_MIN_SIZE 3
# define HASH_MAP_MAX_SHIFT 32
# define HASH_MAP_UNUSED_HASH_VALUE 0
# define HASH_MAP_TOMBSTONE_HASH_VALUE 1

# define HASH_MAP_UNUSED(hash_value) ((hash_value) == HASH_MAP_UNUSED_HASH_VALUE)
# define HASH_MAP_TOMBSTONE(hash_value) ((hash_value) == HASH_MAP_TOMBSTONE_HASH_VALUE)
# define HASH_MAP_USED(hash_value) ((hash_value) > HASH_MAP_TOMBSTONE_HASH_VALUE)
# define HASH_VALUE_IS_VALID(hash_value) ((hash_value) > HASH_MAP_TOMBSTONE_HASH_VALUE)

// ===============start hash map object ==================//
struct ne_someip_map {
    // 当前map的容纳数
    int32_t size;
    int32_t mod;
    uint32_t mask;
    // 已经占用坑位数
    int32_t nnodes;
    // 已经占用坑位数+tombstones数
    int32_t nnodes_tombstones;
    // key连续存放内存空间
    void* keys;
    // value连续存放内存空间
    void* values;
    // hash连续存放内存空间
    uint32_t* hashes;
    // 哈希函数
    ne_someip_map_hash_func hash_func;
    // key值比较函数
    ne_someip_base_compare_func key_equal_func;
    // key数据释放函数
    ne_someip_base_free_func key_destroy_func;
    // value数据释放函数
    ne_someip_base_free_func value_destroy_func;
    NEOBJECT_MEMBER;
};
ne_someip_map_t* ne_someip_map_t_new();
void ne_someip_map_t_free(ne_someip_map_t* map);
NEOBJECT_FUNCTION(ne_someip_map_t)


struct ne_someip_map_iter{
    ne_someip_map_t* map;
    int32_t position;
};
// ===============end hash map object ==================//
// ===============start privated func/data ==================//
static const uint32_t prime_mod[] = {
    1,
    2,
    3,
    7,
    13,
    31,
    61,
    127,
    251,
    509,
    1021,
    2039,
    4093,
    8191,
    16381,
    32749,
    65521,          /* for 1 << 16*/
    131071,
    262139,
    524287,
    1048573,
    2097143,
    4194301,
    8388593,
    16777213,
    33554393,
    67108859,
    134217689,
    268435399,
    536870909,
    1073741789,
    2147483647   /* for 1 << 31*/
};
static uint32_t _ne_someip_map_default_hash_func(const void * key);
static void _ne_someip_map_hash_table_set_shift(ne_someip_map_t* hash_map, int32_t shift);
static void _ne_someip_map_hash_table_set_shift_by_size(ne_someip_map_t* hash_map, int32_t size);
static uint32_t _ne_someip_map_get_closest_shift(uint32_t n);
static uint32_t _ne_someip_map_hash_to_index(ne_someip_map_t* hash_map, uint32_t hash_value);
static uint32_t _ne_someip_map_hash_map_lookup_node_index(ne_someip_map_t* hash_map, const void *key, uint32_t* hash_return);
static void _ne_someip_map_hash_map_remove_node(ne_someip_map_t* hash_map, uint32_t node_index, bool destory_mem);
static bool _ne_someip_map_hash_map_insert_node(ne_someip_map_t* hash_map, uint32_t node_index, uint32_t key_hash, void* new_key, void* new_value, bool use_new_key, bool destory_unused_key);
static bool _ne_someip_map_hash_map_check_need_resize(ne_someip_map_t* hash_map);
static bool _ne_someip_map_hash_map_resize(ne_someip_map_t* hash_map);
static void _ne_someip_map_resize(ne_someip_map_t* hash_map, uint32_t old_size, uint32_t* resize_bitmap);
static void _ne_someip_map_remove_all_node(ne_someip_map_t* map);
static inline bool _ne_someip_map_get_bitmap_status(uint32_t* bitmap, uint32_t index);
static inline void _ne_someip_map_set_bitmap_status(uint32_t* bitmap, uint32_t index);
static inline void* _ne_someip_map_get_key_or_value(void* key_or_value, uint32_t index);
static inline void _ne_someip_map_set_key_or_value(void* map_key_or_value, uint32_t index, void* key_or_value);
static inline void* _ne_someip_map_hash_map_exchange_key_or_value(void* key_or_value, uint32_t index, void* in_data);
static inline bool _ne_someip_map_realloc_arrays(ne_someip_map_t* hash_map, uint32_t old_size);
// ===============end privated func/data ==================//


ne_someip_map_t* ne_someip_map_new(ne_someip_map_hash_func hash_func, ne_someip_base_compare_func key_cmp_func, ne_someip_base_free_func key_free_func, ne_someip_base_free_func value_free_func)
{
    ne_someip_map_t* hash_map = ne_someip_map_t_new();
    if (NULL == hash_map) {
        ne_someip_log_error("[Map] malloc error");
        return NULL;
    }
    // 初始化size等
    _ne_someip_map_hash_table_set_shift(hash_map, HASH_MAP_MIN_SIZE);
    // 初始化成员变量
    hash_map->nnodes = 0;
    hash_map->nnodes_tombstones = 0;
    // 初始化数组
    hash_map->keys = (void*)malloc(sizeof(void *) * hash_map->size);
    hash_map->values = (void*)malloc(sizeof(void *) * hash_map->size);
    hash_map->hashes = (uint32_t*)malloc(sizeof(uint32_t *) * hash_map->size);
    if (NULL == hash_map->keys || NULL == hash_map->values || NULL == hash_map->hashes) {
        ne_someip_log_error("[Map] table malloc error");
        ne_someip_map_unref(hash_map);
        return NULL;
    }

    memset(hash_map->keys, 0, sizeof(void *) * hash_map->size);
    memset(hash_map->values, 0, sizeof(void *) * hash_map->size);
    memset(hash_map->hashes, 0, sizeof(uint32_t *) * hash_map->size);

    // 初始化函数指针
    hash_map->hash_func = hash_func ? hash_func : _ne_someip_map_default_hash_func;
    hash_map->key_equal_func = key_cmp_func;
    hash_map->key_destroy_func = key_free_func;
    hash_map->value_destroy_func = value_free_func;

    return hash_map;
}

ne_someip_map_t* ne_someip_map_ref(ne_someip_map_t* map)
{
    if (!map) {
        ne_someip_log_error("[Map] map is NULL");
        return NULL;
    }
    map = ne_someip_map_t_ref(map);
    return map;
}

void ne_someip_map_unref(ne_someip_map_t* map)
{
    if (!map) {
        ne_someip_log_error("[Map] map is NULL");
        return;
    }
    ne_someip_map_t_unref(map);
    return;
}

bool ne_someip_map_insert(ne_someip_map_t* map, void* key, void* value)
{
    if ((!map) || (!key)) {
        ne_someip_log_error("[Map] map or key is NULL");
        return false;
    }

    // ne_someip_log_debug("[Map] try insert key:%p,value:%p into map:%p", key, value, map);

    uint32_t hash_ret;
    uint32_t node_index;
    node_index = _ne_someip_map_hash_map_lookup_node_index(map, key, &hash_ret);

    return _ne_someip_map_hash_map_insert_node(map, node_index, hash_ret, key, value, true, true);
}

bool ne_someip_map_remove(ne_someip_map_t* map, void* key, bool resize)
{
    if (!map) {
        ne_someip_log_error("[Map] map is NULL");
        return false;
    }

    // ne_someip_log_debug("[Map] try remove key:%p from map:%p", key, map);

    uint32_t node_hash;
    uint32_t node_index = _ne_someip_map_hash_map_lookup_node_index(map, key, &node_hash);

    if (!HASH_VALUE_IS_VALID(map->hashes[node_index])) {
        // 未找到对应key值
        ne_someip_log_error("[Map] remove item is not find");
        return false;
    }

    _ne_someip_map_hash_map_remove_node(map, node_index, true);

    if (resize && _ne_someip_map_hash_map_check_need_resize(map)) {
        _ne_someip_map_hash_map_resize(map);
    }

    return true;
}

void ne_someip_map_remove_all(ne_someip_map_t* map)
{
    if (!map) {
        ne_someip_log_error("[Map] map is NULL");
        return;
    }

    ne_someip_log_debug("[Map] remove all node for map:%p", map);

    _ne_someip_map_remove_all_node(map);
    return;
}

uint32_t ne_someip_map_int32_hash_func(const void* key)
{
    if (!key) {
        ne_someip_log_error("[Map] key is NULL");
        return 0;
    }
    return (uint32_t)(*((int32_t*)key));
}

int32_t ne_someip_map_int32_cmp_func(const void* k1, const void* k2)
{
    if ((!k1) || (!k2)) {
        ne_someip_log_error("[Map] k1 or k2 is NULL");
        return -1;
    }

    return *((int32_t*)k1) - *((int32_t*)k2);
}

uint32_t ne_someip_map_string_hash_func(const void* key)
{
    if (NULL == key) {
        ne_someip_log_error("[Map] key is NULL");
        return 0;
    }
    // JS Hash Func
    uint32_t hash = 0;
    char *p;
    for (p = (char*)key; *p; p++) {
        hash = (hash << 5) + hash + *p;
    }
    return hash;
}

int32_t ne_someip_map_string_cmp_func(const void* k1, const void* k2)
{
    if ((!k1) || (!k2)) {
        ne_someip_log_error("[Map] k1 or k2 is NULL");
        return -1;
    }

    return strcmp(k1, k2);
}

uint32_t ne_someip_map_pointer_hash_func(const void* key)
{
    return _ne_someip_map_default_hash_func(key);
}

int32_t ne_someip_map_pointer_cmp_func(const void* k1, const void* k2)
{
    return k1 - k2;
}

void* ne_someip_map_find(ne_someip_map_t* map, const void* key, uint32_t* hash_return)
{
    if ((!map) || (!hash_return)) {
        ne_someip_log_error("[Map] map or hash_return is NULL");
        return NULL;
    }

    uint32_t node_index = _ne_someip_map_hash_map_lookup_node_index(map, key, hash_return);
    if (HASH_VALUE_IS_VALID(map->hashes[node_index])) {
        return _ne_someip_map_get_key_or_value(map->values, node_index);
    }

    // ne_someip_log_debug("[Map] not find key:%p, from map:%p, hash_ret:%d", key, map, *hash_return);

    return NULL;
}

bool ne_someip_map_check(ne_someip_map_t* map, void* key)
{
    if (!map) {
        ne_someip_log_error("[Map] map is NULL");
        return NULL;
    }

    uint32_t hash_return;

    uint32_t node_index = _ne_someip_map_hash_map_lookup_node_index(map, key, &hash_return);
    if (HASH_VALUE_IS_VALID(map->hashes[node_index])) {
        return true;
    }

    return false;
}

bool ne_someip_map_empty(ne_someip_map_t* map)
{
    if (!map) {
        ne_someip_log_error("[Map] map is NULL");
        return false;
    }

    if (0 < map->nnodes) {
        return false;
    }

    return true;
}

int32_t ne_someip_map_key_size(ne_someip_map_t* map)
{
    if (!map) {
        ne_someip_log_error("[Map] map is NULL");
        return -1;
    }

    return map->nnodes;
}

ne_someip_list_t* ne_someip_map_keys(ne_someip_map_t* map)
{
    if (!map) {
        ne_someip_log_error("[Map] map is NULL");
        return NULL;
    }
    ne_someip_list_t* ret_list = ne_someip_list_create();
    if (NULL == ret_list) {
        ne_someip_log_error("[Map] list create error");
        return NULL;
    }
    for (uint32_t i = 0; i < map->size; i++) {
        if (HASH_VALUE_IS_VALID(map->hashes[i])) {
            ne_someip_list_append(ret_list, _ne_someip_map_get_key_or_value(map->keys, i));
        }
    }

    // ne_someip_log_debug("[Map] dump map:%p all key size:%d", map, ne_someip_list_length(ret_list));

    return ret_list;
}

ne_someip_list_t* ne_someip_map_values(ne_someip_map_t* map)
{
    if (!map) {
        ne_someip_log_error("[Map] map is NULL");
        return NULL;
    }
    ne_someip_list_t* ret_list = ne_someip_list_create();
    if (NULL == ret_list) {
        ne_someip_log_error("[Map] list create error");
        return NULL;
    }
    for (uint32_t i = 0; i < map->size; i++) {
        if (HASH_VALUE_IS_VALID(map->hashes[i])) {
            ne_someip_list_append(ret_list, _ne_someip_map_get_key_or_value(map->values, i));
        }
    }

    // ne_someip_log_debug("[Map] dump map:%p all value size:%d", map, ne_someip_list_length(ret_list));

    return ret_list;
}

ne_someip_map_iter_t* ne_someip_map_iter_new(ne_someip_map_t* map)
{
    if (!map) {
        ne_someip_log_error("[Map] map is NULL");
        return NULL;
    }

    ne_someip_map_iter_t* it = malloc(sizeof(ne_someip_map_iter_t));
    if (!it) {
        ne_someip_log_error("[Map] malloc error");
        return NULL;
    }
    memset(it, 0, sizeof(ne_someip_map_iter_t));

    it->map = map;
    it->position = -1;

    return it;
}

void ne_someip_map_iter_destroy(ne_someip_map_iter_t* iter)
{
    if (iter) {
        free(iter);
        iter = NULL;
    }
    return;
}

bool ne_someip_map_iter_next(ne_someip_map_iter_t* iter, void** key, void** value)
{
    if ((!iter) || (NULL == iter->map) || (iter->position >= iter->map->size)) {
        ne_someip_log_error("[Map] iter or size error");
        return false;
    }
    int32_t pos = iter->position;

    while (true) {
        pos++;
        iter->position = pos;
        if (pos >= iter->map->size) {
            return false;
        }
        if (HASH_VALUE_IS_VALID(iter->map->hashes[pos])) {
            break;
        }
    }

    if (NULL != key) {
        *key = _ne_someip_map_get_key_or_value(iter->map->keys, pos);
    }
    if (NULL != value) {
        *value = _ne_someip_map_get_key_or_value(iter->map->values, pos);
    }

    return true;
}

// ===============start privated func ==================//
uint32_t _ne_someip_map_default_hash_func(const void * key)
{
    return 0;
}

void _ne_someip_map_hash_table_set_shift(ne_someip_map_t* hash_map, int32_t shift)
{
    hash_map->size = 1 << shift;
    hash_map->mod = prime_mod[shift];

    if ((hash_map->size & (hash_map->size - 1)) != 0) {
        ne_someip_log_error("[Map] map size error:%d", hash_map->size);
        return;
    }
    hash_map->mask = hash_map->size - 1;
    return;
}

void _ne_someip_map_hash_table_set_shift_by_size(ne_someip_map_t* hash_map, int32_t size)
{
    uint32_t shift = _ne_someip_map_get_closest_shift(size);
    shift = shift > HASH_MAP_MIN_SIZE ? shift : HASH_MAP_MIN_SIZE;

    _ne_someip_map_hash_table_set_shift(hash_map, shift);
}

uint32_t _ne_someip_map_get_closest_shift(uint32_t n)
{
    uint32_t i;
    for (i = 0; n; i++) {
        n = n >> 1;
    }
    return i;
}

uint32_t _ne_someip_map_hash_to_index(ne_someip_map_t* hash_map, uint32_t hash_value)
{
    /* 根据glib: 为了防止hash函数散列效果很差，所以即使通过函数散列后
    *  也需要乘以一个小的素数，来降低碰撞概率 */
    return (hash_value * 11) % hash_map->mod;
}

uint32_t _ne_someip_map_hash_map_lookup_node_index(ne_someip_map_t* hash_map, const void *key, uint32_t* hash_return)
{
    /* 根据key查找对应位置，如果该位置是空，直接返回该位置。
       否则根据key比较函数，找到相应key值的位置，或返回第一个tombstones位置*/
    uint32_t hash_value = hash_map->hash_func(key);
    if (!HASH_VALUE_IS_VALID(hash_value)) {
        hash_value = 2;
    }
    *hash_return = hash_value;

    uint32_t node_index = _ne_someip_map_hash_to_index(hash_map, hash_value);
    uint32_t node_hash_value = hash_map->hashes[node_index];

    bool has_tombstone = false;
    uint32_t first_tombstone = 0;
    uint32_t step = 0;

    if (hash_map->nnodes_tombstones == hash_map->size) {
        // map数据存在问题，循环查找可能会导致死循环
        ne_someip_log_error("[Map] nnodes_tombstones:%d, size:%d", hash_map->nnodes_tombstones, hash_map->size);
        return node_index;
    }

    while (!HASH_MAP_UNUSED(node_hash_value))
    {
        if (node_hash_value == hash_value) {
            void* node_key = _ne_someip_map_get_key_or_value(hash_map->keys, node_index);
            if (hash_map->key_equal_func) {
                if (0 == hash_map->key_equal_func(key, node_key)) {
                    return node_index;
                }
            }
            else {
                if (key == node_key) {
                    return node_index;
                }
            }
        }
        else if (HASH_MAP_TOMBSTONE(node_hash_value) && (!has_tombstone)) {
            // 记录找到的第一个tombstone位置
            first_tombstone = node_index;
            has_tombstone = true;
        }

        step++;
        node_index += step;
        node_index &= hash_map->mask;
        node_hash_value = hash_map->hashes[node_index];
    }

    // 找到空位置前，有找到一个tombstone位置
    if (has_tombstone) {
        // 返回第一个tombstone位置
        return first_tombstone;
    }

    // 返回空位置的index
    return node_index;
}

static void _ne_someip_map_hash_map_remove_node(ne_someip_map_t* hash_map, uint32_t node_index, bool destory_mem)
{
    void* key = _ne_someip_map_get_key_or_value(hash_map->keys, node_index);
    void* value = _ne_someip_map_get_key_or_value(hash_map->values, node_index);

    // 设置hash值为tombonstones
    hash_map->hashes[node_index] = HASH_MAP_TOMBSTONE_HASH_VALUE;

    _ne_someip_map_set_key_or_value(hash_map->keys, node_index, NULL);
    _ne_someip_map_set_key_or_value(hash_map->values, node_index, NULL);

    hash_map->nnodes--;

    if (destory_mem && hash_map->key_destroy_func) {
        (*(hash_map->key_destroy_func))(key);
    }

    if (destory_mem && hash_map->value_destroy_func) {
        (*(hash_map->value_destroy_func))(value);
    }
}

static bool _ne_someip_map_hash_map_insert_node(ne_someip_map_t* hash_map, uint32_t node_index, uint32_t key_hash, void* new_key, void* new_value, bool use_new_key, bool destory_unused_key)
{
    uint32_t old_key_hash;
    bool already_exists = false;

    old_key_hash = hash_map->hashes[node_index];
    already_exists = HASH_VALUE_IS_VALID(old_key_hash);

    void* key_to_free = NULL;
    void* key_to_use = NULL;
    void* old_value = NULL;
    // 更新hash value
    if (already_exists) {
        // 当前位置存在旧数据
        old_value = _ne_someip_map_get_key_or_value(hash_map->values, node_index);
        if (use_new_key) {
            key_to_free = _ne_someip_map_get_key_or_value(hash_map->keys, node_index);
            key_to_use = new_key;
        }
        else {
            key_to_free = new_key;
            key_to_use = _ne_someip_map_get_key_or_value(hash_map->keys, node_index);
        }
    }
    else {
        hash_map->hashes[node_index] = key_hash;
        key_to_use = new_key;
        key_to_free = NULL;
        old_value = NULL;
    }

    // 更新key值
    _ne_someip_map_set_key_or_value(hash_map->keys, node_index, key_to_use);
    // 更新value值
    _ne_someip_map_set_key_or_value(hash_map->values, node_index, new_value);

    if (!already_exists) {
        // node个数+1
        hash_map->nnodes++;
        if (HASH_MAP_UNUSED(old_key_hash)) {
            // 使用的是空位置，nnodes_tombstones+1
            hash_map->nnodes_tombstones++;
            // 由于空位减少，需要检测是否需要resize
            if (_ne_someip_map_hash_map_check_need_resize(hash_map)) {
                if (false == _ne_someip_map_hash_map_resize(hash_map)) {
                    // resize失败，删除刚刚插入的数据
                    ne_someip_log_error("[Map] resize error!!!");
                    _ne_someip_map_hash_map_remove_node(hash_map, node_index, false);
                    // 尝试重新resize，将tombstone清空，防止不停插入导致tombston写满数据
                    _ne_someip_map_hash_map_resize(hash_map);
                    return false;
                }
            }
        }
    }

    if (already_exists) {
        // 既有数据内存清理
        if (hash_map->value_destroy_func) {
            (*hash_map->value_destroy_func)(old_value);
        }
        if (hash_map->key_destroy_func && destory_unused_key) {
            (*hash_map->key_destroy_func)(key_to_free);
        }
    }

    return true;
}

static bool _ne_someip_map_hash_map_check_need_resize(ne_someip_map_t* hash_map)
{
    int32_t nnodes_tombstones = hash_map->nnodes_tombstones;
    int32_t size = hash_map->size;

    if ((size > (hash_map->nnodes * 4) && size > (1 << HASH_MAP_MIN_SIZE)) \
        || (size <= nnodes_tombstones + (nnodes_tombstones / 16))) {
        // 需要resize
        return true;
    }
    return false;
}

/**
 * 调整map大小
 */
static bool _ne_someip_map_hash_map_resize(ne_someip_map_t* hash_map)
{
    uint32_t *resize_bitmap;
    uint32_t bitmap_size = 0;
    int32_t old_size;

    old_size = hash_map->size;

    _ne_someip_map_hash_table_set_shift_by_size(hash_map, hash_map->nnodes * 1.333);

    if (hash_map->size > old_size) {
        // hash map变大，重新分配内存
        if (false == _ne_someip_map_realloc_arrays(hash_map, old_size)) {
            // 内存分配失败，size改回
            _ne_someip_map_hash_table_set_shift_by_size(hash_map, old_size);
            return false;
        }
        // 计算bitmap大小
        bitmap_size = ((hash_map->size + 31) / 32);
    }
    else {
        // hash map变小
        // 计算bitmap大小
        bitmap_size = ((old_size + 31) / 32);
    }
    // 构建bitmap
    resize_bitmap = (uint32_t*)malloc(bitmap_size * sizeof(uint32_t));
    if (NULL == resize_bitmap) {
        // 内存分配失败，size改回
        _ne_someip_map_hash_table_set_shift_by_size(hash_map, old_size);
        return false;
    }
    memset(resize_bitmap, 0, bitmap_size * sizeof(uint32_t));

    // 重新排布表中key-value
    _ne_someip_map_resize(hash_map, old_size, resize_bitmap);

    // 释放bitmap
    free(resize_bitmap);
    resize_bitmap = NULL;
    if (hash_map->size < old_size) {
        // hash map变小，重新分配内存
        if (false == _ne_someip_map_realloc_arrays(hash_map, old_size)) {
            // 内存分配失败，size改回
            _ne_someip_map_hash_table_set_shift_by_size(hash_map, old_size);
            return false;
        }
    }

    hash_map->nnodes_tombstones = hash_map->nnodes;

    return true;
}

/**
 * 重新计算位置
 */
static void _ne_someip_map_resize(ne_someip_map_t* hash_map, uint32_t old_size, uint32_t* resize_bitmap)
{
    int32_t i;
    for (i = 0; i < old_size; i++) {
        uint32_t cur_hash = hash_map->hashes[i];
        if (!HASH_VALUE_IS_VALID(cur_hash)) {
            // 所有非正常hash值全部初始化为unused
            hash_map->hashes[i] = HASH_MAP_UNUSED_HASH_VALUE;
            continue;
        }

        if (_ne_someip_map_get_bitmap_status(resize_bitmap, i)) {
            // 该位置的数据已经更新完毕
            continue;
        }

        // 换出当前key-value，并重置hash值
        hash_map->hashes[i] = HASH_MAP_UNUSED_HASH_VALUE;
        void* key = _ne_someip_map_hash_map_exchange_key_or_value(hash_map->keys, i, NULL);
        void* value = _ne_someip_map_hash_map_exchange_key_or_value(hash_map->values, i, NULL);
        while (true)
        {
            uint32_t new_index = _ne_someip_map_hash_to_index(hash_map, cur_hash);
            uint32_t step = 0;
            // 持续查找下一个未修改的位置
            while(_ne_someip_map_get_bitmap_status(resize_bitmap, new_index))
            {
                step++;
                new_index += step;
                new_index &= hash_map->mask;
            }

            // 更新为已修改
            _ne_someip_map_set_bitmap_status(resize_bitmap, new_index);
            // 取出当前位置的hash值，更新hash值
            uint32_t replace_hash = hash_map->hashes[new_index];
            hash_map->hashes[new_index] = cur_hash;

            // 新位置原本是否为空
            if (!HASH_VALUE_IS_VALID(replace_hash)) {
                // 原本无数据，直接将数据插入
                _ne_someip_map_set_key_or_value(hash_map->keys, new_index, key);
                _ne_someip_map_set_key_or_value(hash_map->values, new_index, value);
                break;
            }
            else {
                // 将当前数据插入，并换出原始数据
                key = _ne_someip_map_hash_map_exchange_key_or_value(hash_map->keys, new_index, key);
                value = _ne_someip_map_hash_map_exchange_key_or_value(hash_map->values, new_index, value);
                cur_hash = replace_hash;
            }
        }
    }
}

/**
 * 移除所有key-value对
 */
void _ne_someip_map_remove_all_node(ne_someip_map_t* map)
{
    if (map->nnodes == 0) {
        return;
    }
    // 清空保存数据
    map->nnodes = 0;
    map->nnodes_tombstones = 0;

    if ((NULL == map->key_destroy_func) && (NULL == map->value_destroy_func)) {
        memset(map->hashes, 0, sizeof(uint32_t) * map->size);
        memset(map->keys, 0, sizeof(void *) * map->size);
        memset(map->values, 0, sizeof(void *) * map->size);
        return;
    }
    void* key;
    void* value;
    for (int32_t i = 0; i < map->size; i++) {
        if (HASH_VALUE_IS_VALID(map->hashes[i])) {
            key = _ne_someip_map_get_key_or_value(map->keys, i);
            value = _ne_someip_map_get_key_or_value(map->values, i);
            map->hashes[i] = HASH_MAP_UNUSED_HASH_VALUE;
            if (map->key_destroy_func) {
                (*map->key_destroy_func)(key);
            }
            if (map->value_destroy_func) {
                (*map->value_destroy_func)(value);
            }
        }
    }
    memset(map->hashes, 0, sizeof(uint32_t) * map->size);
    return ;
}

static inline bool _ne_someip_map_realloc_arrays(ne_someip_map_t* hash_map, uint32_t old_size)
{
    if (0 >= hash_map->size) {
        ne_someip_log_error("[Map] size error:%d", hash_map->size);
        return false;
    }
    void* new_keys = (void*)realloc(hash_map->keys, hash_map->size * sizeof(void*));
    if (NULL == new_keys) {
        ne_someip_log_error("[Map] realloc error:%d", hash_map->size);
        return false;
    }
    else {
        hash_map->keys = new_keys;
    }
    void* new_values = (void*)realloc(hash_map->values, hash_map->size * sizeof(void*));
    if (NULL == new_values) {
        ne_someip_log_error("[Map] realloc error:%d", hash_map->size);
        return false;
    }
    else {
        hash_map->values = new_values;
    }
    uint32_t* new_hashes = (uint32_t*)realloc(hash_map->hashes, hash_map->size * sizeof(uint32_t));
    if (NULL == new_hashes) {
        ne_someip_log_error("[Map] realloc error:%d", hash_map->size);
        return false;
    }
    else {
        hash_map->hashes = new_hashes;
        if (old_size < hash_map->size) {
            memset(&hash_map->hashes[old_size], 0, ((hash_map->size - old_size) * sizeof(uint32_t)));
        }
    }

    return true;
}

static inline bool _ne_someip_map_get_bitmap_status(uint32_t* bitmap, uint32_t index)
{
    return (bitmap[index / 32] >> (index % 32)) & 1;
}

static inline void _ne_someip_map_set_bitmap_status(uint32_t* bitmap, uint32_t index)
{
    bitmap[index / 32] = bitmap[index / 32] | (1U << (index % 32));
    return ;
}

static inline void* _ne_someip_map_get_key_or_value(void* key_or_value, uint32_t index)
{
    return *(((void**)key_or_value) + index);
}

static inline void _ne_someip_map_set_key_or_value(void* map_key_or_value, uint32_t index, void* key_or_value)
{
    *(((void**)map_key_or_value) + index) = key_or_value;
    return;
}

static inline void* _ne_someip_map_hash_map_exchange_key_or_value(void* key_or_value, uint32_t index, void* in_data)
{
    void* ret = *(((void**)key_or_value) + index);
    *(((void**)key_or_value) + index) = in_data;
    return ret;
}
// ===============end privated func ==================//

// ==================start object ====================//

ne_someip_map_t* ne_someip_map_t_new()
{
    ne_someip_map_t* map;
    map = (ne_someip_map_t*)malloc(sizeof(ne_someip_map_t));
    if (NULL == map) {
        ne_someip_log_error("[Map] malloc error");
        return NULL;
    }
    memset(map, 0, sizeof(ne_someip_map_t));

    ne_someip_map_t_ref_count_init(map);

    ne_someip_log_debug("[Map] create map %p", map);
    return map;
}

void ne_someip_map_t_free(ne_someip_map_t* map)
{
    if (NULL == map) {
        ne_someip_log_error("[Map] map is NULL");
        return ;
    }
    ne_someip_log_debug("[Map] free map %p", map);
    ne_someip_map_t_ref_count_deinit(map);

    // 移除所有node，且不做size变换
    _ne_someip_map_remove_all_node(map);
    // 三个表需要free
    free(map->hashes);
    map->hashes = NULL;
    free(map->keys);
    map->keys = NULL;
    free(map->values);
    map->values = NULL;

    // 释放map
    free(map);
    map = NULL;
    return;
}


// ==================end object ====================//
