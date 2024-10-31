#pragma once

#include <utility>
#include <functional>

#include "framework/concurrency/detail/concurrent_hashmap_impl.h"

namespace netaos {
namespace framework {
namespace concurrency {

// 参照Java 1.7 ConcurrentHashMap
// 采用分段锁

template<
    typename KeyType,
    typename ValueType,
    typename HashFn = std::hash<KeyType>,
    typename KeyEqual = std::equal_to<KeyType>
    >
class ConcurrentHashMap {
public:
    class ConstIterator;
    typedef KeyType key_type;
    typedef ValueType mapped_type;
    typedef std::pair<const KeyType, ValueType> value_type;
    typedef std::size_t size_type;
    typedef HashFn hasher;
    typedef KeyEqual key_equal;
    typedef ConstIterator const_iterator;
  
    static constexpr size_t DEFAULT_INITIAL_CAPACITY = 16;
    static constexpr float DEFAULT_LOAD_FACTOR = 0.75;
    static constexpr size_t DEFAULT_CONCURRENCY_LEVEL = 32;
    static constexpr size_t MAXIMUM_CAPACITY = 1 << 30;
    static constexpr size_t MIN_SEGMENT_TABLE_CAPACITY = 2;
    static constexpr size_t MAX_SEGMENTS = 1 << 16;
    static constexpr size_t RETRIES_BEFORE_LOCK = 2;
  
private:
    using SegmentT = detail::Segment<KeyType, ValueType, HashFn, KeyEqual>;
    // should not modify
    size_t _segment_mask;
    size_t _segment_sift;
    size_t _hash_seed;

    size_t _seg_tab_size;
    float _load_factor;

    mutable std::vector<std::atomic<SegmentT*>> _segments;

public:
    // constructor
    ConcurrentHashMap(size_t init_capacity, float load_factor, size_t concurrency_level) {
        if (concurrency_level > MAX_SEGMENTS) {
            concurrency_level = MAX_SEGMENTS;
        }
        size_t ssift = 0;
        size_t ssize = 1;
        while (ssize < concurrency_level) {
            ssize <<= 1;
            ++ssift;
        }
        _segment_sift = 32 - ssift;
        _segment_mask = ssize - 1;
        if (init_capacity > MAXIMUM_CAPACITY) {
            init_capacity = MAXIMUM_CAPACITY;
        }
        size_t c = init_capacity / ssize;
        if (c * ssize < init_capacity) {
            ++c;
        }
        size_t cap = MIN_SEGMENT_TABLE_CAPACITY;
        while (cap < c) {
            cap <<= 1;
        }
        _seg_tab_size = cap;
        _load_factor = load_factor;
        
        std::vector<std::atomic<SegmentT*>> segments(ssize);
        _segments.swap(segments);

        _hash_seed = std::hash<void*>()(this);
    }

    ConcurrentHashMap(size_t init_capacity, float load_factor) : ConcurrentHashMap(init_capacity, load_factor, DEFAULT_CONCURRENCY_LEVEL)
    {}

    explicit ConcurrentHashMap(size_t init_capacity) : ConcurrentHashMap(init_capacity, DEFAULT_LOAD_FACTOR, DEFAULT_CONCURRENCY_LEVEL)
    {}

    ConcurrentHashMap() : ConcurrentHashMap(DEFAULT_INITIAL_CAPACITY, DEFAULT_LOAD_FACTOR, DEFAULT_CONCURRENCY_LEVEL)
    {}

    ~ConcurrentHashMap() {
        for (size_t i = 0; i < _segments.size(); i++) {
        SegmentT* seg = _segments[i].load(std::memory_order_relaxed);
            if (seg) {
                delete seg;
            }
        }
    }

    // modifiers
    // key值不存在时插入
    bool insert(const value_type& value) {
        size_t h = hash(value.first);
        size_t idx = hash_to_idx(h);
        return ensure_segment(idx)->do_insert(value.first, h, value.second, true);
    }
  
    template<typename Key, typename Value>
    bool insert(Key&& k, Value&& v) {
        size_t h = hash(k);
        size_t idx = hash_to_idx(h);
        return ensure_segment(idx)->do_insert(std::forward<Key>(k), h, std::forward<Value>(v), true);    
    }

    // insert时返回true, assign时返回false 
    bool insert_or_assign(const key_type& k, const mapped_type& v) {
        size_t h = hash(k);
        size_t idx = hash_to_idx(h);
        return ensure_segment(idx)->do_insert(k, h, v, false);
    }

    size_type erase(const key_type& key) {
        size_t h = hash(key);
        size_t idx = hash_to_idx(h);
        SegmentT* seg = _segments[idx].load(std::memory_order_acquire);
        if (seg == nullptr) {
            return 0;
        }
        return seg->erase(key, h);
    }

    bool erase_if_equal(const key_type& key, const mapped_type& expected) {
        size_t h = hash(key);
        size_t idx = hash_to_idx(h);
        SegmentT* seg = _segments[idx].load(std::memory_order_acquire);
        if (!seg) {
            return false;
        }
        return seg->erase_if_equal(key, h, expected);
    }

    bool assign(const key_type& key, const mapped_type& val) {
        size_t h = hash(key);
        size_t idx = hash_to_idx(h);
        SegmentT* seg = _segments[idx].load(std::memory_order_acquire);
        if (seg == nullptr) {
            return false;
        }
        return seg->assign(key, h, val);
    }

    bool assign_if_equal(const key_type& key, const mapped_type& expected, const mapped_type& desired) {
        size_t h = hash(key);
        size_t idx = hash_to_idx(h);
        SegmentT* seg = _segments[idx].load(std::memory_order_acquire);
        if (seg == nullptr) {
            return false;
        }
        return seg->assign_if_equal(key, h, expected, desired); 
    }

    void for_each(std::function<void(value_type&)> action) {
        for (size_t i = 0; i < _segments.size(); ++i) {
            SegmentT* seg = _segments[i].load(std::memory_order_acquire);
            if (seg == nullptr) {
                continue;
            }
            seg->for_each(action);
        }
    }

    void clear() {
        for (size_t i = 0; i < _segments.size(); ++i) {
        SegmentT* seg = _segments[i].load(std::memory_order_acquire);
            if (seg) {
                seg->clear();
            }
        }
    }

    // lookup
    const mapped_type at(const key_type& key) const {
        mapped_type v;
        bool ret = get_internal(key, &v);
        if (!ret) {
            throw std::out_of_range("key not found!");
        }
        return v;
    }

  // const mapped_type operator[](const key_type& key) {
  //   mapped_type default_value{};
  //   bool ret = insert_or_assign(key, default_value);
    
  // }

    bool get(const key_type& key, mapped_type* out) const {
        if (out == nullptr) {
            throw std::invalid_argument("out is null!");
        }
        return get_internal(key, out);
    }
  
    bool contains(const key_type& key) const {
        return get_internal(key, nullptr);
    }

    // capacity
    bool empty() const noexcept {
        for (size_t i = 0; i < _segments.size(); i++) {
        SegmentT* seg = _segments[i].load(std::memory_order_acquire);
            if (seg) {
                if (!seg->empty()) {
                return false;
                }
            }
        }
        return true;
    }

    // This is a rolling size, and is not exact at any moment in time.
    size_type size() const {
        size_t res = 0;
        for (size_t i = 0; i < _segments.size(); ++i) {
            SegmentT* seg = _segments[i].load(std::memory_order_acquire);
            if (seg) {
                res += seg->size();
            }
        }
        return res;
    }

    ConstIterator cend() const noexcept {
        return ConstIterator(nullptr, _segments.size());
    }

    ConstIterator cbegin() const noexcept {
        return ConstIterator(this);
    }

    ConstIterator end() const noexcept {
        return cend();
    }

    ConstIterator begin() const noexcept {
        return cbegin();
    }

class ConstIterator {
public:
    friend class ConcurrentHashMap;

    // cbegin call this
    explicit ConstIterator(const ConcurrentHashMap* parent)
        : _it(parent->ensure_segment(0)->cbegin())
        , _seg_idx(0)
        , _parent(parent) {
        next();
    }

    ConstIterator(const ConcurrentHashMap* parent, size_t seg_idx)
        : _seg_idx(seg_idx), _parent(parent) {
    }

    ConstIterator(const ConstIterator&) = delete;
    ConstIterator(ConstIterator&& o) noexcept
        : _it(std::move(o._it))
        , _seg_idx(o._seg_idx)
        , _parent(o._parent) {

        o._seg_idx = _parent->_segments.size();
        o._parent = nullptr;
    }

    ConstIterator& operator=(const ConstIterator&) = delete;
    ConstIterator& operator=(ConstIterator&& o) noexcept {
        if (this != &o) {
            _it = std::move(o._it);
            _seg_idx = o._seg_idx;
            _parent = o._parent;
            o._seg_idx = _parent._segments.size();
            o._parent = nullptr;
        }
    }

    const value_type& operator*() const {
        return *_it;
    }

    const value_type* operator->() const {
        return &(*_it);
    }

    bool operator==(const ConstIterator& o) const {
        return _it == o._it && _seg_idx == o._seg_idx;
    }

    bool operator!=(const ConstIterator& o) const {
        return !(*this == o);
    }

    const ConstIterator& operator++() {
        ++_it;
        next();
        return *this;
    }

private:
    void next() {
        typename SegmentT::Iterator seg_end{};
        const size_t ss_size = _parent->_segments.size();
        while (_it == seg_end) {      
            SegmentT* seg = nullptr;
            while (!seg) {
                ++_seg_idx;
                if (_seg_idx == ss_size) {
                    // end
                    return;
                }
                seg = _parent->_segments[_seg_idx].load(std::memory_order_acquire);        
            }
            assert(seg != nullptr);
            _it = seg->cbegin();
        }
    }

private:
    typename SegmentT::Iterator _it;
    size_t _seg_idx;
    const ConcurrentHashMap* _parent;
};

private:
    size_t hash(const key_type& k) const {
        size_t h = _hash_seed;
        h ^= HashFn()(k);

        // Spread bits to regularize both segment and index locations,
        // using variant of single-word Wang/Jenkins hash.
        h += (h << 15) ^ 0xffffcd7d;
        h ^= (h >> 10);
        h += (h <<  3);
        h ^= (h >>  6);
        h += (h <<  2) + (h << 14);
        return h ^ (h >> 16);
    }

    size_t hash_to_idx(size_t hash) const {
        return (hash >> _segment_sift) & _segment_mask;
    }

    SegmentT* ensure_segment(size_t idx) const {
        SegmentT* seg = _segments[idx].load(std::memory_order_acquire);
        if (seg == nullptr) {
            SegmentT* new_seg = new SegmentT(_seg_tab_size, _load_factor);
            if (!_segments[idx].compare_exchange_strong(seg, new_seg)) {
                delete new_seg;
            } else {
                seg = new_seg;
            }
        }
        assert(seg != nullptr && "seg is null");
        return seg;
    }

    bool get_internal(const key_type& key, mapped_type* out) const {
        size_t h = hash(key);
        size_t idx = hash_to_idx(h);
        SegmentT* seg = _segments[idx].load(std::memory_order_acquire);
        if (seg == nullptr) {
            return false;
        }
        return seg->get(key, h, out);
    }
};

} // namespace concurrency
} // namespace framework
} // namespace netaos
