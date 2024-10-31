#pragma once

#include <functional>
#include <mutex>
#include <atomic>
#include <memory>
#include <vector>
#include <utility>

#include "spdlog/spdlog.h"
#include "framework/concurrency/detail/utility.h"
#include "framework/concurrency/detail/types.h"

namespace netaos {
namespace framework {
namespace concurrency {
namespace detail {

template <
    typename KeyType,
    typename ValueType>
struct NodeT {
    typedef std::pair<const KeyType, ValueType> value_type;

    NodeT(size_t h, const KeyType& k, const ValueType& v, SharedPtr<NodeT> n = nullptr)
        : _hash(h)
        , _item(k, v)
        , _next(n){
    }

    ~NodeT() noexcept {}

    value_type& get_item() {
        return _item;
    }

    const KeyType& get_key() const {
        return _item.first;
    }

    const ValueType& get_value() const {
        return _item.second;
    }

    size_t get_hash() const {
        return _hash;
    }

    SharedPtr<NodeT> load_next(std::memory_order order = std::memory_order_seq_cst) const {
        return _next.load(order);
    }

    void store_next(SharedPtr<NodeT> n, std::memory_order order = std::memory_order_seq_cst) {
        _next.store(n, order);
    }

    static SharedPtr<NodeT> create(size_t h, const KeyType& k, const ValueType& v, SharedPtr<NodeT> n = nullptr) {
        return SharedPtr<NodeT>(new NodeT(h, k, v, n));
    }

private:
    const size_t _hash;
    value_type _item;
    AtomicSharedPtr<NodeT> _next;
};

template<typename KeyType, typename ValueType>
class BucketsT {
    using Node = NodeT<KeyType, ValueType>;
    using NodePtr = SharedPtr<Node>;
    std::vector<AtomicSharedPtr<Node>> _table;

public:
    explicit BucketsT(size_t size) : _table(size) {
    }

    static SharedPtr<BucketsT> create(size_t size) {
        return SharedPtr<BucketsT>(new BucketsT(size));
    }

    AtomicSharedPtr<Node>& operator[](size_t i) {
        return _table[i];
    }

    NodePtr table_at(size_t i, std::memory_order order = std::memory_order_acquire) const {
        return _table[i].load(order);
    }

    void set_table_at(size_t i, NodePtr node, std::memory_order order = std::memory_order_release) {
        _table[i].store(node, order);
    }

    void insert_front(size_t i, NodePtr node) {
        NodePtr head = _table[i].load(std::memory_order_relaxed);
        node->store_next(head, std::memory_order_relaxed);
        _table[i].store(node, std::memory_order_relaxed);
    }

    size_t size() const {
        return _table.size();
    }
};

template <
    typename KeyType,
    typename ValueType,
    typename HashFn = std::hash<KeyType>,
    typename KeyEqual = std::equal_to<KeyType>
  >
//class alignas(64) Segment {
class Segment {
    using Node = NodeT<KeyType, ValueType>;
    using NodePtr = SharedPtr<Node>;
    using Buckets = BucketsT<KeyType, ValueType>;
    using BucketsPtr = SharedPtr<Buckets>;

public:
    using key_type = KeyType;
    using mapped_type = ValueType;
    using value_type = std::pair<const KeyType, ValueType>;
    using size_type = std::size_t;
    using hasher = HashFn;
    using key_equal = KeyEqual;

    class Iterator;

    static constexpr float DEFAULT_LOAD_FACTOR = 0.75;
    static constexpr size_t MAX_TABLE_SIZE = 1 << 30;

    Segment(size_t init_table_size, float load_factor) : _load_factor(load_factor) {
        if (init_table_size > MAX_TABLE_SIZE) {
            init_table_size = MAX_TABLE_SIZE;
        } else {
            init_table_size = folly::detail::next_pow_two(init_table_size);
        }
        _threshold = static_cast<size_t>(load_factor * init_table_size);
        // spdlog::info("init_table_size: {}, threshold: {}", init_table_size, _threshold);
        _buckets.store(Buckets::create(init_table_size), std::memory_order_relaxed);
    }

    ~Segment() noexcept {
    }

    void rehash(NodePtr node) {
        BucketsPtr old_table = _buckets.load(std::memory_order_relaxed);
        const size_t old_table_size = old_table->size();
        size_t new_table_size = old_table_size << 1;
        _threshold = static_cast<size_t>(_load_factor * new_table_size);
        // spdlog::info("_threshold = {}, new_table_size = {}", _threshold, new_table_size);
        size_t size_mask = new_table_size - 1;
        BucketsPtr new_table = Buckets::create(new_table_size);
        for (size_t i = 0; i < old_table_size; ++i) {
            NodePtr head = old_table->table_at(i, std::memory_order_relaxed);
            if (head == nullptr) {
                continue;
            }
            size_t idx = head->get_hash() & size_mask;
            NodePtr next = head->load_next(std::memory_order_relaxed);
            if (next == nullptr) {
                new_table->set_table_at(idx, head, std::memory_order_relaxed);
                continue;
            }
            size_t last_run_idx = idx;
            NodePtr last_run = head;
            for (NodePtr last = next; last; last = last->load_next(std::memory_order_relaxed)) {
                size_t k = last->get_hash() & size_mask;
                if (k != last_run_idx) {
                    last_run_idx = k;
                    last_run = last;
                }
            }

            new_table->set_table_at(last_run_idx, last_run);
        
            // Clone remaining nodes
            // Why use clone?
            for (NodePtr e = head; e != last_run; e = e->load_next(std::memory_order_relaxed)) {
                size_t k = e->get_hash() & size_mask;
                new_table->insert_front(k, Node::create(e->get_hash(), e->get_key(), e->get_value()));
            }
        }
        // 插入新节点
        size_t new_idx = node->get_hash() & size_mask;
        new_table->insert_front(new_idx, node);

        _buckets.store(new_table, std::memory_order_relaxed);
    }

public:

    // 向哈希表中插入元素，
    // only_if_absent为true时，键k存在，返回false, 键k不存在，插入且返回true
    // only_if_absent为false时，插入返回true, 覆盖返回false
    bool do_insert(const key_type& k, size_t hash, const mapped_type& v, bool only_if_absent) {
        std::lock_guard<std::mutex> lk(_m);
        auto buckets = _buckets.load(std::memory_order_relaxed);
        const size_t table_size = buckets->size();
        size_t index = (table_size - 1) & hash;
        NodePtr first = buckets->table_at(index, std::memory_order_relaxed);
        NodePtr prev = nullptr;
        key_equal eq;
        for (NodePtr e = first;;) {
            if (e != nullptr) {
                if (eq(k, e->get_key())) {
                // found
                    if (!only_if_absent) {
                        // 为保持原子性，不予许修改Node的value字段
                        // Node必须作为一个整体，即需要删除原节点，再插入新节点
                        NodePtr next = e->load_next(std::memory_order_relaxed);
                        NodePtr new_node = Node::create(hash, k, v, next);
                        if (prev == nullptr) {
                            // 说明e是头结点
                            buckets->set_table_at(index, new_node, std::memory_order_relaxed);
                        } else {
                            prev->store_next(new_node, std::memory_order_relaxed);
                        }
                        ++_mod_count;
                        return false;
                    }
                    return false;
                }
                prev = e;
                e = e->load_next(std::memory_order_relaxed);
            } else {
                // not found
                NodePtr new_node = Node::create(hash, k, v, first);
                size_t new_size = _size + 1;
                if (new_size > _threshold && table_size < MAX_TABLE_SIZE) {
                    // spdlog::info("rehash new_size = {}, thresold = {}", new_size, _threshold);
                    rehash(new_node);
                } else {
                    buckets->set_table_at(index, new_node, std::memory_order_relaxed);
                }
                ++_mod_count;
                _size = new_size;
                return true;
            }
        }
    }

    bool get(const key_type& k, size_t hash, mapped_type* out) {
        BucketsPtr buckets = _buckets.load(std::memory_order_acquire);
        size_t idx = (buckets->size() - 1) & hash;
        NodePtr e = buckets->table_at(idx, std::memory_order_acquire);
        key_equal eq;
        while (e != nullptr) {
            if (eq(k, e->get_key())) {
                if (out != nullptr) {
                *out = e->get_value();
                }
                return true;
            }
            e = e->load_next(std::memory_order_acquire);
        }
        return false;
    }

    template<typename MatchFunc>
    size_type erase_internal(const key_type& k, size_t hash, MatchFunc&& match) {
        std::lock_guard<std::mutex> lk(_m);
        BucketsPtr buckets = _buckets.load(std::memory_order_relaxed);
        size_t idx = (buckets->size() - 1) & hash;
        NodePtr e = buckets->table_at(idx, std::memory_order_relaxed);
        NodePtr pred = nullptr;
        key_equal eq;
        while (e != nullptr) {
            NodePtr next = e->load_next(std::memory_order_relaxed);
            if (eq(k, e->get_key())) {
                // match value
                if (!match(e->get_value())) {
                    return 0;
                }
                if (pred == nullptr) {
                    // 头结点匹配了
                    buckets->set_table_at(idx, next);
                } else {
                    pred->store_next(next, std::memory_order_relaxed);
                }
                ++_mod_count;
                --_size;
                return 1;
            }
            pred = e;
            e = next;
        }
    
        return 0;
    }

    size_type erase(const key_type& k, size_t hash) {
        return erase_internal(k, hash, [] (const mapped_type&) { return true; });
    }

    size_type erase_if_equal(const key_type& k, size_t hash, const mapped_type& expected) {
        return erase_internal(k, hash, [&expected] (const mapped_type& v) {
            return v == expected;
        });
    }

    template <typename MatchFunc>
    bool assign_internal(const key_type& k, size_t hash, const mapped_type& v, MatchFunc&& match) {
        std::lock_guard<std::mutex> lk(_m);
        auto buckets = _buckets.load(std::memory_order_relaxed);
        const size_t table_size = buckets->size();
        size_t index = (table_size - 1) & hash;
        NodePtr e = buckets->table_at(index, std::memory_order_relaxed);
        NodePtr prev = nullptr;
        key_equal eq;
        while (e != nullptr) {
            NodePtr next = e->load_next(std::memory_order_relaxed);
            if (eq(k, e->get_key())) {
                // match value
                if (!match(e->get_value())) {
                    return false;
                }
                NodePtr new_node = Node::create(hash, k, v, next);
                if (prev == nullptr) {
                    // 说明e是头结点
                    buckets->set_table_at(index, new_node, std::memory_order_relaxed);
                } else {
                    prev->store_next(new_node, std::memory_order_relaxed);
                }
                return true;
            }
            prev = e;
            e = next;
        }
        return false;
    }

    bool assign(const key_type& k, size_t hash, const mapped_type& v) {
        return assign_internal(k, hash, v, [] (const mapped_type&) { return true; });
    }

    bool assign_if_equal(const key_type& k, size_t hash, const mapped_type& expected, const mapped_type& desired) {
        return assign_internal(k, hash, desired, [&expected] (const mapped_type& v) { return v == expected; });
    }

    void for_each(std::function<void(value_type&)> action) {
        std::lock_guard<std::mutex> lk(_m);
        auto buckets = _buckets.load(std::memory_order_relaxed);
        const size_t table_size = buckets->size(); 
        for (size_t i = 0; i < table_size; ++i) {
            for (NodePtr e = buckets->table_at(i, std::memory_order_relaxed);
                    e != nullptr; e = e->load_next(std::memory_order_relaxed)) {
                action(e->get_item());
            }
        }       
    }

    void clear() {
        std::lock_guard<std::mutex> lk(_m);
        BucketsPtr buckets = _buckets.load(std::memory_order_relaxed);
        BucketsPtr new_buckets = Buckets::create(buckets->size());
        _buckets.store(new_buckets, std::memory_order_relaxed);
        ++_mod_count;
        _size = 0;
    }

    size_type size() const {
        return _size;
    }

    bool empty() const {
        return size() == 0;
    }

    Iterator cbegin() {
        auto buckets = _buckets.load(std::memory_order_acquire);
        Iterator res(nullptr, std::move(buckets), 0);
        res.next();
        return res;
    }

    Iterator cend() {
        return Iterator{};
    }

    class Iterator {
    public:
        Iterator(NodePtr curr, BucketsPtr buckets, int bucket_idx)
            : _curr(std::move(curr)), _buckets(std::move(buckets))
            , _bucket_count(_buckets->size()), _bucket_idx(bucket_idx) {

        }
        Iterator() {}
        ~Iterator() {}

        Iterator(const Iterator&) = delete;
        Iterator(Iterator&& o) noexcept 
            : _curr(std::move(o._curr))
            , _buckets(std::move(o._buckets))
            , _bucket_count(o._bucket_count)
            , _bucket_idx(o._bucket_idx) {
            
            o._bucket_count = 0;
            o._bucket_idx = 0;
        }
        
        Iterator& operator=(const Iterator&) = delete;

        Iterator& operator=(Iterator&& o) noexcept {
            if (this != &o) {
                _curr = std::move(o._curr);
                _buckets = std::move(o._buckets);
                _bucket_count = o._bucket_count;
                _bucket_idx = o._bucket_idx;
                o._bucket_count = 0;
                o._bucket_idx = 0;
            }
            return *this;
        }

        const value_type& operator*() const {
            assert(_curr);
            return _curr->get_item();
        }

        const value_type* operator->() const {
            assert(_curr);
            return &(_curr->get_item());
        }

        const Iterator& operator++() {
            assert(_curr);
            _curr = _curr->load_next(std::memory_order_acquire);
            if (_curr == nullptr) {
                ++_bucket_idx;
                next();
            }
            return *this;
        }

        bool operator==(const Iterator& o) const {
            return _curr == o._curr;
        }

        bool operator!=(const Iterator& o) const {
            return !(*this == o);
        }

        // 当当前节点为空时，移动至当前及后续桶中第一个不为空的头结点
        // 先决条件： _curr为空
        // 后置条件： _curr不为空，或达到结束（_bucket_idx == _bucket_count)
        void next() {
            assert(_curr == nullptr);
            for (;_bucket_idx < _bucket_count; ++_bucket_idx) {
                _curr = _buckets->table_at(_bucket_idx, std::memory_order_acquire);
                // spdlog::info("bucket_idx: {}, curr_node: {}", _bucket_idx, (void*)_curr.get());
                if (_curr) {
                break;
                }
            }
            assert(_curr || _bucket_idx == _bucket_count);
        }

    private:
        NodePtr _curr{nullptr};
        BucketsPtr _buckets{nullptr};
        int _bucket_count{0};
        int _bucket_idx{0};
    };

private:
    std::mutex _m;

    AtomicSharedPtr<Buckets> _buckets;
    size_t _size{0};
    size_t _mod_count{0};
    size_t _threshold;
    const float _load_factor;
};

} // namespace detail
} // namespace concurrency
} // namespace framework
} // namespace netaos
