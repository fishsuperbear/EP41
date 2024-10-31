/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 * Description: thread safe unordered_map implementation
 * Create: 2019-11-19
 */
#ifndef INC_ARA_VCC_SAFEMAP_H
#define INC_ARA_VCC_SAFEMAP_H
#include <map>
#include <mutex>
namespace vrtf {
namespace vcc {
namespace utils {
template <class Key, class Value>
class SafeMap {
public:
    void Insert(const Key &key, const Value &value)
    {
        std::lock_guard<std::mutex> lock {mutex_};
        static_cast<void>(safeMap.emplace(key, value));
    }

    void Erase(const typename std::map<Key, Value>::const_iterator iter)
    {
        std::lock_guard<std::mutex> lock {mutex_};
        static_cast<void>(safeMap.erase(iter));
    }

    void Erase(const Key &key)
    {
        std::lock_guard<std::mutex> lock {mutex_};
        static_cast<void>(safeMap.erase(key));
    }

    size_t Size() const
    {
        std::lock_guard<std::mutex> lock {mutex_};
        return safeMap.size();
    }

    bool Empty() const
    {
        std::lock_guard<std::mutex> lock {mutex_};
        return safeMap.empty();
    }

    void Clear()
    {
        std::lock_guard<std::mutex> lock {mutex_};
        safeMap.clear();
    }

    bool Find(const Key &k, Value &v) const /* risk on ref to items destroyed */
    {
        std::lock_guard<std::mutex> lock{mutex_};
        auto it = safeMap.find(k);
        if (it == safeMap.end()) {
            return false;
        }
        v = it->second;
        return true;
    }

    bool Find(const Key &k) const /* risk on ref to items destroyed */
    {
        std::lock_guard<std::mutex> lock{mutex_};
        auto it = safeMap.find(k);
        return !(it == safeMap.end());
    }
    // user add lock to keep iterator is valid
    typename std::map<Key, Value>::iterator Begin() noexcept
    {
        std::lock_guard<std::mutex> lock{mutex_};
        return safeMap.begin();
    }

    typename std::map<Key, Value>::const_iterator CBegin() noexcept
    {
        std::lock_guard<std::mutex> lock{mutex_};
        return safeMap.cbegin();
    }

    typename std::map<Key, Value>::iterator End() noexcept
    {
        std::lock_guard<std::mutex> lock{mutex_};
        return safeMap.end();
    }

    typename std::map<Key, Value>::const_iterator CEnd() noexcept
    {
        std::lock_guard<std::mutex> lock{mutex_};
        return safeMap.cend();
    }

    Value& operator [](const Key& k)
    {
        std::lock_guard<std::mutex> lock{mutex_};
        return safeMap[k];
    }

private:
    std::map<Key, Value> safeMap; /* Fixme about support unordered_map */
    mutable std::mutex mutex_;
};
}
}
}
#endif
