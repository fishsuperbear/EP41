#ifndef _CUTILS_HPP
#define _CUTILS_HPP
#include <algorithm>
#include <vector>
#include <mutex>

class IdxManager {
    public:
        // init idx list
        IdxManager(size_t max_idx) : max_idx_(max_idx)
    {
        for (size_t i = 1; i <= max_idx; ++i) {
            free_idxs_.push_back(i);
        }
        used_idxs_.reserve(max_idx);
    }

        // get a free idx
        size_t get_free_idx()
        {
            std::lock_guard<std::mutex> lock(mtx_);
            if (free_idxs_.empty()) {
                return -1; // no enough resource
            }
            size_t idx = free_idxs_.back();
            free_idxs_.pop_back();
            used_idxs_.push_back(idx);
            return idx;
        }

        // put the idx from used to free
        void release_idx(size_t idx)
        {
            std::lock_guard<std::mutex> lock(mtx_);
            auto it = std::find(used_idxs_.begin(), used_idxs_.end(), idx);
            if (it != used_idxs_.end()) {
                used_idxs_.erase(it);
                free_idxs_.push_back(idx);
            }
        }

        // get free idx list
        std::vector<size_t> get_free_idxs() const
        {
            std::lock_guard<std::mutex> lock(mtx_);
            return free_idxs_;
        }

    private:
        const size_t max_idx_;
        std::vector<size_t> free_idxs_;
        std::vector<size_t> used_idxs_;
        mutable std::mutex mtx_;
};
#endif
