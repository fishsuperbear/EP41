
#ifndef E2EXF_INDEX_H_
#define E2EXF_INDEX_H_

#include <array>
#include <cstdint>

#include "e2e/e2exf_cpp/include/e2exf_config.h"
#include "e2e/e2exf_cpp/include/e2exf_state.h"
namespace hozon {
namespace netaos {
namespace e2e {
constexpr std::uint8_t kDataIDLength{16U};

class E2EXf_Index final {
   public:
    E2EXf_Index() = default;

    ~E2EXf_Index() = default;

    E2EXf_Index(const E2EXf_Index&) = default;

    explicit E2EXf_Index(const std::uint32_t& DataID, const std::uint32_t& UniqueID) : dataid_(DataID), is_using_dataid_(true), unique_id_(UniqueID) {}

    explicit E2EXf_Index(const uint8_t DataIDList[16], const std::uint32_t& UniqueID) : is_using_dataid_(false), unique_id_(UniqueID) {
        for (uint8_t i = 0; i < 16; i++) dataidlist_[i] = DataIDList[i];
    }

    void SetDataID(const std::uint32_t DataID) {
        dataid_ = DataID;
        is_using_dataid_ = true;
    }

    void SetDataIDList(const uint8_t DataIDList[16]) {
        for (uint8_t i = 0; i < 16; i++) dataidlist_[i] = DataIDList[i];
        is_using_dataid_ = false;
    };

    const std::array<std::uint8_t, kDataIDLength>& GetDataIDList() const { return dataidlist_; }

    const std::uint32_t& GetDataID() const { return dataid_; }

    bool operator<(const E2EXf_Index& Index) const {
        if (is_using_dataid_) {
            if (dataid_ != Index.dataid_) return dataid_ > Index.dataid_;
        } else {
            for (uint8_t i = 0; i < kDataIDLength; i++) {
                if (dataidlist_[i] == Index.dataidlist_[i]) continue;
                return dataidlist_[i] > Index.dataidlist_[i];
            }
        }
        return unique_id_ > Index.unique_id_;
    }

    explicit operator bool() const;

    E2EXf_Index& operator=(const E2EXf_Index&) = default;

   private:
    std::uint32_t dataid_;
    std::array<std::uint8_t, kDataIDLength> dataidlist_;
    bool is_using_dataid_;

    std::uint32_t unique_id_;

    E2EXf_Index(const std::uint32_t& DataID, const std::array<std::uint8_t, kDataIDLength>& DataIDList, bool IsUsingDataID)
        : dataid_(DataID), dataidlist_{DataIDList}, is_using_dataid_{IsUsingDataID} {}
};

}  // namespace e2e
}  // namespace netaos
}  // namespace hozon

#endif
