/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
 * Description: socket can interface data buffer
 */
#ifndef CANSTACK_DATA_BUFFER_H
#define CANSTACK_DATA_BUFFER_H

#include <condition_variable>
#include <map>


namespace hozon {
namespace netaos {
namespace canstack {

class DataBuffer {
 public:
  static DataBuffer& Instance();

  ~DataBuffer();

  // New interface
  // Wait data available at index [indx] or timeout specified by [seconds].
  // If seconds 0 means infinite. std::map is used to map index to buffer.
  bool WaitData(int index, int seconds);
  bool WaitData(int* indexes, int count, int seconds);
  bool WaitDataNotification(int index, int seconds);
  // Wait data to index [indx]. std::map is used to map index to buffer.
  // If the data at index already exist, it will be overwritten by this call.
  bool WriteData(int indx, unsigned char* data, int bytes, bool notify = false);
  bool WriteData(int indx, unsigned char* data, int bytes, int startbit,
                 bool notify = false);
  // Read the data at index [indx]. std::map is used to map index to buffer.
  bool ReadData(int indx, unsigned char* data_buf, int bytes,
                bool del_flg = false);
  // uint64_t ReadDataByPosition(int indx, int frame, uint8_t start_bit,
  //                             uint8_t length);
  // uint64_t LengthMask(uint8_t length);
  // Test whether has data at index [indx]. std::map is used to map index to
  // buffer.
  bool HasData(int indx);
  void NotifyAll();
  void ClearAll();
  size_t GetDataMapSize();
  const uint8_t defaultObjsize = 2 * 8U;

  // std::map<int, std::shared_ptr<BufNode>> data_map_;

 private:
  DataBuffer();

  struct BufNode {
    BufNode() : buf(nullptr), pos(0), size(0) {}

    BufNode(int s) : buf(nullptr), pos(0), size(s) {
      buf = new unsigned char[s];
    }

    ~BufNode() {
      if (buf) {
        delete[] buf;
        buf = nullptr;
        size = 0;
      }
    }
    unsigned char* buf;
    int pos = 0;
    int size = 0;
  };

  // std::map<int, std::shared_ptr<BufNode>> data_map_;
  std::condition_variable cv_;
  bool notified_;

  unsigned char* buf_;
//   int block_bytes_;
//   int block_num_;
//   bool grp_cache_;
  static DataBuffer* sinstance_;


 public:
  std::map<int, std::shared_ptr<BufNode>> data_map_;
};

}  // namespace canstack
}
}  // namespace hozon

#endif