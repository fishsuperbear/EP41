// Copyright 2020, Robotec.ai sp. z o.o.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <memory>

#include "rosbag2_cpp/cache/cache_consumer.hpp"
#include "rosbag2_cpp/logging.hpp"

namespace rosbag2_cpp
{
namespace cache
{

CacheConsumer::CacheConsumer(
  std::shared_ptr<MessageCacheInterface> message_cache,
  consume_callback_function_t consume_callback)
: message_cache_(message_cache),
  consume_callback_(consume_callback)
{
  consumer_thread_ = std::thread(&CacheConsumer::exec_consuming, this);
}

CacheConsumer::~CacheConsumer()
{
  stop();
}

void CacheConsumer::stop()
{
  message_cache_->begin_flushing();
  is_stop_issued_ = true;

  ROSBAG2_CPP_LOG_INFO << "Writing remaining messages from cache to the bag. It may take a while";

  if (consumer_thread_.joinable()) {
    consumer_thread_.join();
  }
  message_cache_->done_flushing();
}

void CacheConsumer::start()
{
  is_stop_issued_ = false;
  if (!consumer_thread_.joinable()) {
    consumer_thread_ = std::thread(&CacheConsumer::exec_consuming, this);
  }
}

void CacheConsumer::exec_consuming()
{
  bool exit_flag = false;
  bool flushing = false;
  while (!exit_flag) {
    message_cache_->wait_for_data();
    message_cache_->swap_buffers();
    // Get the current consumer buffer.
    auto consumer_buffer = message_cache_->get_consumer_buffer();
    consume_callback_(consumer_buffer->data());
    consumer_buffer->clear();
    message_cache_->release_consumer_buffer();

    if (flushing) {exit_flag = true;}  // this was the final run
    if (is_stop_issued_) {flushing = true;}  // run one final time to flush
  }
}

}  // namespace cache
}  // namespace rosbag2_cpp
