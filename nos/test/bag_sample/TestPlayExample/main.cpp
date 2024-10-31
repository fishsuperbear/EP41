#include "player.hpp"
#include <rosbag2_cpp/reader.hpp>

int main()
{
    auto player = std::make_shared<rosbag2_transport::Player>(
        std::move(reader), storage_options_, play_options_);
    player->play();
}