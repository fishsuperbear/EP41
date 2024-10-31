#include <thread>
#include "player.h"

using namespace hozon::netaos::bag;

int main(int argc, char** argv) {
    Player* player = new Player();

    std::thread recThread = std::thread([player] {             //新起线程
        std::this_thread::sleep_for(std::chrono::seconds(5));  //5s后调用Stop停止录制
        if (nullptr != player) {
            player->Stop();
        }
    });

    PlayerOptions play_options;
    play_options.uri = "/home/sw/work/netaos/2023-07-31_10-20-51/2023-07-31_10-20-51_0.mcap";
    player->Start(play_options);  //开始播放, 主线程阻塞，直到player Stop()被调用
    delete player;                //释放资源
    player = nullptr;

    recThread.join();
    return 0;
}