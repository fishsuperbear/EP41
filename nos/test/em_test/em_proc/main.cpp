#include <iostream>
#include <thread>
#include <chrono>
#include <string>
#include <cstring>
#include <signal.h>

sig_atomic_t g_stopFlag = 0;

void HandlerSignal(int32_t sig)
{
    g_stopFlag = 1;
}

void ActThread()
{
    std::cout << "em sub proc sample"<< std::endl;
    while (!g_stopFlag) {
        std::this_thread::sleep_for(std::chrono::seconds(1u));
    }
}


int main(int argc, char* argv[])
{
    signal(SIGTERM, HandlerSignal);

    std::thread act(ActThread);
    act.join();

    return 0;
}
