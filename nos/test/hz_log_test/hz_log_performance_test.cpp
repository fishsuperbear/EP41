#include <signal.h>
#include <stdio.h>
#include <iostream>
#include <thread>
#include <vector>

#include "logging.h"

#ifdef BUILD_FOR_ORIN
#include "log_block_producer.h"
using hozon::netaos::logblock::LogBlockProducer;
#endif

using namespace hozon::netaos::log;

bool stopFlag = false;

std::uint32_t GetTickCount() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
}

uint64_t NowMS() {
#ifdef BUILD_FOR_ORIN
    uint64_t io_tsc_ns;
    uint64_t tsc;
    __asm__ __volatile__ ("mrs %[tsc], cntvct_el0" : [tsc] "=r" (tsc));
    io_tsc_ns = tsc * 32;

    return io_tsc_ns;
#else
    return GetTickCount();
#endif
}

void SigHandler(int signum) {
    std::cout << "--- log test sigHandler enter, signum [" << signum << "] ---" << std::endl;
    stopFlag = 1;
}


// 256 Byte
// 100W
void test1_1(bool is_sleep, int32_t time_ms, uint64_t max_count = 1000000) {
    std::string data(256, 'A');
    data += "\n";

    auto test1_1 = CreateLogger("TEST1-1", "test 1.1", LogLevel::kDebug);

    if (is_sleep) {
        auto begin_time = NowMS();
        std::uint64_t count{0};
        while (count <= max_count && !stopFlag) {
            //LogBlockProducer::Instance().Write("TEST1-1", data.c_str(), data.size());
            test1_1->LogDebug() << data;
            count++;

            auto curr_end = NowMS();
            if (curr_end - begin_time < time_ms * 1000000ull * count / max_count) {
                struct timespec duration;
                duration.tv_sec = 0;
                duration.tv_nsec = (1000000ull * time_ms * count / max_count) - (curr_end - begin_time);
                nanosleep(&duration, NULL);
            }
        }
        return;
    }

    std::uint64_t count{0};
    while (count <= max_count && !stopFlag) {
        test1_1->LogDebug() << data;
        count++;
    }
}
// 500 Byte
// 100W
void test1_2(bool is_sleep, int32_t time_ms, uint64_t max_count = 1000000) {
    std::string data(500, 'A');
    data += "\n";

    auto test1_1 = CreateLogger("TEST1-1", "test 1.1", LogLevel::kDebug);

    if (is_sleep) {
        auto begin_time = NowMS();
        std::uint64_t count{0};
        while (count <= max_count && !stopFlag) {
            //LogBlockProducer::Instance().Write("TEST1-1", data.c_str(), data.size());
            test1_1->LogDebug() << data;
            count++;

            auto curr_end = NowMS();
            if (curr_end - begin_time < time_ms * 1000000ull * count / max_count) {
                struct timespec duration;
                duration.tv_sec = 0;
                duration.tv_nsec = (1000000ull * time_ms * count / max_count) - (curr_end - begin_time);
                nanosleep(&duration, NULL);
            }
        }
        return;
    }

    std::uint64_t count{0};
    while (count <= max_count && !stopFlag) {
        test1_1->LogDebug() << data;
        count++;
    }
}
// 1000 Byte
// 100W
void test1_3(bool is_sleep, int32_t time_ms, uint64_t max_count = 1000000) {
    std::string data(1000, 'A');
    data += "\n";

    auto test1_1 = CreateLogger("TEST1-1", "test 1.1", LogLevel::kDebug);

    if (is_sleep) {
        auto begin_time = NowMS();
        std::uint64_t count{0};
        while (count <= max_count && !stopFlag) {
            test1_1->LogDebug() << data;
            //LogBlockProducer::Instance().Write("TEST1-1", data.c_str(), data.size());
            count++;

            auto curr_end = NowMS();
            if (curr_end - begin_time < time_ms * 1000000ull * count / max_count) {
                struct timespec duration;
                duration.tv_sec = 0;
                duration.tv_nsec = (1000000ull * time_ms * count / max_count) - (curr_end - begin_time);
                nanosleep(&duration, NULL);
            }
        }
        return;
    }

    std::uint64_t count{0};
    while (count <= max_count && !stopFlag) {
        test1_1->LogDebug() << data;
        count++;
    }
}
// 1950 Byte
// 100W
void test1_4(bool is_sleep, int32_t time_ms, uint64_t max_count = 1000000) {
    std::string data(1950, 'A');
    data += "\n";

    auto test1_1 = CreateLogger("TEST1-1", "test 1.1", LogLevel::kDebug);

    if (is_sleep) {
        auto begin_time = NowMS();
        std::uint64_t count{0};
        while (count <= max_count && !stopFlag) {
            //LogBlockProducer::Instance().Write("TEST1-1", data.c_str(), data.size());
            test1_1->LogDebug() << data;
            count++;

            auto curr_end = NowMS();
            if (curr_end - begin_time < time_ms * 1000000ull * count / max_count) {
                struct timespec duration;
                duration.tv_sec = 0;
                duration.tv_nsec = (1000000ull * time_ms * count / max_count) - (curr_end - begin_time);
                nanosleep(&duration, NULL);
            }
        }
        return;
    }

    std::uint64_t count{0};
    while (count <= max_count && !stopFlag) {
        test1_1->LogDebug() << data;
        count++;
    }
}
// 3950 Byte
// 100W
void test1_5(bool is_sleep, int32_t time_ms, uint64_t max_count = 1000000) {
    std::string data(3950, 'A');
    data += "\n";

    auto test1_1 = CreateLogger("TEST1-1", "test 1.1", LogLevel::kDebug);

    if (is_sleep) {
        auto begin_time = NowMS();
        std::uint64_t count{0};
        while (count <= max_count && !stopFlag) {
            test1_1->LogDebug() << data;
            //LogBlockProducer::Instance().Write("TEST1-1", data.c_str(), data.size());
            count++;

            auto curr_end = NowMS();
            if (curr_end - begin_time < time_ms * 1000000ull * count / max_count) {
                struct timespec duration;
                duration.tv_sec = 0;
                duration.tv_nsec = (1000000ull * time_ms * count / max_count) - (curr_end - begin_time);
                nanosleep(&duration, NULL);
            }
        }
        return;
    }

    std::uint64_t count{0};
    while (count <= max_count && !stopFlag) {
        test1_1->LogDebug() << data;
        count++;
    }
}
// 7950 Byte
// 100W
void test1_6(bool is_sleep, int32_t time_ms, uint64_t max_count = 1000000) {
    std::string data(7950, 'A');
    data += "\n";

    auto test1_1 = CreateLogger("TEST1-1", "test 1.1", LogLevel::kDebug);

    if (is_sleep) {
        auto begin_time = NowMS();
        std::uint64_t count{0};
        while (count <= max_count && !stopFlag) {
            test1_1->LogDebug() << data;
            //LogBlockProducer::Instance().Write("TEST1-1", data.c_str(), data.size());
            count++;

            auto curr_end = NowMS();
            if (curr_end - begin_time < time_ms * 1000000ull * count / max_count) {
                struct timespec duration;
                duration.tv_sec = 0;
                duration.tv_nsec = (1000000ull * time_ms * count / max_count) - (curr_end - begin_time);
                nanosleep(&duration, NULL);
            }
        }
        return;
    }

    std::uint64_t count{0};
    while (count <= max_count && !stopFlag) {
        test1_1->LogDebug() << data;
        count++;
    }
}
// 256 Byte
// 1000W
void test2_1(bool is_sleep, int32_t time_ms, uint64_t max_count = 10000000) {
    std::string data(256, 'A');
    data += "\n";

    auto test1_1 = CreateLogger("TEST1-1", "test 1.1", LogLevel::kDebug);

    if (is_sleep) {
        auto begin_time = NowMS();
        std::uint64_t count{0};
        while (count <= max_count && !stopFlag) {
            //LogBlockProducer::Instance().Write("TEST1-1", data.c_str(), data.size());
            test1_1->LogDebug() << data;
            count++;

            auto curr_end = NowMS();
            if (curr_end - begin_time < time_ms * 1000000ull * count / max_count) {
                struct timespec duration;
                duration.tv_sec = 0;
                duration.tv_nsec = (1000000ull * time_ms * count / max_count) - (curr_end - begin_time);
                nanosleep(&duration, NULL);
            }
        }
        return;
    }

    std::uint64_t count{0};
    while (count <= max_count && !stopFlag) {
        test1_1->LogDebug() << data;
        count++;
    }
}
// 256 Byte
// 2000W
void test2_2(bool is_sleep, int32_t time_ms, uint64_t max_count = 20000000) {
    std::string data(256, 'A');
    data += "\n";

    auto test1_1 = CreateLogger("TEST1-1", "test 1.1", LogLevel::kDebug);

    if (is_sleep) {
        auto begin_time = NowMS();
        std::uint64_t count{0};
        while (count <= max_count && !stopFlag) {
            //LogBlockProducer::Instance().Write("TEST1-1", data.c_str(), data.size());
            test1_1->LogDebug() << data;
            count++;

            auto curr_end = NowMS();
            if (curr_end - begin_time < time_ms * 1000000ull * count / max_count) {
                struct timespec duration;
                duration.tv_sec = 0;
                duration.tv_nsec = (1000000ull * time_ms * count / max_count) - (curr_end - begin_time);
                nanosleep(&duration, NULL);
            }
        }
        return;
    }

    std::uint64_t count{0};
    while (count <= max_count && !stopFlag) {
        test1_1->LogDebug() << data;
        count++;
    }
}
// 256 Byte
// 4000W
void test2_3(bool is_sleep, int32_t time_ms, uint64_t max_count = 40000000) {
    std::string data(256, 'A');
    data += "\n";

    auto test1_1 = CreateLogger("TEST1-1", "test 1.1", LogLevel::kDebug);

    if (is_sleep) {
        auto begin_time = NowMS();
        std::uint64_t count{0};
        while (count <= max_count && !stopFlag) {
            //LogBlockProducer::Instance().Write("TEST1-1", data.c_str(), data.size());
            test1_1->LogDebug() << data;
            count++;

            auto curr_end = NowMS();
            if (curr_end - begin_time < time_ms * 1000000ull * count / max_count) {
                struct timespec duration;
                duration.tv_sec = 0;
                duration.tv_nsec = (1000000ull * time_ms * count / max_count) - (curr_end - begin_time);
                nanosleep(&duration, NULL);
            }
        }
        return;
    }

    std::uint64_t count{0};
    while (count <= max_count && !stopFlag) {
        test1_1->LogDebug() << data;
        count++;
    }
}

// float
// 100W
void test3_1(bool is_sleep, int32_t time_ms, uint64_t max_count = 1000000) {
    float value = 3.14f;
    float value2 = 3.15f;

    auto test1_1 = CreateLogger("TEST1-1", "test 1.1", LogLevel::kDebug);

    if (is_sleep) {
        auto begin_time = NowMS();
        std::uint64_t count{0};
        char data[8];
        *((float*)data) = value;
        *((float*)(data + 4 )) = value2;
        while (count <= max_count && !stopFlag) {
            //LogBlockProducer::Instance().Write("TEST1-1", data, 8);
            test1_1->LogDebug() << value << value2;
            count++;

            auto curr_end = NowMS();
            if (curr_end - begin_time < time_ms * 1000000ull * count / max_count) {
                struct timespec duration;
                duration.tv_sec = 0;
                duration.tv_nsec = (1000000ull * time_ms * count / max_count) - (curr_end - begin_time);
                nanosleep(&duration, NULL);
            }
        }
        return;
    }

    std::uint64_t count{0};
    while (count <= max_count && !stopFlag) {
        test1_1->LogDebug() << value << value2;
        count++;
    }
}
// double
// 100W
void test3_2(bool is_sleep, int32_t time_ms, uint64_t max_count = 1000000) {
    double value = 3.14159265359;

    auto test1_1 = CreateLogger("TEST1-1", "test 1.1", LogLevel::kDebug);

    if (is_sleep) {
        auto begin_time = NowMS();
        std::uint64_t count{0};
        char data[8];
        *((double*)data) = value;
        while (count <= max_count && !stopFlag) {
            //LogBlockProducer::Instance().Write("TEST1-1", data, 8);
            test1_1->LogDebug() << value;
            count++;

            auto curr_end = NowMS();
            if (curr_end - begin_time < time_ms * 1000000ull * count / max_count) {
                struct timespec duration;
                duration.tv_sec = 0;
                duration.tv_nsec = (1000000ull * time_ms * count / max_count) - (curr_end - begin_time);
                nanosleep(&duration, NULL);
            }
        }
        return;
    }

    std::uint64_t count{0};
    while (count <= max_count && !stopFlag) {
        test1_1->LogDebug() << value;
        count++;
    }
}
// int
// 100W
void test3_3(bool is_sleep, int32_t time_ms, uint64_t max_count = 1000000) {
    int idate = 1990;
    int idate2 = 1991;

    auto test1_1 = CreateLogger("TEST1-1", "test 1.1", LogLevel::kDebug);

    if (is_sleep) {
        auto begin_time = NowMS();
        std::uint64_t count{0};
        char data[8];
        *((int*)data) = idate;
        *((int*)(data + 4)) = idate2;
        while (count <= max_count && !stopFlag) {
            //LogBlockProducer::Instance().Write("TEST1-1", data, 8);
            test1_1->LogDebug() << idate << idate2;
            count++;

            auto curr_end = NowMS();
            if (curr_end - begin_time < time_ms * 1000000ull * count / max_count) {
                struct timespec duration;
                duration.tv_sec = 0;
                duration.tv_nsec = (1000000ull * time_ms * count / max_count) - (curr_end - begin_time);
                nanosleep(&duration, NULL);
            }
        }
        return;
    }

    std::uint64_t count{0};
    while (count <= max_count && !stopFlag) {
        test1_1->LogDebug() << idate << idate2;
        count++;
    }
}
// int && string 1：1
// 100W
void test3_4(bool is_sleep, int32_t time_ms, uint64_t max_count) {
    int value1 = 199;
    std::string data(4, 'A');

    auto test1_1 = CreateLogger("TEST1-1", "test 1.1", LogLevel::kDebug);

    if (is_sleep) {
        auto begin_time = NowMS();
        std::uint64_t count{0};
        char data1[8];
        *((int*)data1) = value1;
        memcpy(data1 + 4, data.c_str(), data.size());
        while (count <= max_count && !stopFlag) {
            //LogBlockProducer::Instance().Write("TEST1-1", data1, 8);
            test1_1->LogDebug() << value1 << data1;
            count++;

            auto curr_end = NowMS();
            if (curr_end - begin_time < time_ms * 1000000ull * count / max_count) {
                struct timespec duration;
                duration.tv_sec = 0;
                duration.tv_nsec = (1000000ull * time_ms * count / max_count) - (curr_end - begin_time);
                nanosleep(&duration, NULL);
            }
        }
        return;
    }

    std::uint64_t count{0};
    while (count <= max_count && !stopFlag) {
        test1_1->LogDebug() << value1 << data;
        count++;
    }
}


// 打印 help
void printHelp()
{
    std::cout << R"(
        用法: 
            ./hz_log_performance_test 21 1
            参数1： 测试用例 （目前取值范围有1-6， 11-13， 21-24）
            参数2： 启动线程数量
            argv3:  is sleep mode? 0/1
            argv4:  time_ms
            argv5:  logto, default=2(HZ_LOG2FILE)
    )" << std::endl;
}

#define RUN_THREAD_IMPL(TAG, NUM_THREADS, IS_SLEEP, TIME_MS, MAX_COUNT) \
    do { \
        std::vector<std::thread> threads; \
        for (int i = 0; i < NUM_THREADS; ++i) { \
            threads.push_back(std::thread(test##TAG, IS_SLEEP, TIME_MS, MAX_COUNT)); \
        } \
        for (std::thread& thread : threads) { \
            thread.join(); \
        } \
    } while (0);

#define RUN_THREAD(CASE_NUM, CASE_SEQ, NUM_THREADS, IS_SLEEP, TIME_MS, MAX_COUNT) \
            RUN_THREAD_IMPL(CASE_NUM##_##CASE_SEQ, NUM_THREADS, IS_SLEEP, TIME_MS, MAX_COUNT)

int main(int argc, char* argv[]) {
    signal(SIGINT, SigHandler);
    signal(SIGTERM, SigHandler);
    if (argc < 3) {
        std::cout << "please check input params num. " << std::endl;
        printHelp();
        return -1;
    }
    std::string msg = argv[1];
    std::string thread_num = argv[2];
    bool is_sleep = false;
    int32_t time_ms = 1000;
    int logto = HZ_LOG2FILE;
    if (argc > 3) {
        is_sleep = atoi(argv[3]) > 0 ? true : false;
        time_ms = atoi(argv[4]);
        logto = atoi(argv[5]);
    }

    int num_threads = std::stoi(thread_num);;
    InitLogging("LOG-TEST", "hz log test", LogLevel::kTrace, logto, "/opt/usr/hz_map/hzlog/", 10, 10);

    if (msg == "1") {
        RUN_THREAD(1, 1, num_threads, is_sleep, time_ms, 1000000);
    } else if (msg == "2") {
        RUN_THREAD(1, 2, num_threads, is_sleep, time_ms, 1000000);
    } else if (msg == "3") {
        RUN_THREAD(1, 3, num_threads, is_sleep, time_ms, 1000000);
    } else if (msg == "4") {
        RUN_THREAD(1, 4, num_threads, is_sleep, time_ms, 1000000);
    } else if (msg == "5") {   
        RUN_THREAD(1, 5, num_threads, is_sleep, time_ms, 1000000);
    } else if (msg == "6") {
        RUN_THREAD(1, 6, num_threads, is_sleep, time_ms, 1000000);
    } else if (msg == "11") {
        RUN_THREAD(2, 1, num_threads, is_sleep, time_ms, 10000000);
    } else if (msg == "12") {
        RUN_THREAD(2, 2, num_threads, is_sleep, time_ms, 20000000);
    } else if (msg == "13") {
        RUN_THREAD(2, 3, num_threads, is_sleep, time_ms, 40000000);
    } else if (msg == "21") {
        RUN_THREAD(3, 1, num_threads, is_sleep, time_ms, 1000000);
    } else if (msg == "22") {
        RUN_THREAD(3, 2, num_threads, is_sleep, time_ms, 1000000);
    } else if (msg == "23") {
        RUN_THREAD(3, 3, num_threads, is_sleep, time_ms, 1000000);
    } else if (msg == "24") {
        RUN_THREAD(3, 4, num_threads, is_sleep, time_ms, 1000000);
    } else if (msg == "30") {
        RUN_THREAD(1, 1, num_threads, is_sleep, time_ms, 20000);
    } else if (msg == "31") {
        RUN_THREAD(1, 1, num_threads, is_sleep, time_ms, 10000);
    } else if (msg == "32") {
        RUN_THREAD(1, 1, num_threads, is_sleep, time_ms, 2000);
    }

    return 0;
}
