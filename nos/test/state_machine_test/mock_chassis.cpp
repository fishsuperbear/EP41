#include <unistd.h>
#include <atomic>
#include <thread>
#include "proto/soc/chassis.pb.h"
#include "proto/statemachine/state_machine.pb.h"
#include "cm/include/proto_cm_writer.h"
#include "log/include/default_logger.h"

#define COMMAND_MAX_LEN 1024
#define ARGS_MAX_NUM 10
#define ARG_MAX_LEN 128

static hozon::netaos::cm::ProtoCMWriter<hozon::soc::Chassis> chassis_writer;
static hozon::soc::Chassis chassis{};
static std::atomic<bool>   _quit(false);

void help() {
    printf("Usage: state [options] ... \n");
    printf("Options: \n");
    printf("  h, -h, --help       ------> Explain the meaning of parameters. \n");
    printf("  s, -s, --pasw       ------> start.\n");
    printf("  i, -i, --parkingin  ------> parking.\n");
    printf("  o, -o, --parkingout ------> reset.\n");
    printf("  q, -q, --quit       ------> quit.\n");
}

void parse_args(char *line, int &argc, char **argv) {

    while (*line != '\0' && argc < ARGS_MAX_NUM - 1) {
        while (*line == ' ' || *line == '\t' || *line == '\n') {
            *line++ = '\0';
        }
        argv[argc++] = line;
        while (*line != '\0' && *line != ' ' && *line != '\t' && *line != '\n') {
            line++;
        }
    }
    argv[argc] = NULL;
}

void parse(int argc, char **args) {
    for (auto index = 0; index < argc; ) {
        if (strcmp(args[index], "h") == 0 || strcmp(args[index], "-h") == 0 || strcmp(args[index], "--help") == 0) {
            help();
            break;
        } else if (strcmp(args[index], "s") == 0 || strcmp(args[index], "-s") == 0 || strcmp(args[index], "--pasw") == 0) {

            chassis.mutable_avm_pds_info()->set_cdcs11_pasw(1);             // pasw按键
            chassis.mutable_avm_pds_info()->set_cdcs11_apa_functionmode(1); // FAPA泊入

            index += 1;
        } else if (strcmp(args[index], "i") == 0 || strcmp(args[index], "-i") == 0 || strcmp(args[index], "--parkingin") == 0) {

            chassis.mutable_avm_pds_info()->set_cdcs11_parkinginreq(1);     // 点击了开始泊入按键

            index += 1;
        } else if (strcmp(args[index], "o") == 0 || strcmp(args[index], "-o") == 0 || strcmp(args[index], "--parkingout") == 0) {

            chassis.mutable_avm_pds_info()->set_cdcs11_apa_functionmode(2);  // FAPA泊出
            chassis.mutable_avm_pds_info()->set_cdcs11_parkingoutreq(1);     // 点击了开始泊出按键

            index += 1;
        } else if (strcmp(args[index], "q") == 0 || strcmp(args[index], "-q") == 0 || strcmp(args[index], "--quit") == 0) {
            _quit = true;
            printf("-> quit \n");
            return;
        } else {
            printf("not correct command: %s\n", args[index]);
            index += 1;
        }
    }
}

void send() {
    int i = 0;
    while (!_quit.load()) {
        DF_LOG_INFO << "Write data seq=" << i++;
        // printf("==========%d\n", i++);
        int ret1 = chassis_writer.Write(chassis);
        if (ret1 < 0) {
            DF_LOG_ERROR << "Fail to write chassis " << ret1;
        }

        sleep(1);
    }
}

int main(int argc, char*argv[]) {
    char command[COMMAND_MAX_LEN];
    char *args[ARGS_MAX_NUM];
    
    DefaultLogger::GetInstance().InitLogger();

    int32_t ret1 = chassis_writer.Init(0, "/soc/chassis");
    if (ret1 < 0) {
        DF_LOG_ERROR << "Fail to init chasssis writer " << ret1;
        return -1;
    }

    std::thread th = std::thread(send);

    while (1) {
        int agc = 0;
        printf("please continue to input :\n");
        if (fgets(command, COMMAND_MAX_LEN, stdin) == NULL) {
            continue;
        }

        if (strlen(command) == 1 && command[0] == '\n') {
            continue;
        }

        // 去除行末的换行符
        if (command[strlen(command) - 1] == '\n') {
            command[strlen(command) - 1] = '\0';
        }

        // 解析command line
        parse_args(command, agc, args);

        // 解析args
        parse(agc, args);
        if (_quit)
            break;
    }

    th.join();
    chassis_writer.Deinit();
    DF_LOG_INFO << "Deinit end." << ret1;
}