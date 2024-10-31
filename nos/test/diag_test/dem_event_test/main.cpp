
#include "demNotifyPhm.h"
#include "testDiagEvent.h"
#include "demEventPub.h"
#include <iostream>
#include <thread>

int32_t main(int32_t argc, char* argv[])
{
    printf("diag test\n");
    DiagServerEventPub cDiagServerEventPub;
    TestDiagEvent cTestDiagEvent;
    DemEventPub cPub;

	while(1) {
        int inputNum = -1;
        printf("\n--------1.occur 2.recover 3.sendFaultRecover 4.send test msg--------\n");
        if (scanf("%d", &inputNum)) {}
        int iTmp = 0;
        do {
            iTmp = getchar();
        } while ((iTmp != '\n') && (iTmp != EOF));
        printf("inputNum:%d\n", inputNum);

        switch(inputNum) {
        case 1:
            {
                std::thread pubThd = std::thread([&](){
                    cPub.runOccur();
                });
                pubThd.join();
            }
            break;
        case 2:
            {
                std::thread pubThd = std::thread([&](){
                    cPub.runRecover();
                });
                pubThd.join();
            }
            break;
        case 3:
            {
                cDiagServerEventPub.sendFaultEvent(800401, 0x00);
            }
            break;
        case 4:
            {
                // int iTmp = 0;
                // do {
                //     iTmp = getchar();
                // } while ((iTmp != '\n') && (iTmp != EOF));
                char data[128] = {0};
                if(fgets(data, 128, stdin)) {};

                std::vector<uint8_t> param;
                char * pch = strtok (data, " ");
                while (pch != NULL) {
                    param.push_back(std::stoul(pch, nullptr, 16));
                    pch = strtok (NULL, " ");
                }

                for(auto i: param) {
                    printf("0x%x ", i);
                }
                printf("\n");

                cTestDiagEvent.sendFaultEvent(param[0], param);
            }
            break;

        default:
            break;
    	}
    }

    return 0;
}
