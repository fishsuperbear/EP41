#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string>
#include <map>
#include <iostream>
#include <csignal>
#include "symmetric_crypto.h"
#include "crypto_tool_log.h"

using namespace hozon::netaos::log;
using namespace hozon::netaos::crypto;

sig_atomic_t g_stopFlag = 0;


void SigHandler(int signum)
{

    std::cout << "Received signal: " << signum  << ". Quitting\n";
    signal(SIGINT, SIG_IGN);
    signal(SIGTERM, SIG_IGN);
    signal(SIGQUIT, SIG_IGN);
    signal(SIGHUP, SIG_IGN);

    g_stopFlag = true;

    signal(SIGINT, SIG_DFL);
    signal(SIGTERM, SIG_DFL);
    signal(SIGQUIT, SIG_DFL);
    signal(SIGHUP, SIG_DFL);
}

/** Sets up signal handler.*/
void SigSetup(void)
{
    struct sigaction action
    {
    };
    action.sa_handler = SigHandler;

    sigaction(SIGINT, &action, nullptr);
    sigaction(SIGTERM, &action, nullptr);
    sigaction(SIGQUIT, &action, nullptr);
    sigaction(SIGHUP, &action, nullptr);
}

std::map<std::string,std::string> enfiles,defiles;
int main(int argc, char*argv[])
{
 	SigSetup();

    CryptoToolLog::GetInstance().setLogLevel(static_cast<int32_t>(CryptoToolLog::CryptoLogLevelType::CRYPTO_INFO));
    CryptoToolLog::GetInstance().InitLogging("crypto_tool", "crypto tool",
                                                CryptoToolLog::CryptoLogLevelType::CRYPTO_INFO,                  //the log level of application
                                                hozon::netaos::log::HZ_LOG2CONSOLE,  //the output log mode
                                                "/opt/usr/log/soc_log/",                                                                  //the log file directory, active when output log to file
                                                10,                                                                    //the max number log file , active when output log to file
                                                20                                                                     //the max size of each  log file , active when output log to file
    );
    CryptoToolLog::GetInstance().CreateLogger("CryptoTool");
    CRYTOOL_INFO<< "Crypto tool Log init finish.";

    std::shared_ptr<SymmetricCrypto> crypto = std::make_shared<SymmetricCrypto>();
	crypto->Init();


    if(argc < 2 ){
	    crypto->Help();
	}else if(argc == 2){
		if(strcmp("-h",argv[1]) == 0 || strcmp("-H",argv[1]) == 0){
            crypto->Help();
		}
	}else if (argc > 2){
        if(strcmp("-en",argv[1]) == 0){
			// std::string pname = argv[2];
            // size_t pos = pname.find("Process");
			for(int i=2;i<argc;i++){
				std::string plain_file(argv[i]);
				std::string en_file = plain_file + "_en";
				std::pair<std::string,std::string> temp(plain_file,en_file);
				enfiles.insert(temp);
			}

		}else if (strcmp("-de",argv[1]) == 0){
			for(int i=2;i<argc;i++){
				std::string plain_file(argv[i]);
				std::string en_file = plain_file + "_de";
				std::pair<std::string,std::string> temp(plain_file,en_file);
				defiles.insert(temp);
			}
		}
	}

	for(auto it:enfiles){
		std::cout<<"need to encrypto file:"<<it.first<<std::endl;
		crypto->encrypto(it.first);
	}

    for(auto it:defiles){
		std::cout<<"need to decrypto file:"<<it.first<<std::endl;
		crypto->decrypto(it.first);
	}

    crypto->DeInit();
    return 0;
}
