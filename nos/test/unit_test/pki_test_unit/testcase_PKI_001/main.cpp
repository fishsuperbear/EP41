#include <chrono>
#include <filesystem>
#include <iostream>
#include <thread>

#include "gtest/gtest.h"
#include "tsp_pki_cert_manage.h"
#include "tsp_pki_config.h"
#include "tsp_pki_log.h"
#include "tsp_pki_utils.h"

using namespace std;
using namespace hozon::netaos::tsp_pki;

#define PKI_WAIT_TIME 5

class PkiFuncTest : public ::testing::Test {
 protected:
    static void SetUpTestSuite() {
        cout << "=== SetUpTestSuite ===" << endl;
    }
    static void TearDownTestSuite() {
        cout << "=== TearDownTestSuite ===" << endl;
    }
    void SetUp() override {
        int res = system("export LD_LIBRARY_PATH=/app/lib:$LD_LIBRARY_PATH");
		sleep(PKI_WAIT_TIME);
    }
    void TearDown() override {}

 protected:
    
};

TEST_F(PkiFuncTest, PkiConfigYamlExits) {
    string config_path = "/app/runtime_service/pki_service/conf/pki_service.yaml";
    ASSERT_EQ(TspPkiUtils::IsFileExist(config_path), true);
}

TEST_F(PkiFuncTest, PkiReadConfig) {
    string config_path = "/app/runtime_service/pki_service/conf/pki_service.yaml";
    TspPkiConfig::Instance().SetConfigYaml(config_path);
    bool result = TspPkiConfig::Instance().ReadConfig();
    ASSERT_EQ(result, true);
}

TEST_F(PkiFuncTest, GetCertStatus) {
    TspPkiCertManage::CertStatus cert_status = TspPkiCertManage::Instance().GetCertStatus();
    ASSERT_EQ(cert_status, TspPkiCertManage::kCertStatusNone);
}

int main(int argc, char* argv[]) {
    TspPKILog::GetInstance().InitLogging("unit_test", "utest",
                                         TspPKILog::CryptoLogLevelType::CRYPTO_INFO,
                                         hozon::netaos::log::HZ_LOG2FILE,
                                         "./",
                                         10,
                                         20);
	TspPKILog::GetInstance().CreateLogger("pki_fuc_test");
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}