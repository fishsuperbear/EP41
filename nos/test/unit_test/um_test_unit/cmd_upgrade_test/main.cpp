#include <iostream>
#include <thread>
#include <chrono>
#include <string>
#include <signal.h>
#include <unistd.h>
#include <thread>
#include <memory>

#include "update_manager/log/update_manager_logger.h"
#include "update_manager/state_machine/update_state_machine.h"
#include "update_manager/cmd_line_upgrade/cmd_upgrade_manager.h"
#include "update_manager/update_check/update_check.h"
#include "update_manager/config/update_settings.h"
#include "update_manager/record/ota_store.h"

#include "gtest/gtest.h"

using namespace hozon::netaos::update;
using namespace std;

class CmdUpgradeTest:public ::testing::Test{

protected:
	void SetUp() override {
		UM_DEBUG << "call SetUp";
		instance->Init();
	}
	void TearDown() override {
		UM_DEBUG << "call TearDown";
		instance->Deinit();
	}  
protected:
	CmdUpgradeManager* instance = CmdUpgradeManager::Instance();
};

// 获取当前状态，切换状态后，再获取当前状态
TEST_F(CmdUpgradeTest, getUpdateStatusTest)
{
	UpdateStateMachine::Instance()->InitStateMap();
    UpdateStateMachine::Instance()->SetInitialState(State::NORMAL_IDLE);

	std::shared_ptr<common_req_t> req = make_shared<common_req_t>();
	std::shared_ptr<update_status_resp_t> resp = make_shared<update_status_resp_t>();
	req->platform = 1;
	instance->UpdateStatusMethod(req, resp);
	EXPECT_EQ(resp->error_msg, "success");
	ASSERT_EQ(resp->error_code, 0);
	EXPECT_EQ(resp->update_status, "NORMAL_IDLE");

	UpdateStateMachine::Instance()->SwitchState(State::OTA_PRE_UPDATE);
	instance->UpdateStatusMethod(req, resp);
	EXPECT_EQ(resp->error_msg, "success");
	ASSERT_EQ(resp->error_code, 0);
	EXPECT_EQ(resp->update_status, "OTA_PRE_UPDATE");

	UpdateStateMachine::Instance()->Deinit();
}
// cmd的前置条件检查接口
TEST_F(CmdUpgradeTest, preCheckTest)
{
	UpdateCheck::Instance().Init();
	std::shared_ptr<common_req_t> req = make_shared<common_req_t>();
	req->platform = 1;
	std::shared_ptr<precheck_resp_t> resp = make_shared<precheck_resp_t>();
	instance->PreCheckMethod(req, resp);
	EXPECT_EQ(resp->error_msg, "success");
	ASSERT_EQ(resp->error_code, 0);
	EXPECT_EQ(resp->gear, true);
	EXPECT_EQ(resp->speed, false);
	EXPECT_EQ(resp->space, false);

	UpdateCheck::Instance().Deinit();
}
// 设置初始状态IDLE，查询进度的场景
TEST_F(CmdUpgradeTest, getProgressTest1)
{
	UpdateStateMachine::Instance()->InitStateMap();
    UpdateStateMachine::Instance()->SetInitialState(State::NORMAL_IDLE);
	UpdateCheck::Instance().Init();
	std::shared_ptr<common_req_t> req = make_shared<common_req_t>();
	req->platform = 1;
	std::shared_ptr<progress_resp_t> resp = make_shared<progress_resp_t>();
	instance->ProgressMethod(req, resp);
	EXPECT_NE(resp->error_msg, "success");
	ASSERT_EQ(resp->error_code, -1);
	EXPECT_EQ(resp->progress, 0);

	UpdateCheck::Instance().Deinit();
	UpdateStateMachine::Instance()->Deinit();
}
// 设置初始状态OTA_UPDATE_FAILED，查询进度的场景
TEST_F(CmdUpgradeTest, getProgressTest2)
{
	UpdateStateMachine::Instance()->InitStateMap();
    UpdateStateMachine::Instance()->SetInitialState(State::OTA_UPDATE_FAILED);
	UpdateCheck::Instance().Init();
	std::shared_ptr<common_req_t> req = make_shared<common_req_t>();
	req->platform = 1;
	std::shared_ptr<progress_resp_t> resp = make_shared<progress_resp_t>();
	instance->ProgressMethod(req, resp);
	EXPECT_NE(resp->error_msg, "success");
	ASSERT_EQ(resp->error_code, -2);
	EXPECT_EQ(resp->progress, 0);

	UpdateCheck::Instance().Deinit();
	UpdateStateMachine::Instance()->Deinit();
}
// 设置初始状态OTA_PRE_UPDATE，查询进度的场景
TEST_F(CmdUpgradeTest, getProgressTest3)
{
	UpdateStateMachine::Instance()->InitStateMap();
    UpdateStateMachine::Instance()->SetInitialState(State::OTA_PRE_UPDATE);
	UpdateCheck::Instance().Init();
	std::shared_ptr<common_req_t> req = make_shared<common_req_t>();
	req->platform = 1;
	std::shared_ptr<progress_resp_t> resp = make_shared<progress_resp_t>();
	instance->ProgressMethod(req, resp);
	EXPECT_EQ(resp->error_msg, "success");
	ASSERT_EQ(resp->error_code, 0);
	EXPECT_EQ(resp->progress, 0);

	UpdateCheck::Instance().Deinit();
	UpdateStateMachine::Instance()->Deinit();
}
// 设置初始状态OTA_UPDATED，查询进度的场景
TEST_F(CmdUpgradeTest, getProgressTest4)
{
	UpdateStateMachine::Instance()->InitStateMap();
    UpdateStateMachine::Instance()->SetInitialState(State::OTA_UPDATED);
	UpdateCheck::Instance().Init();
	std::shared_ptr<common_req_t> req = make_shared<common_req_t>();
	req->platform = 1;
	std::shared_ptr<progress_resp_t> resp = make_shared<progress_resp_t>();
	instance->ProgressMethod(req, resp);
	EXPECT_EQ(resp->error_msg, "success");
	ASSERT_EQ(resp->error_code, 0);
	EXPECT_EQ(resp->progress, 90);

	UpdateCheck::Instance().Deinit();
	UpdateStateMachine::Instance()->Deinit();
}
// 设置初始状态OTA_ACTIVED，查询进度的场景
TEST_F(CmdUpgradeTest, getProgressTest5)
{
	UpdateStateMachine::Instance()->InitStateMap();
    UpdateStateMachine::Instance()->SetInitialState(State::OTA_ACTIVED);
	UpdateCheck::Instance().Init();
	std::shared_ptr<common_req_t> req = make_shared<common_req_t>();
	req->platform = 1;
	std::shared_ptr<progress_resp_t> resp = make_shared<progress_resp_t>();
	instance->ProgressMethod(req, resp);
	EXPECT_EQ(resp->error_msg, "success");
	ASSERT_EQ(resp->error_code, 0);
	EXPECT_EQ(resp->progress, 100);

	UpdateCheck::Instance().Deinit();
	UpdateStateMachine::Instance()->Deinit();
}
// 设置初始状态OTA_UPDATING，查询进度的场景
// 此状态查询进度相当于直接发送22 01 07,这个case留到diag_agent的测试中，目前做简要测试
TEST_F(CmdUpgradeTest, getProgressTest6)
{
	UpdateStateMachine::Instance()->InitStateMap();
    UpdateStateMachine::Instance()->SetInitialState(State::OTA_UPDATING);
	UpdateCheck::Instance().Init();
	std::shared_ptr<common_req_t> req = make_shared<common_req_t>();
	req->platform = 1;
	std::shared_ptr<progress_resp_t> resp = make_shared<progress_resp_t>();
	instance->ProgressMethod(req, resp);
	EXPECT_EQ(resp->error_msg, "success");
	ASSERT_EQ(resp->error_code, 0);
	EXPECT_EQ(resp->progress, 100);

	std::this_thread::sleep_for(std::chrono::seconds(5));
	UpdateCheck::Instance().Deinit();
	UpdateStateMachine::Instance()->Deinit();
}
// 设置初始状态OTA_ACTIVED，查询进度的场景
TEST_F(CmdUpgradeTest, startUpdateTest1)
{
	UpdateStateMachine::Instance()->InitStateMap();
    UpdateStateMachine::Instance()->SetInitialState(State::OTA_ACTIVED);
	UpdateCheck::Instance().Init();
	std::shared_ptr<start_update_req_t> req = make_shared<start_update_req_t>();
	req->package_path = "/ota/a.zip";
	req->start_with_precheck = true;
	std::shared_ptr<start_update_resp_t> resp = make_shared<start_update_resp_t>();
	instance->StartUpdateMethod(req, resp);

	EXPECT_NE(resp->error_msg, "success");
	ASSERT_EQ(resp->error_code, -2);

	UpdateCheck::Instance().Deinit();
	UpdateStateMachine::Instance()->Deinit();
}
// 设置初始状态NORMAL_IDLE，但是preCheck失败的场景
TEST_F(CmdUpgradeTest, startUpdateTest2)
{
	UpdateStateMachine::Instance()->InitStateMap();
    UpdateStateMachine::Instance()->SetInitialState(State::NORMAL_IDLE);
	UpdateCheck::Instance().Init();
	std::shared_ptr<start_update_req_t> req = make_shared<start_update_req_t>();
	req->package_path = "/ota/a.zip";
	req->start_with_precheck = true;
	std::shared_ptr<start_update_resp_t> resp = make_shared<start_update_resp_t>();
	instance->StartUpdateMethod(req, resp);

	EXPECT_NE(resp->error_msg, "success");
	ASSERT_EQ(resp->error_code, -1);

	UpdateCheck::Instance().Deinit();
	UpdateStateMachine::Instance()->Deinit();
}
// 设置初始状态NORMAL_IDLE，但是preCheck失败的场景
TEST_F(CmdUpgradeTest, startUpdateTest3)
{
	UpdateStateMachine::Instance()->InitStateMap();
    UpdateStateMachine::Instance()->SetInitialState(State::NORMAL_IDLE);
	UpdateCheck::Instance().Init();
	std::shared_ptr<start_update_req_t> req = make_shared<start_update_req_t>();
	req->package_path = "/ota/a.zip";
	req->start_with_precheck = false;
	std::shared_ptr<start_update_resp_t> resp = make_shared<start_update_resp_t>();
	instance->StartUpdateMethod(req, resp);

	EXPECT_NE(resp->error_msg, "success");
	ASSERT_EQ(resp->error_code, -3);

	UpdateCheck::Instance().Deinit();
	UpdateStateMachine::Instance()->Deinit();
}
// 手动precheck，再调用升级, 此时升级包不存在或者校验出问题的场景
TEST_F(CmdUpgradeTest, startUpdateTest4)
{
	UpdateStateMachine::Instance()->InitStateMap();
    UpdateStateMachine::Instance()->SetInitialState(State::NORMAL_IDLE);
	UpdateCheck::Instance().Init();

	std::shared_ptr<common_req_t> req1 = make_shared<common_req_t>();
	req1->platform = 1;
	std::shared_ptr<precheck_resp_t> resp1 = make_shared<precheck_resp_t>();
	instance->PreCheckMethod(req1, resp1);
	EXPECT_EQ(resp1->error_msg, "success");
	ASSERT_EQ(resp1->error_code, 0);
	EXPECT_EQ(resp1->gear, true);
	EXPECT_EQ(resp1->speed, false);
	EXPECT_EQ(resp1->space, false);

	std::shared_ptr<start_update_req_t> req2 = make_shared<start_update_req_t>();
	req2->package_path = "/ota/a.zip";
	req2->start_with_precheck = false;
	std::shared_ptr<start_update_resp_t> resp2 = make_shared<start_update_resp_t>();
	instance->StartUpdateMethod(req2, resp2);

	EXPECT_NE(resp2->error_msg, "success");
	ASSERT_EQ(resp2->error_code, -4);

	UpdateCheck::Instance().Deinit();
	UpdateStateMachine::Instance()->Deinit();
}

// 手动precheck，再调用升级, 此时升级包校验成功，开始升级的场景
TEST_F(CmdUpgradeTest, startUpdateTest5)
{
	UpdateSettings::Instance().Init();
	UpdateStateMachine::Instance()->InitStateMap();
    UpdateStateMachine::Instance()->SetInitialState(State::NORMAL_IDLE);
	UpdateCheck::Instance().Init();

	std::shared_ptr<common_req_t> req1 = make_shared<common_req_t>();
	req1->platform = 1;
	std::shared_ptr<precheck_resp_t> resp1 = make_shared<precheck_resp_t>();
	instance->PreCheckMethod(req1, resp1);
	EXPECT_EQ(resp1->error_msg, "success");
	ASSERT_EQ(resp1->error_code, 0);
	EXPECT_EQ(resp1->gear, true);
	EXPECT_EQ(resp1->speed, false);
	EXPECT_EQ(resp1->space, false);

	std::shared_ptr<start_update_req_t> req2 = make_shared<start_update_req_t>();
	req2->package_path = "/ota/a.zip";
	req2->start_with_precheck = false;
	std::shared_ptr<start_update_resp_t> resp2 = make_shared<start_update_resp_t>();
	instance->StartUpdateMethod(req2, resp2);

	EXPECT_EQ(resp2->error_msg, "success");
	ASSERT_EQ(resp2->error_code, 0);

	UpdateCheck::Instance().Deinit();
	UpdateStateMachine::Instance()->Deinit();
	UpdateSettings::Instance().Deinit();
}
// 查询版本的Case
TEST_F(CmdUpgradeTest, getVersionTest)
{
	OTAStore::Instance()->Init();
	std::shared_ptr<common_req_t> req = make_shared<common_req_t>();
	req->platform = 1;
	std::shared_ptr<get_version_resp_t> resp = make_shared<get_version_resp_t>();
	instance->GetVersionMethod(req, resp);
	// TODO

	EXPECT_EQ(resp->error_msg, "success");
	ASSERT_EQ(resp->error_code, 0);
	OTAStore::Instance()->Deinit();
}

// 除了OTA_UPDATE_FAILED OTA_ACTIVED，其他状态都返回错误
TEST_F(CmdUpgradeTest, finishTest1) 
{
	UpdateStateMachine::Instance()->InitStateMap();
    UpdateStateMachine::Instance()->SetInitialState(State::OTA_UPDATE_FAILED);
	std::shared_ptr<common_req_t> req = make_shared<common_req_t>();
	req->platform = 1;
	std::shared_ptr<start_finish_resp_t> resp = make_shared<start_finish_resp_t>();
	instance->StartFinishMethod(req, resp);

	EXPECT_EQ(resp->error_msg, "success");
	ASSERT_EQ(resp->error_code, 0);

	UpdateStateMachine::Instance()->Deinit();
}
TEST_F(CmdUpgradeTest, finishTest2)
{
	UpdateStateMachine::Instance()->InitStateMap();
    UpdateStateMachine::Instance()->SetInitialState(State::OTA_ACTIVED);
	std::shared_ptr<common_req_t> req = make_shared<common_req_t>();
	req->platform = 1;
	std::shared_ptr<start_finish_resp_t> resp = make_shared<start_finish_resp_t>();
	instance->StartFinishMethod(req, resp);

	EXPECT_EQ(resp->error_msg, "success");
	ASSERT_EQ(resp->error_code, 0);

	UpdateStateMachine::Instance()->Deinit();
}
TEST_F(CmdUpgradeTest, finishTest3)
{
	UpdateStateMachine::Instance()->InitStateMap();
    UpdateStateMachine::Instance()->SetInitialState(State::NORMAL_IDLE);
	std::shared_ptr<common_req_t> req = make_shared<common_req_t>();
	req->platform = 1;
	std::shared_ptr<start_finish_resp_t> resp = make_shared<start_finish_resp_t>();
	instance->StartFinishMethod(req, resp);

	EXPECT_NE(resp->error_msg, "success");
	ASSERT_EQ(resp->error_code, -1);

	UpdateStateMachine::Instance()->Deinit();
}
TEST_F(CmdUpgradeTest, finishTest4)
{
	UpdateStateMachine::Instance()->InitStateMap();
    UpdateStateMachine::Instance()->SetInitialState(State::OTA_PRE_UPDATE);
	std::shared_ptr<common_req_t> req = make_shared<common_req_t>();
	req->platform = 1;
	std::shared_ptr<start_finish_resp_t> resp = make_shared<start_finish_resp_t>();
	instance->StartFinishMethod(req, resp);

	EXPECT_NE(resp->error_msg, "success");
	ASSERT_EQ(resp->error_code, -1);

	UpdateStateMachine::Instance()->Deinit();
}
TEST_F(CmdUpgradeTest, finishTest5)
{
	UpdateStateMachine::Instance()->InitStateMap();
    UpdateStateMachine::Instance()->SetInitialState(State::OTA_UPDATING);
	std::shared_ptr<common_req_t> req = make_shared<common_req_t>();
	req->platform = 1;
	std::shared_ptr<start_finish_resp_t> resp = make_shared<start_finish_resp_t>();
	instance->StartFinishMethod(req, resp);

	EXPECT_NE(resp->error_msg, "success");
	ASSERT_EQ(resp->error_code, -1);

	UpdateStateMachine::Instance()->Deinit();
}
TEST_F(CmdUpgradeTest, finishTest6)
{
	UpdateStateMachine::Instance()->InitStateMap();
    UpdateStateMachine::Instance()->SetInitialState(State::OTA_UPDATED);
	std::shared_ptr<common_req_t> req = make_shared<common_req_t>();
	req->platform = 1;
	std::shared_ptr<start_finish_resp_t> resp = make_shared<start_finish_resp_t>();
	instance->StartFinishMethod(req, resp);

	EXPECT_NE(resp->error_msg, "success");
	ASSERT_EQ(resp->error_code, -1);

	UpdateStateMachine::Instance()->Deinit();
}
TEST_F(CmdUpgradeTest, finishTest7)
{
	UpdateStateMachine::Instance()->InitStateMap();
    UpdateStateMachine::Instance()->SetInitialState(State::OTA_ACTIVING);
	std::shared_ptr<common_req_t> req = make_shared<common_req_t>();
	req->platform = 1;
	std::shared_ptr<start_finish_resp_t> resp = make_shared<start_finish_resp_t>();
	instance->StartFinishMethod(req, resp);

	EXPECT_NE(resp->error_msg, "success");
	ASSERT_EQ(resp->error_code, -1);

	UpdateStateMachine::Instance()->Deinit();
}

int main(int argc, char* argv[])
{
    hozon::netaos::log::InitLogging("unit_test","utest",hozon::netaos::log::LogLevel::kTrace, hozon::netaos::log::HZ_LOG2FILE | hozon::netaos::log::HZ_LOG2CONSOLE, "./", 10, 20);
	UpdateManagerLogger::GetInstance().CreateLogger("cmd_upgrade_test");
	UM_DEBUG << "test um cmd_upgrade_test start ...";

	testing::InitGoogleTest(&argc,argv);
    return  RUN_ALL_TESTS();
}
