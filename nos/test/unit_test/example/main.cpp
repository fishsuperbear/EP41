#include <iostream>
#include "gtest/gtest.h"
#include <memory>
class MMM {

	public:
	MMM(){};
	~MMM(){};

	int myadd(int a, int b) {
		return a + b;
	}
};

class MMMTest:public ::testing::Test{
protected:
	void SetUp() override {
		instance = new MMM();
	}

	void TearDown() override {
		delete instance;
	}
protected:
	MMM* instance;
public:
	
};

TEST(MMM, createInstance) {
	MMM* instance = new MMM();
	EXPECT_NE(instance, nullptr);
	int res = instance->myadd(2, 3);
	EXPECT_EQ(res, 5);
}

TEST(MMM, createInstance2) {
	MMM* instance = new MMM();
	EXPECT_NE(instance, nullptr);
	int res = instance->myadd(2, 3);
	EXPECT_EQ(res, 5);
}


int main(int argc, char* argv[])
{
	int aaa = 10;
	std::cout << aaa << std::endl;
	testing::InitGoogleTest(&argc,argv);
    int res = RUN_ALL_TESTS();
	return res;
}
