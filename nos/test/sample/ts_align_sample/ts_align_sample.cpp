#include "adf/include/ts_align/ts_align.h"
#include <iostream>
#include <unistd.h>

struct TestData {
    std::string str;
};

void OnAlignSucc(hozon::netaos::adf::TsAlignDataBundle& bundle) {
    std::cout << "In callback" << std::endl;
    std::shared_ptr<TestData> a = std::static_pointer_cast<TestData>(bundle["A"]);
    if (a) {
        std::cout << "A: " << a->str << std::endl;
    }

    std::shared_ptr<TestData> b = std::static_pointer_cast<TestData>(bundle["B"]);
    if (b) {
        std::cout << "B: " << b->str << std::endl;
    }

    std::shared_ptr<TestData> c = std::static_pointer_cast<TestData>(bundle["C"]);
    if (c) {
        std::cout << "C: " << c->str << std::endl;
    }
}

int main() {
    // 1. init TsAlign
    hozon::netaos::adf::TsAlign ts_align;
    ts_align.Init(5, 100, OnAlignSucc);

    // 2. register all sources
    ts_align.RegisterSource("A");
    ts_align.RegisterSource("B");
    ts_align.RegisterSource("C");

    // 3. push data with timestamp and process in callback
    std::shared_ptr<TestData> a1(new TestData{"a1"});
    ts_align.Push("A", a1, 10000);

    std::shared_ptr<TestData> b1(new TestData{"b1"});
    ts_align.Push("B", b1, 10010);

    std::shared_ptr<TestData> c1(new TestData{"c1"});
    ts_align.Push("C", c1, 10020);

    // 4. wait callback && deinit
    sleep(10);
    ts_align.Deinit();
}