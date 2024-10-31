#include <iostream>

class Println {
private:
    bool logEnabled;

public:
    Println() : logEnabled(true) {}

    void enableLog(bool enable) {
        logEnabled = enable;
    }

    template<typename T>
    Println& operator<<(const T& value) {
        if (logEnabled) {
            std::cout << value;
        }
        return *this;
    }

    Println& operator<<(std::ostream& (*manipulator)(std::ostream&)) {
        if (logEnabled) {
            manipulator(std::cout);
        }
        return *this;
    }
};

#define PRINTLN Println() << std::endl