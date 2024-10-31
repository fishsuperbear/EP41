#pragma once
#include <functional>
#include <iostream>

class ConvertBase {
   public:
    virtual void RegistMessageCallback(std::function<void(std::string, std::string, int64_t time, std::vector<std::uint8_t>)> callbackFunction) = 0;
    virtual void Convert(std::string input_file, const std::vector<std::string>& exclude_topics, const std::vector<std::string>& topics) = 0;
    ConvertBase(){};
    virtual ~ConvertBase(){};
};
