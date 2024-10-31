#include <iostream>
#include <string>

#ifndef _ADVC_HMAC_H_
#define _ADVC_HMAC_H_

int HmacEncode(const char *algo,
               const std::string&  key, unsigned int key_length,
               const char *input, unsigned int input_length,
               unsigned char *&output, unsigned int &output_length);


std::string sha256(std::string str);

#endif
