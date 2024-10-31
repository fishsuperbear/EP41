#ifndef ARA_COM_INTERNAL_TYPES_H_
#define ARA_COM_INTERNAL_TYPES_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

#include <vector>
#include <stdint.h>

#define INITIALIZER(f) \
static void f(void) __attribute__((constructor)); \
static void f(void)

#define DEINITIALIZER(f) \
static void f(void) __attribute__((destructor)); \
static void f(void)

using BufferList = std::vector<std::vector<uint8_t>>;

enum SerializationType {
    SerializationType_Any = 0,
    SerializationType_SOMEIP = 1,
    SerializationType_DDS = 2,
    SerializationType_IPC = 4,
    SerializationType_S2S = 8
};

enum LoadConfigurationType {
    LoadConfigurationType_CPP = 0,
    LoadConfigurationType_JSON = 1
};

class TagBase {
public:
    virtual ~TagBase() = default;
    int32_t serialization_type;
};

#endif // ARA_COM_INTERNAL_TYPES_H_
/* EOF */
