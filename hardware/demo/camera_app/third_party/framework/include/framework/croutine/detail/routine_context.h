#ifndef CYBER_CROUTINE_ROUTINE_CONTEXT_H_
#define CYBER_CROUTINE_ROUTINE_CONTEXT_H_

#include <cstdlib>
#include <cstring>
#include <iostream>

#include "framework/common/log.h"

extern "C" {
extern void ctx_swap(void**, void**) asm("ctx_swap");
};

namespace netaos {
namespace framework {
namespace croutine {

constexpr size_t STACK_SIZE = 2 * 1024 * 1024;
#if defined __aarch64__
constexpr size_t REGISTERS_SIZE = 160;
#else
constexpr size_t REGISTERS_SIZE = 56;
#endif

typedef void (*func)(void*);
struct RoutineContext {
  char stack[STACK_SIZE];
  char* sp = nullptr;
#if defined __aarch64__
} __attribute__((aligned(16)));
#else
};
#endif

void MakeContext(const func& f1, const void* arg, RoutineContext* ctx);

inline void SwapContext(char** src_sp, char** dest_sp) {
  ctx_swap(reinterpret_cast<void**>(src_sp), reinterpret_cast<void**>(dest_sp));
}

}  // namespace croutine
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_CROUTINE_ROUTINE_CONTEXT_H_
