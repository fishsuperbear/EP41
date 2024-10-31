#ifndef NONCOPYABLE_H
#define NONCOPYABLE_H
#pragma once

namespace advc {

    class NonCopyable {
    protected:
        NonCopyable() {}

        ~NonCopyable() {}

    private:  // emphasize the following members are private
        NonCopyable(const NonCopyable &);

        const NonCopyable &operator=(const NonCopyable &);
    };
}  // namespace advc

#endif  // NONCOPYABLE_H
