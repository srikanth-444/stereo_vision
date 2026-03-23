#ifndef WEAK_PTR_UTILS_H
#define WEAK_PTR_UTILS_H

#include <memory>
#include <functional>

struct Frame; // forward declaration

struct WeakPtrHash {
    inline std::size_t operator()(const std::weak_ptr<Frame>& wp) const noexcept {
        auto sp = wp.lock();
        return std::hash<Frame*>()(sp.get());
    }
};

struct WeakPtrEqual {
    inline bool operator()(const std::weak_ptr<Frame>& a, const std::weak_ptr<Frame>& b) const noexcept {
        return a.lock().get() == b.lock().get();
    }
};

#endif 