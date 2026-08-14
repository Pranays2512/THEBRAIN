#pragma once
// Tiny debug-logging guard. Production code had scattered printf("DEBUG...") /
// printf("[...]") with non-thread-safe `static int print_count` gates that polluted
// stdout and couldn't be silenced without editing source. Route them through
// B2DEBUG(...) — silent unless the env var BRAIN2_DEBUG is set (checked once).
#include <cstdio>
#include <cstdlib>

namespace b2 {
inline bool debug_on() {
    static const bool on = (std::getenv("BRAIN2_DEBUG") != nullptr);
    return on;
}
}  // namespace b2

#define B2DEBUG(...) do { if (::b2::debug_on()) std::printf(__VA_ARGS__); } while (0)
