#include "core/brain.hpp"
int main() {
    brain2::Brain b(8, 8, 16);
    b.episodic.save("checkpoints/stage5_math/episodic.bin");
    return 0;
}
