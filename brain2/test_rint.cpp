#include <cstdio>
#include <cmath>
int main() {
    float val = 4.0f / 32.0f; // 0.125
    double val_d = 4.0 / 32.0;
    char buf[32];
    snprintf(buf, sizeof(buf), "%.2f", std::rint(val_d * 100.0) / 100.0);
    printf("rint: %s\n", buf);
    
    val_d = 1.0 / 8.0; // 0.125
    snprintf(buf, sizeof(buf), "%.2f", std::rint(val_d * 100.0) / 100.0);
    printf("rint 1/8: %s\n", buf);
}
