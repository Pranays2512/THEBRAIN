#include <cstdio>
#include <cmath>
int main() {
    float val = 4.0f / 32.0f;
    char buf[32];
    snprintf(buf, sizeof(buf), "%.2f", val);
    printf("snprintf: %s\n", buf);
    printf("val: %.20f\n", val);
    printf("val * 100: %.20f\n", val * 100.0f);
    printf("floor(val * 100 + 0.5): %.0f\n", std::floor(val * 100.0f + 0.5f));
}
