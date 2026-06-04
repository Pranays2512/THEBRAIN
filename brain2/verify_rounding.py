# Python // already does floor division, so 3x+48=20 => (20-48)//3 = -28//3 = -10 (floor)
print((20-48)//3)  # Python = -10
# C++ std::floor(-28.0/3.0) = std::floor(-9.333) = -10. Same.
import math
print(math.floor(-28/3))  # should be -10 too
