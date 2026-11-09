// Native test for src/tracer/cubin_params.h.
//   c++ -std=c++17 -o /tmp/t tests/test_cubin_params.cpp && /tmp/t file.fatbin kernel expected_total
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <vector>
#include "../src/tracer/cubin_params.h"
bool readable(const void*, size_t) { return true; }
int main(int argc, char** argv) {
    if (argc < 4) return 2;
    std::ifstream in(argv[1], std::ios::binary);
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(in)), {});
    ParamLayout layout;
    const bool ok = param_layout(data.data() + (argc > 4 ? std::strtoul(argv[4], nullptr, 0) : 0), argv[2], layout);
    std::printf("%s ok=%d count=%u total=%u first=%u@%u\n", argv[2], ok, layout.count, layout.total, layout.size[0], layout.offset[0]);
    return ok && layout.total == std::strtoul(argv[3], nullptr, 0) && layout.size[0] == layout.total ? 0 : 1;
}
