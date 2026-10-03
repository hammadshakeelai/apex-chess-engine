#pragma once

#include "types.h"
#include "position.h"
#include <string>

namespace Apex {

#pragma pack(push, 1)
struct PolyglotEntry {
    uint64_t key;
    uint16_t move;
    uint16_t weight;
    uint32_t learn;
};
#pragma pack(pop)

class OpeningBook {
public:
    OpeningBook();
    ~OpeningBook();

    bool load(const std::string& filepath);
    void close();
    bool is_loaded() const { return entries != nullptr && count > 0; }

    Move probe(Position& pos);
    static uint64_t compute_key(const Position& pos);

private:
    PolyglotEntry* entries = nullptr;
    size_t count = 0;
};

extern OpeningBook GlobalBook;

} // namespace Apex
