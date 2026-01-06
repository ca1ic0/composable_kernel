// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

namespace ck_tile {

// the fields may be set on the client side
struct PersistentAsyncInputScheduler
{
    uint32_t tiles_per_chunk_m = 0;

    uint32_t* chunk_signals = nullptr;

    uint32_t tile_idx_pivot_m = 0;
};

} // namespace ck_tile
