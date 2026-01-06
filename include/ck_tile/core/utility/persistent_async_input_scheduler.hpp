// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

namespace ck_tile {

/// @brief Persistent async input scheduler structure for managing chunk-based tile scheduling.
///
/// This structure contains scheduling parameters for persistent async input processing,
/// enabling efficient chunk-based tile distribution across workgroups.
struct PersistentAsyncInputScheduler
{
    /// @brief Number of tiles per chunk in the M dimension.
    uint32_t tiles_per_chunk_m;

    /// @brief Pointer to chunk completion signals in device memory.
    uint32_t* chunk_signals;

    /// @brief Pivot tile index in the M dimension for scheduling decisions.
    uint32_t tile_idx_pivot_m;
};

} // namespace ck_tile
