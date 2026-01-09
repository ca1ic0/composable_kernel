// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"

namespace ck_tile {

struct workgroup_barrier
{
    CK_TILE_DEVICE workgroup_barrier(uint32_t* ptr) : base_ptr(ptr) {}

    CK_TILE_DEVICE uint32_t ld(uint32_t offset = 0)
    {
        return __atomic_load_n(base_ptr + offset, __ATOMIC_RELAXED);
    }

    CK_TILE_DEVICE void wait_eq(uint32_t value, uint32_t offset = 0)
    {
        if(threadIdx.x == 0)
        {
            while(ld(offset) != value) {}
        }
        __syncthreads();
    }

    CK_TILE_DEVICE void wait_lt(uint32_t value, uint32_t offset = 0)
    {
        if(threadIdx.x == 0)
        {
            while(ld(offset) < value) {}
        }
        __syncthreads();
    }

    CK_TILE_DEVICE void wait_signal(uint32_t offset = 0)
    {
        if(threadIdx.x == 0)
        {
#ifdef CK_TILE_DEBUG
            uint32_t iterations = 0;
            constexpr uint32_t max_iterations = 10000000;  // ~300-430ms timeout
#endif
            while(ld(offset) == 0)
            {
                __builtin_amdgcn_s_sleep(1);  // ~64 cycles (~30-43 ns)
#ifdef CK_TILE_DEBUG
                if(++iterations >= max_iterations)
                {
                    __builtin_trap();  // Signal timeout
                }
#endif
            }
        }
        __syncthreads();
    }

    CK_TILE_DEVICE void wait_set(uint32_t compare, uint32_t value, uint32_t offset = 0)
    {
        if(threadIdx.x == 0)
        {
            while(atomicCAS(base_ptr + offset, compare, value) != compare) {}
        }
        __syncthreads();
    }

    // enter critical zoon, assume buffer is zero when launch kernel
    CK_TILE_DEVICE void aquire(uint32_t offset = 0) { wait_set(offset, 0, 1); }

    // exit critical zoon, assume buffer is zero when launch kernel
    CK_TILE_DEVICE void release(uint32_t offset = 0) { wait_set(offset, 1, 0); }

    CK_TILE_DEVICE void inc(uint32_t offset = 0)
    {
        __syncthreads();
        if(threadIdx.x == 0)
        {
            atomicAdd(base_ptr + offset, 1);
        }
    }

    uint32_t* base_ptr;
};

} // namespace ck_tile
