//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include <optix.h>

#include "optixTriangle.h"
#include <cuda/helpers.h>

#include <sutil/vec_math.h>
#define lights_num 2

extern "C" {
__constant__ Params params;
}
__constant__ float3 lights[lights_num] = {{-1.0f, 0.0f, 1.0f},{1.0f, 0.0f, 1.0f}};



static __forceinline__ __device__ void setPayload( float3 p )
{
    optixSetPayload_0( __float_as_uint( p.x ) );
    optixSetPayload_1( __float_as_uint( p.y ) );
    optixSetPayload_2( __float_as_uint( p.z ) );
}


static __forceinline__ __device__ void computeRay( uint3 idx, uint3 dim, float3& origin, float3& direction )
{
    const float3 U = params.cam_u;
    const float3 V = params.cam_v;
    const float3 W = params.cam_w;
    const float2 d = 2.0f * make_float2(
            static_cast<float>( idx.x ) / static_cast<float>( dim.x ),
            static_cast<float>( idx.y ) / static_cast<float>( dim.y )
            ) - 1.0f;

    origin    = params.cam_eye;
    direction = normalize( d.x * U + d.y * V + W );
}


extern "C" __global__ void __raygen__rg()
{
    // Lookup our location within the launch grid
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    // Map our launch idx to a screen location and create a ray from the camera
    // location through the screen
    float3 ray_origin, ray_direction;
    computeRay( idx, dim, ray_origin, ray_direction );

    // Trace the ray against our scene hierarchy
    unsigned int p0, p1, p2;
    optixTrace(
            params.handle,
            ray_origin,
            ray_direction,
            0.0f,                // Min intersection distance
            1e16f,               // Max intersection distance
            0.0f,                // rayTime -- used for motion blur
            OptixVisibilityMask( 255 ), // Specify always visible
            OPTIX_RAY_FLAG_NONE,
            0,                   // SBT offset   -- See SBT discussion
            1,                   // SBT stride   -- See SBT discussion
            0,                   // missSBTIndex -- See SBT discussion
            p0, p1, p2 );
    float3 result;
    result.x = __uint_as_float( p0 );
    result.y = __uint_as_float( p1 );
    result.z = __uint_as_float( p2 );

    // Record results in our output raster
    params.image[idx.y * params.image_width + idx.x] = make_color( result );
}


extern "C" __global__ void __miss__ms()
{
    MissData* miss_data  = reinterpret_cast<MissData*>( optixGetSbtDataPointer() );
    
    
    setPayload(miss_data->bg_color);
}


extern "C" __global__ void __closesthit__ch()
{
    // When built-in triangle intersection is used, a number of fundamental
    // attributes are provided by the OptiX API, indlucing barycentric coordinates.
    const float2 barycentrics = optixGetTriangleBarycentrics();
    unsigned int index = optixGetPrimitiveIndex();
    float3 data[3];
    optixGetTriangleVertexData(params.handle, index, 0, 0, data);

    
    float3 world_position = optixGetWorldRayOrigin() + optixGetWorldRayDirection() * optixGetRayTmax();
    float result = 100.f;
    for (int i = 0; i < lights_num; i++)
    {
        float distance = sqrt((world_position.x - lights[i].x) * (world_position.x - lights[i].x) +
            (world_position.y - lights[i].y) * (world_position.y - lights[i].y) +
            (world_position.z - lights[i].z) * (world_position.z - lights[i].z));
        float3 direction = { lights[i].x - world_position.x,lights[i].y - world_position.y, lights[i].z - world_position.z };
        float db = 100.f;
        float tmin = 0.f;
        int n = 0;
        while ((tmin < distance)) {
            unsigned int p0, p1, p2;
            n++;
            optixTrace(
                params.handle,
                world_position,
                normalize(direction),
                tmin,                // Min intersection distance
                distance,            // Max intersection distance
                0.0f,                // rayTime -- used for motion blur
                OptixVisibilityMask(254), // Specify always visible
                OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
                1,                   // SBT offset   -- See SBT discussion
                1,                   // SBT stride   -- See SBT discussion
                1,                   // missSBTIndex -- See SBT discussion
                p0, p1, p2);
            db -= __uint_as_float(p0);
            tmin = __uint_as_float(p1);
        }
        if (db < result) result = db;
        result = n * 10;
    }
    float red = (result)/100.f;
    if (red < 0.) red = 0;
    setPayload(make_float3(red, 0, 0));


    /*if (barycentrics.y < 0.03  || barycentrics.x < 0.03 || barycentrics.x + barycentrics.y > 0.97)
    {
        setPayload(make_float3(0, 0, 0));
    }
    else
    {
        setPayload(make_float3(1. / (distance * distance), 0, 0));
    }*/
    //setPayload(make_float3(1, 0, 0));
}
extern "C" __global__ void __closesthit__ch_sh()
{
    float3 origin = optixGetWorldRayOrigin();
    float tmax = optixGetRayTmax();
    float3 world_position = origin + optixGetWorldRayDirection() * tmax;
    float distance = sqrt((world_position.x - origin.x) * (world_position.x - origin.x) +
        (world_position.y - origin.y) * (world_position.y - origin.y) +
        (world_position.z - origin.z) * (world_position.z - origin.z)) - optixGetRayTmin();
    setPayload(make_float3(distance * 100.f, tmax, 0));
}

extern "C" __global__ void __miss__ms_sh()
{
    float3 origin = optixGetWorldRayOrigin();
    float tmax = optixGetRayTmax();
    float3 world_position = origin + optixGetWorldRayDirection() * tmax;
    float distance = sqrt((world_position.x - origin.x) * (world_position.x - origin.x) +
        (world_position.y - origin.y) * (world_position.y - origin.y) +
        (world_position.z - origin.z) * (world_position.z - origin.z)) - optixGetRayTmin();
    setPayload(make_float3(distance * 100, tmax, 0));
}
