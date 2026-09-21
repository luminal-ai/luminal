#include <cuda.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
static int check(CUresult r, const char *where) {
    if (r == CUDA_SUCCESS) return 1;
    const char *name = "unknown"; cuGetErrorName(r, &name);
    printf("%s: %s (%d)\n", where, name, r); return 0;
}
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
int main(void) {
    CUdevice dev; CUcontext ctx; CUstream stream; int supported=0,version=0;
    if (!check(cuInit(0),"init") || !check(cuDeviceGet(&dev,0),"device") || !check(cuDevicePrimaryCtxRetain(&ctx,dev),"context") || !check(cuCtxSetCurrent(ctx),"current") || !check(cuStreamCreate(&stream,CU_STREAM_NON_BLOCKING),"stream")) return 1;
    cuDriverGetVersion(&version); cuDeviceGetAttribute(&supported,CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED,dev);
    printf("driver_api_version=%d device_memory_pools_supported=%d\n",version,supported);
    CUmemoryPool default_pool; cuuint64_t threshold=0;
    if (check(cuDeviceGetDefaultMemPool(&default_pool,dev),"default_pool") && check(cuMemPoolGetAttribute(default_pool,CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,&threshold),"threshold")) printf("default_device_pool_release_threshold=%llu\n",(unsigned long long)threshold);
    const CUmemLocationType locations[]={CU_MEM_LOCATION_TYPE_DEVICE,CU_MEM_LOCATION_TYPE_HOST_NUMA,CU_MEM_LOCATION_TYPE_HOST};
    const char *names[]={"DEVICE","HOST_NUMA","HOST"};
    for(int i=1;i<2;i++) {
        CUmemPoolProps props={0}; props.allocType=CU_MEM_ALLOCATION_TYPE_PINNED; props.location.type=locations[i]; props.location.id=0;
        CUmemoryPool pool=NULL; printf("pool=%s ",names[i]);
        if (!check(cuMemPoolCreate(&pool,&props),"create")) continue;
        printf("create=success\n");
        const size_t bytes=32658510964ULL; cuuint64_t keep=64ULL*1024*1024*1024; if(!check(cuMemPoolSetAttribute(pool,CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,&keep),"set_threshold")) {cuMemPoolDestroy(pool);continue;}
        for(int trial=0;trial<2;trial++) {
            CUdeviceptr ptr=0; double t=now();
            if(!check(cuMemAllocFromPoolAsync(&ptr,bytes,pool,stream),"allocate")) break;
            if(!check(cuStreamSynchronize(stream),"allocation_sync")) break;
            double alloc_ms=(now()-t)*1e3;
            unsigned int type=0; cuPointerGetAttribute(&type,CU_POINTER_ATTRIBUTE_MEMORY_TYPE,ptr);
            if (i>0) {
                double zero_start=now(); memset((void*)(uintptr_t)ptr,0x5a,bytes); printf("  fill_ms=%.3f\n",(now()-zero_start)*1e3);
                CUdeviceptr gpu=0; unsigned char got=0;
                if(check(cuMemAlloc(&gpu,1024*1024),"gpu_alloc") && check(cuMemcpyHtoD(gpu,(void*)(uintptr_t)ptr,1024*1024),"H2D") && check(cuMemcpyDtoH(&got,gpu,1),"D2H")) printf("  host_write_and_transfer=%s\n",got==0x5a?"pass":"FAIL");
                if(gpu)cuMemFree(gpu);
            }
            t=now();check(cuMemFreeAsync(ptr,stream),"free");check(cuStreamSynchronize(stream),"free_sync");double free_ms=(now()-t)*1e3;
            cuuint64_t reserved=0;cuMemPoolGetAttribute(pool,CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,&reserved);
            printf("  trial=%d bytes=32658510964 memory_type=%u alloc_sync_ms=%.3f free_sync_ms=%.3f retained=%llu\n",trial,type,alloc_ms,free_ms,(unsigned long long)reserved);
        }
        double destroy_start=now(); check(cuMemPoolDestroy(pool),"destroy"); printf("pool_destroy_ms=%.3f\n",(now()-destroy_start)*1e3);
    }
    check(cuStreamDestroy(stream),"stream_destroy");check(cuDevicePrimaryCtxRelease(dev),"context_release");return 0;
}
