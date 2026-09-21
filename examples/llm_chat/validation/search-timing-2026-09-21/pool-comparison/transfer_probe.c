#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
static int ok(CUresult r,const char *where){if(r==CUDA_SUCCESS)return 1;const char *n="unknown";cuGetErrorName(r,&n);printf("%s: %s\n",where,n);return 0;}
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(void){
 setbuf(stdout,NULL);CUdevice dev;CUcontext ctx;CUstream stream;CUdeviceptr gpu;CUevent a,b;const size_t n=32658510964ULL;
 if(!ok(cuInit(0),"init")||!ok(cuDeviceGet(&dev,0),"device")||!ok(cuDevicePrimaryCtxRetain(&ctx,dev),"ctx")||!ok(cuCtxSetCurrent(ctx),"setctx")||!ok(cuStreamCreate(&stream,CU_STREAM_NON_BLOCKING),"stream")||!ok(cuMemAlloc(&gpu,n),"gpu_alloc")||!ok(cuEventCreate(&a,0),"event")||!ok(cuEventCreate(&b,0),"event"))return 1;
 const char *names[]={"legacy_host_alloc","host_pool_cpu_access","host_pool_gpu_access"};
 for(int mode=0;mode<3;mode++){
  void *host=NULL;CUmemoryPool pool=NULL;CUdeviceptr raw=0;printf("mode=%s bytes=%zu\n",names[mode],n);double start=now();
  if(mode==0){if(!ok(cuMemHostAlloc(&host,n,0),"host_alloc"))continue;}
  else{CUmemPoolProps p={0};p.allocType=CU_MEM_ALLOCATION_TYPE_PINNED;p.location.type=CU_MEM_LOCATION_TYPE_HOST_NUMA;p.location.id=0;
   if(!ok(cuMemPoolCreate(&pool,&p),"pool_create"))continue;
   if(mode==2){CUmemAccessDesc access={0};access.location.type=CU_MEM_LOCATION_TYPE_DEVICE;access.location.id=0;access.flags=CU_MEM_ACCESS_FLAGS_PROT_READWRITE;if(!ok(cuMemPoolSetAccess(pool,&access,1),"gpu_access")){cuMemPoolDestroy(pool);continue;}}
   if(!ok(cuMemAllocFromPoolAsync(&raw,n,pool,stream),"pool_alloc")||!ok(cuStreamSynchronize(stream),"sync")){cuMemPoolDestroy(pool);continue;}
   host=(void*)(uintptr_t)raw;
  }
  printf("alloc_seconds=%.6f\n",now()-start);start=now();memset(host,0x5a,n);printf("fill_seconds=%.6f\n",now()-start);
  for(int trial=0;trial<3;trial++){
   ok(cuEventRecord(a,stream),"start_event");start=now();int copied=ok(cuMemcpyHtoDAsync(gpu,host,n,stream),"H2D");double api=now()-start;ok(cuEventRecord(b,stream),"end_event");ok(cuStreamSynchronize(stream),"copy_sync");double wall=now()-start;float ms=0;ok(cuEventElapsedTime(&ms,a,b),"elapsed");printf("trial=%d api_seconds=%.6f wall_seconds=%.6f event_ms=%.3f success=%d\n",trial,api,wall,ms,copied);
  }
  unsigned char first=0,last=0;ok(cuMemcpyDtoH(&first,gpu,1),"read_first");ok(cuMemcpyDtoH(&last,gpu+n-1,1),"read_last");printf("first=%u last=%u\n",first,last);
  start=now();if(mode==0)ok(cuMemFreeHost(host),"free_host");else{ok(cuMemFreeAsync(raw,stream),"free_pool");ok(cuStreamSynchronize(stream),"free_sync");ok(cuMemPoolDestroy(pool),"pool_destroy");}printf("release_seconds=%.6f\n",now()-start);
 }
 cuEventDestroy(a);cuEventDestroy(b);cuMemFree(gpu);cuStreamDestroy(stream);cuDevicePrimaryCtxRelease(dev);return 0;
}
