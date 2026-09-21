"""Isolate the driver's allocation/free and CPU zeroing at the traced size."""
import ctypes, json, time
from pathlib import Path
cu=ctypes.CDLL('libcuda.so.1'); libc=ctypes.CDLL(None)
U=ctypes.c_uint; I=ctypes.c_int; P=ctypes.c_void_p; Z=ctypes.c_size_t
for name,args in [('cuInit',[U]),('cuDeviceGet',[ctypes.POINTER(I),I]),('cuCtxCreate_v2',[ctypes.POINTER(P),U,I]),('cuMemHostAlloc',[ctypes.POINTER(P),Z,U]),('cuMemFreeHost',[P]),('cuCtxDestroy_v2',[P])]:
 f=getattr(cu,name);f.argtypes=args;f.restype=I
libc.memset.argtypes=[P,I,Z];libc.memset.restype=P
def call(name,*args):
 rc=getattr(cu,name)(*args)
 if rc:raise RuntimeError((name,rc))
call('cuInit',0);dev=I();call('cuDeviceGet',ctypes.byref(dev),0)
ctx=P();call('cuCtxCreate_v2',ctypes.byref(ctx),0,dev)
n=32658510964; ptr=P(); result={'bytes':n, 'note':'Independent one-allocation microbenchmark; not included in Llama search totals'}
t=time.perf_counter();call('cuMemHostAlloc',ctypes.byref(ptr),n,0);result['cuMemHostAlloc_seconds']=time.perf_counter()-t
print(json.dumps(result),flush=True)
t=time.perf_counter();libc.memset(ptr,0,n);result['memset_seconds']=time.perf_counter()-t
print(json.dumps(result),flush=True)
t=time.perf_counter();call('cuMemFreeHost',ptr);result['cuMemFreeHost_seconds']=time.perf_counter()-t
call('cuCtxDestroy_v2',ctx)
Path(__file__).with_name('pinned-probe.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
