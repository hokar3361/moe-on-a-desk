# llama-model.cpp: opt-in LLAMA_GPU_OWN_WEIGHTS=1 -> GPU weights go to buffers the backend allocates (copied from the mmap),
# CPU tensors (the n-gram table) stay mmap-wrapped
import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
old = "        if ((ml.use_mmap || is_lazy_mapped) && use_mmap_buffer && buffer_from_host_ptr_supported && is_default_buft) {"
assert s.count(old) == 1
s = s.replace(old, """        // opt-in: weights of non-CPU devices are copied into backend-owned buffers instead of wrapping the mmap
        const bool own_weights = getenv("LLAMA_GPU_OWN_WEIGHTS") != nullptr && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU && !is_lazy_mapped;

        if ((ml.use_mmap || is_lazy_mapped) && use_mmap_buffer && buffer_from_host_ptr_supported && is_default_buft && !own_weights) {""")
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
