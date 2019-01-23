Occasionally, I ran into the error:
```
2019-01-22 23:11:00.177003: F tensorflow/core/common_runtime/bfc_allocator.cc:458] Check failed: c->in_use() && (c->bin_num == kInvalidBinNum) 
Aborted (core dumped)

2019-01-23 14:53:34.210690: F tensorflow/core/common_runtime/bfc_allocator.cc:380] Check failed: h != kInvalidChunkHandle 
Aborted (core dumped)

```
It was reported on Dec 12, 2018. It is deemed to be a bug with Tensorflow: https://github.com/tensorflow/tensorflow/issues/22750. One suggestion is to try TF nightly, pending the next TF release.

