继续对`initModule`函数中的功能进行解读。

### 6、ATen的初始化

``` c++
#if defined(USE_CUDNN) || defined(__HIP_PLATFORM_HCC__)
  PyObject *has_cudnn = Py_True;
#else
  PyObject *has_cudnn = Py_False;
#endif
 ASSERT_TRUE(set_module_attr("has_cudnn", has_cudnn));

  // force ATen to initialize because it handles
  // setting up TH Errors so that they throw C++ exceptions
  at::init();

  // Automatically translate errors thrown from pybind11 functions
  py::register_exception_translator([](std::exception_ptr e) { // NOLINT
    try {
      if (e) {
        std::rethrow_exception(e);
      }
    }
    CATCH_TH_ERRORS()
  });

  auto py_module = py::reinterpret_borrow<py::module>(module);
  py_module.def("_demangle", &c10::demangle);
  py_module.def("_log_api_usage_once", &LogAPIUsageOnceFromPython);

  ASSERT_TRUE(set_module_attr("has_openmp", at::hasOpenMP() ? Py_True : Py_False));
  ASSERT_TRUE(set_module_attr("has_mkl", at::hasMKL() ? Py_True : Py_False));
  ASSERT_TRUE(set_module_attr("has_lapack", at::hasLAPACK() ? Py_True : Py_False));

#ifdef USE_CUDA
  PyObject *has_cuda = Py_True;
#else
  PyObject *has_cuda = Py_False;
#endif
  ASSERT_TRUE(set_module_attr("has_cuda", has_cuda));

  ASSERT_TRUE(set_module_attr("has_mkldnn", at::hasMKLDNN() ? Py_True : Py_False));

#ifdef _GLIBCXX_USE_CXX11_ABI
  ASSERT_TRUE(set_module_attr("_GLIBCXX_USE_CXX11_ABI", _GLIBCXX_USE_CXX11_ABI ? Py_True : Py_False));
#else
  ASSERT_TRUE(set_module_attr("_GLIBCXX_USE_CXX11_ABI", Py_False));
#endif

  auto defaultGenerator = at::detail::getDefaultCPUGenerator();
  THPDefaultCPUGenerator = (THPGenerator*)THPGenerator_initDefaultGenerator(defaultGenerator);
  // This reference is meant to be given away, so no need to incref here.
  ASSERT_TRUE(set_module_attr("default_generator", (PyObject*)THPDefaultCPUGenerator, /* incref= */ false));
```

这一段代码会进行`ATen`的初始化，然后使用`at::detail::getDefaultCPUGenerator()`进行generator的初始化。

另外，PyTorch会根据编译环境和用户配置，然后向`torch._C`上注册一些flag。这些flag有has_cudnn、has_mkl、has_lapack、_GLIBCXX_USE_CXX11_ABI。

### 7、`torch._C._THNN`和`torch._C._THCUNN`的初始化

PyTorch1.0在这一小节里注册了`torch._C._THNN`和`torch._C._THCUNN`类，在1.4版本的pytorch中没有看到这部分语句：

```text
PyObject* initModule() {
  ......
  torch::nn::init__THNN(module);
  torch::nn::init__THCUNN(module);
  ......
}
```

这两个类都拥有数量巨大的op函数，一个是CPU版的，一个是CUDA版的



### `initModule`之后

在`initModule()`函数初始化完毕之后，`import torch`的初始化工作还没有结束。因为在这之后，python的初始化脚本还要调用以下2个API才算真正完成全部的初始化：

```python3
_C._initExtension(manager_path())
_C._init_names(list(torch._storage_classes))
```

其中主要的工作都是在`_C._initExtension`中，这个初始化做了以下的工作：

```text
torch::utils::initializeLayouts();
torch::utils::initializeDtypes();
torch::tensors::initialize_python_bindings();

THPDoubleStorage_postInit(module);
THPFloatStorage_postInit(module);
THPHalfStorage_postInit(module);
THPLongStorage_postInit(module);
THPIntStorage_postInit(module);
THPShortStorage_postInit(module);
THPCharStorage_postInit(module);
THPByteStorage_postInit(module);
THPBoolStorage_postInit(module);
//定义在THPStorage_(postInit)函数中，因为THPStorage_会被宏替换THPDoubleStorage_ \
//THPFloatStorage_、THPHalfStorage_、THPLongStorage_......

THPAutograd_initFunctions();
```

最后的`THPAutograd_initFunctions()`则是初始化了torch的自动微分系统，这是PyTorch动态图框架的基础。



