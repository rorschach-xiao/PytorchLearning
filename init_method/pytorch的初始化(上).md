## 当我们使用 import torch时，我们都做了什么？

当我们在使用pytorch时都会用到下面这句话：

```python
import torch
```

按照python的规范，运行时会找到torch文件夹下的```  __init__.py``` 文件，在这个文件中我们可以看到这么一句话

```python
from torch._C import *
```

其中其中```torch._C```就是``` _C.cpython-37m-x86_64-linux-gnu.so``` 。按照python规范，默认引擎是CPython，而CPython的C/C++扩展是一个共享库，这个共享库安装在PYTHONPATH目录下，并且文件名要和module的名字一样,这个共享库要实现`PyInit_modulename`符号来作为import时候的逻辑入口。

对于PyTorch来说这个modulename 是_C，因此我们可以揣测，在`torch/csrc/stub.cpp`中一定实现了PyInit_C这个函数。事实也是这样的：

```c++
extern PyObject* initModule();

#if PY_MAJOR_VERSION == 2
PyMODINIT_FUNC init_C()
{
  initModule();
}
#else
PyMODINIT_FUNC PyInit__C()
{
  return initModule();
}
#endif
```

pytorch所有的初始化工作都是在`initModule`这个函数里完成的，这个函数的位置在`torch/csrc/module.cpp`中。

### 1、`torch._C`的诞生：

在`initModule()`这个函数中产生了`torch._C`这个类，并注册了多个类函数。

``` c++
PyObject* initModule() { 
 THPUtils_addPyMethodDefs(methods, TorchMethods);
 THPUtils_addPyMethodDefs(methods, DataLoaderMethods);
 THPUtils_addPyMethodDefs(methods, torch::autograd::python_functions());
 THPUtils_addPyMethodDefs(methods, torch::multiprocessing::python_functions());
#ifdef USE_CUDA
  THPUtils_addPyMethodDefs(methods, THCPModule_methods());
#endif
#ifdef USE_DISTRIBUTED
#ifdef USE_C10D
  THPUtils_addPyMethodDefs(methods, torch::distributed::c10d::python_functions());
  THPUtils_addPyMethodDefs(methods, torch::distributed::rpc::python_functions());
  THPUtils_addPyMethodDefs(
      methods, torch::distributed::autograd::python_functions());
  THPUtils_addPyMethodDefs(methods, torch::distributed::rpc::testing::python_functions());
#endif
#endif

#if PY_MAJOR_VERSION == 2
  ASSERT_TRUE(module = Py_InitModule("torch._C", methods.data()));
#else
  static struct PyModuleDef torchmodule = {
     PyModuleDef_HEAD_INIT,
     "torch._C",
     nullptr,
     -1,
     methods.data()
  };
  ASSERT_TRUE(module = PyModule_Create(&torchmodule));
#endif
  ......
}
```

其中`TorchMethods`注册了29个方法，`DataLoaderMethods`注册了4个方法，`torch::autograd::python_functions`注册了4个方法，`torch::multiprocessing::python_functions`注册了一个方法，`THCPModule_methods()`注册了37个CUDA相关的函数等等。

总之，`initModule()`开始创造了`torch._C`符号并且向`torch._C`注册了一百余个函数，涉及torch、dataloader、autograd、multiprocess、cuda、cudnn、distribute、c10d方面。

### 2、一些关键类型

以下代码先后初始化了`torch._C.PtrWrapper`、`torch._C.Generator`（含5个方法）、 `FatalError`、 `torch.Size`、`torch.dtype`、`torch.info`、`torch.layout`、`torch.memory_format`、`torch.device`.

``` c++
ASSERT_TRUE(THPWrapper_init(module));
ASSERT_TRUE(THPGenerator_init(module));
ASSERT_TRUE(THPException_init(module));
THPSize_init(module);
THPDtype_init(module);
THPDTypeInfo_init(module);
THPLayout_init(module);
THPMemoryFormat_init(module);
THPQScheme_init(module);
THPDevice_init(module);
```

其中`torch.layout` 是`torch.tensor`自带的属性之一，表示张量的内存布局。现支持`torch.strided` (dense Tensors)和`torch.sparse_coo` (sparse COO Tensors)两种类型，其中前者比较常用。`torch.layout`=`torch.strided`时，它会得到一个n维向量，n即为`torch.tensor`的维度，第k维表示 `torch.tensor`在第k维单个元素所占的内存。例子如下：

```python
>>> x = torch.Tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
>>> x.stride()
(5, 1)
```



### 3、``torch._C._TensorBase``的诞生

```c++
PyObject* initModule() {
  ......
  THPVariable_initModule(module);
  THPFunction_initModule(module);
  THPEngine_initModule(module);
  ......
}
```

`THPVariable_initModule` 定义在`torch/csrc/autograd/python_variable.cpp`中，它创建了`torch._C._TensorBase`，这个所有tensor的基类。

``` c++
bool THPVariable_initModule(PyObject *module)
{
  static std::vector<PyMethodDef> methods;
  THPUtils_addPyMethodDefs(methods, torch::autograd::variable_methods);
  THPUtils_addPyMethodDefs(methods, extra_methods);
  THPVariableType.tp_methods = methods.data();
  if (PyType_Ready(&THPVariableType) < 0)
    return false;
  Py_INCREF(&THPVariableType);
  PyModule_AddObject(module, "_TensorBase",   (PyObject *)&THPVariableType);
  torch::autograd::initTorchFunctions(module);
  torch::autograd::initTensorImplConversion(module);
  return true;
}
```

`THPFunction_initModule`定义在 `torch/csrc/autograd/python_function.cpp`中，创建了`torch._C._FunctionBase` ，

```c++
bool THPFunction_initModule(PyObject *module)
{
  if (PyType_Ready(&THPFunctionType) < 0)
    return false;
  Py_INCREF(&THPFunctionType);
  PyModule_AddObject(module, "_FunctionBase", (PyObject *)&THPFunctionType);
  return true;
}
```

以下两个类继承了`torch._C._FunctionBase`

```python
class Function(with_metaclass(FunctionMeta, _C._FunctionBase, _ContextMethodMixin, _HookMixin))
class BackwardCFunction(_C._FunctionBase, _ContextMethodMixin, _HookMixin)
```

`THPEngine_initModule`定义在`torch/csrc/autograd/python_engine.cpp`中，创建了`torch._C._EngineBase ` , ``_EngineBase``这个类负责动态图执行之前的`preprocess`，`_EngineBase`会将`torch.autograd`的`backward`之类的请求预处理后送给真正的`Engine`去执行。

```c++
bool THPEngine_initModule(PyObject *module)
{
  if (PyType_Ready(&THPEngineType) < 0)
    return false;
  Py_INCREF(&THPEngineType);
  PyModule_AddObject(module, "_ImperativeEngine", (PyObject *)&THPEngineType);
  set_default_engine_stub(get_python_engine);
  return true;
}
```



### 4、pybind11绑定

接下来是pybind11的绑定，该库用于将C++的类型暴露给python，使得C++的代码更好的绑定到python。

```c++
torch::onnx::initONNXBindings(module);
torch::jit::initJITBindings(module);
torch::impl::dispatch::initDispatchBindings(module);
torch::throughput_benchmark::initThroughputBenchmarkBindings(module);
torch::autograd::initNNFunctions(module);
torch::autograd::init_legacy_variable(module);
torch::python::init_bindings(module);
#ifdef USE_CUDA
torch::cuda::initModule(module);
#endif
```

`initONNXBindings`是ONNX的python binding：`torch._C._onnx.TensorProtoDataType`和`torch._C._onnx.OperatorExportTypes`：

```python
>>> dir(torch._C._onnx.TensorProtoDataType)
['BOOL', 'COMPLEX128', 'COMPLEX64', 'DOUBLE', 'FLOAT', 'FLOAT16', 'INT16', 'INT32', 'INT64', 'INT8', 'STRING', 'UINT16', 'UINT32', 'UINT64', 'UINT8', 'UNDEFINED', '__class__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getstate__', '__gt__', '__hash__', '__init__', '__int__', '__le__', '__lt__', '__members__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__setstate__', '__sizeof__', '__str__', '__subclasshook__', 'name']
>>> dir(torch._C._onnx.OperatorExportTypes)
['ONNX', 'ONNX_ATEN', 'ONNX_ATEN_FALLBACK', 'RAW', '__class__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getstate__', '__gt__', '__hash__', '__init__', '__int__', '__le__', '__lt__', '__members__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__setstate__', '__sizeof__', '__str__', '__subclasshook__', 'name']
```

`initJITBindings`则是通过pybind11往`torch._C`上注册了一堆和JIT相关的C++函数/对象；

`initNNFunctions`初始化了一个`torch._C._nn `对象，并注册了一些nn相关的函数;

`init_legacy_variable`注册了`torch._C._LegacyVariableBase`,`_LegacyVariableBase`类会派生出`Variable`类（该类的`_execution_engine`会初始化为`torch._C._EngineBase`）：

```python
class Variable(with_metaclass(VariableMeta, torch._C._LegacyVariableBase))
```

`init_bindings`和`torch::cuda::initModule`类似，通过pybind11往torch._C上注册了一些cuda相关的函数。

### 5、数据存储类型初始化

这一部分即往torch._C上注册`StorageBase`类。

```c++
  ASSERT_TRUE(THPDoubleStorage_init(module));
  ASSERT_TRUE(THPFloatStorage_init(module));
  ASSERT_TRUE(THPHalfStorage_init(module));
  ASSERT_TRUE(THPLongStorage_init(module));
  ASSERT_TRUE(THPIntStorage_init(module));
  ASSERT_TRUE(THPShortStorage_init(module));
  ASSERT_TRUE(THPCharStorage_init(module));
  ASSERT_TRUE(THPByteStorage_init(module));
  ASSERT_TRUE(THPBoolStorage_init(module));
  ASSERT_TRUE(THPQUInt8Storage_init(module));
  ASSERT_TRUE(THPQInt8Storage_init(module));
  ASSERT_TRUE(THPQInt32Storage_init(module));
  ASSERT_TRUE(THPBFloat16Storage_init(module));

#ifdef USE_CUDA
  // This will only initialise base classes and attach them to library namespace
  // They won't be ready for real usage until importing cuda module, that will
  // complete the process (but it defines Python classes before calling back into
  // C, so these lines have to execute first)..
  ASSERT_TRUE(THCPDoubleStorage_init(module));
  ASSERT_TRUE(THCPFloatStorage_init(module));
  ASSERT_TRUE(THCPHalfStorage_init(module));
  ASSERT_TRUE(THCPLongStorage_init(module));
  ASSERT_TRUE(THCPIntStorage_init(module));
  ASSERT_TRUE(THCPShortStorage_init(module));
  ASSERT_TRUE(THCPCharStorage_init(module));
  ASSERT_TRUE(THCPByteStorage_init(module));
  ASSERT_TRUE(THCPBoolStorage_init(module));
  ASSERT_TRUE(THCPBFloat16Storage_init(module));
......
#endif
```

这些方法在`torch._C`中注册了下面这些类：

``` python
CudaByteStorageBase
CudaCharStorageBase
CudaDoubleStorageBase
CudaFloatStorageBase
CudaHalfStorageBase
CudaIntStorageBase
CudaLongStorageBase
CudaShortStorageBase

ByteStorageBase
CharStorageBase
DoubleStorageBase
FloatStorageBase
HalfStorageBase
IntStorageBase
LongStorageBase
ShortStorageBase
```









