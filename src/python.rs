//! `__cuda_array_interface__` helpers for PyTorch/CuPy interop.

use cudarc::driver::sys::CUdeviceptr;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

#[derive(Debug, Clone)]
pub struct CudaArrayInfo {
    pub ptr: CUdeviceptr,
    pub shape: Vec<usize>,
    pub strides: Option<Vec<usize>>,
    pub dtype: String,
    pub read_only: bool,
}

pub fn cuda_array_interface<'py>(
    py: Python<'py>,
    ptr: CUdeviceptr,
    shape: &[usize],
    dtype: &str,
) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("data", (ptr as usize, false))?;
    dict.set_item("shape", shape.to_vec())?;
    dict.set_item("strides", compute_strides(shape, dtype_size(dtype)))?;
    dict.set_item("typestr", dtype)?;
    dict.set_item("version", 3)?;
    Ok(dict.into())
}

pub fn read_cuda_array_interface(obj: &Bound<'_, PyAny>) -> PyResult<CudaArrayInfo> {
    let interface = obj.getattr("__cuda_array_interface__")?;
    let dict = interface.downcast::<PyDict>()?;

    let version: i32 = dict
        .get_item("version")?
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("missing 'version'"))?
        .extract()?;
    if version < 2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("__cuda_array_interface__ version {} not supported (need >= 2)", version)
        ));
    }

    let data = dict
        .get_item("data")?
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("missing 'data'"))?;
    let data_tuple = data.downcast::<PyTuple>()?;
    let ptr: usize = data_tuple.get_item(0)?.extract()?;
    let read_only: bool = data_tuple.get_item(1)?.extract().unwrap_or(false);

    let shape: Vec<usize> = dict
        .get_item("shape")?
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("missing 'shape'"))?
        .extract()?;

    let strides: Option<Vec<usize>> = dict
        .get_item("strides")?
        .and_then(|s| if s.is_none() { None } else { Some(s) })
        .map(|s| s.extract())
        .transpose()?;

    let dtype: String = dict
        .get_item("typestr")?
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("missing 'typestr'"))?
        .extract()?;

    Ok(CudaArrayInfo {
        ptr: ptr as CUdeviceptr,
        shape,
        strides,
        dtype,
        read_only,
    })
}

pub trait CudaArrayInterface {
    fn cuda_ptr(&self) -> CUdeviceptr;
    fn cuda_shape(&self) -> Vec<usize>;
    fn cuda_dtype(&self) -> &str;

    fn to_cuda_array_interface<'py>(&self, py: Python<'py>) -> PyResult<Py<PyDict>> {
        cuda_array_interface(py, self.cuda_ptr(), &self.cuda_shape(), self.cuda_dtype())
    }
}

pub fn create_cuda_array_interface<'py>(
    py: Python<'py>,
    device_ptr: CUdeviceptr,
    shape: (usize, usize, usize),
    dtype: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let (height, width, channels) = shape;
    let cuda_array_dict = cuda_array_interface(py, device_ptr, &[height, width, channels], dtype)?;
    let locals = PyDict::new(py);
    locals.set_item("__cuda_array_interface__", cuda_array_dict)?;
    let cuda_array_cls = py.eval(
        pyo3::ffi::c_str!(
            "type('CudaArray', (), {\
                '__cuda_array_interface__': __cuda_array_interface__, \
                'shape': __cuda_array_interface__['shape'], \
                'device_ptr': __cuda_array_interface__['data'][0]\
            })"
        ),
        None,
        Some(&locals),
    )?;

    let cuda_array = cuda_array_cls.call0()?;
    Ok(cuda_array)
}

pub fn create_image_cuda_array<'py>(
    py: Python<'py>,
    device_ptr: CUdeviceptr,
    width: u32,
    height: u32,
    channels: u32,
) -> PyResult<Bound<'py, PyAny>> {
    create_cuda_array_interface(
        py,
        device_ptr,
        (height as usize, width as usize, channels as usize),
        "<u1",
    )
}

fn compute_strides(shape: &[usize], itemsize: usize) -> Vec<usize> {
    let mut strides = vec![0usize; shape.len()];
    if shape.is_empty() {
        return strides;
    }
    strides[shape.len() - 1] = itemsize;
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

pub fn dtype_size(dtype: &str) -> usize {
    match dtype {
        "<u1" | "|u1" | "uint8" => 1,
        "<u2" | "|u2" | "uint16" => 2,
        "<u4" | "|u4" | "uint32" => 4,
        "<u8" | "|u8" | "uint64" => 8,
        "<i1" | "|i1" | "int8" => 1,
        "<i2" | "|i2" | "int16" => 2,
        "<i4" | "|i4" | "int32" => 4,
        "<i8" | "|i8" | "int64" => 8,
        "<f2" | "float16" => 2,
        "<f4" | "float32" => 4,
        "<f8" | "float64" => 8,
        _ => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_size() {
        assert_eq!(dtype_size("<u1"), 1);
        assert_eq!(dtype_size("<f4"), 4);
        assert_eq!(dtype_size("<f8"), 8);
    }

    #[test]
    fn test_compute_strides() {
        assert_eq!(compute_strides(&[256, 256, 4], 1), vec![1024, 4, 1]);
        assert_eq!(compute_strides(&[10, 20, 30], 4), vec![2400, 120, 4]);
        assert_eq!(compute_strides(&[100], 4), vec![4]);
        assert!(compute_strides(&[], 4).is_empty());
    }
}
