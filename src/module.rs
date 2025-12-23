use crate::error::Result;
use cudarc::driver::{CudaFunction as CudarcFunction, CudaModule as CudarcModule};
use std::ops::Deref;
use std::sync::Arc;

pub struct CudaModule {
    inner: Arc<CudarcModule>,
}

impl CudaModule {
    pub(crate) fn new(inner: Arc<CudarcModule>) -> Self {
        Self { inner }
    }

    pub fn get_function(&self, name: &str) -> Result<CudaFunction> {
        let func = self.inner.load_function(name)?;
        Ok(CudaFunction { inner: func })
    }

    pub fn as_cudarc(&self) -> &Arc<CudarcModule> {
        &self.inner
    }
}

impl Deref for CudaModule {
    type Target = CudarcModule;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

pub struct CudaFunction {
    inner: CudarcFunction,
}

impl CudaFunction {
    pub fn as_cudarc(&self) -> &CudarcFunction {
        &self.inner
    }
}

impl Deref for CudaFunction {
    type Target = CudarcFunction;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
