use cudarc::driver::CudaStream as CudarcStream;
use std::ops::Deref;
use std::sync::Arc;

pub struct CudaStream {
    inner: Arc<CudarcStream>,
}

impl CudaStream {
    pub(crate) fn new(inner: Arc<CudarcStream>) -> Self {
        Self { inner }
    }

    pub fn synchronize(&self) -> crate::error::Result<()> {
        self.inner.synchronize()?;
        Ok(())
    }

    pub fn as_cudarc(&self) -> &Arc<CudarcStream> {
        &self.inner
    }
}

impl Deref for CudaStream {
    type Target = CudarcStream;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
