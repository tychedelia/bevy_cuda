mod external_memory;

#[cfg(windows)]
mod windows;

#[cfg(unix)]
mod linux;

#[cfg(windows)]
pub(crate) use windows::VkCudaBuffer;

#[cfg(unix)]
pub(crate) use linux::VkCudaBuffer;
