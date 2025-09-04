//! Memory layout optimizations and utilities.
//!
//! This module provides utilities for memory-efficient data structures
//! and operations that improve cache locality and reduce allocations.

use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;

/// A vector type with guaranteed memory alignment for optimal performance.
///
/// This is particularly useful for SIMD operations and cache-friendly access patterns.
pub struct AlignedVec<T> {
    ptr: NonNull<T>,
    len: usize,
    capacity: usize,
    alignment: usize,
}

impl<T> AlignedVec<T> {
    /// Creates a new aligned vector with the specified alignment.
    ///
    /// # Arguments
    /// * `alignment` - Memory alignment in bytes (must be power of 2)
    ///
    /// # Panics
    /// Panics if alignment is not a power of 2.
    pub fn new(alignment: usize) -> Self {
        assert!(alignment.is_power_of_two(), "Alignment must be power of 2");

        Self {
            ptr: NonNull::dangling(),
            len: 0,
            capacity: 0,
            alignment,
        }
    }

    /// Creates a new aligned vector with the default cache line alignment.
    pub fn new_cache_aligned() -> Self {
        Self::new(super::MEMORY_ALIGNMENT)
    }

    /// Reserves capacity for at least `additional` more elements.
    pub fn reserve(&mut self, additional: usize) {
        let new_capacity = self.len + additional;
        if new_capacity <= self.capacity {
            return;
        }

        let new_capacity = new_capacity.next_power_of_two();
        self.reallocate(new_capacity);
    }

    /// Pushes an element to the end of the vector.
    pub fn push(&mut self, value: T) {
        if self.len == self.capacity {
            let new_capacity = if self.capacity == 0 {
                4
            } else {
                self.capacity * 2
            };
            self.reallocate(new_capacity);
        }

        unsafe {
            self.ptr.as_ptr().add(self.len).write(value);
        }
        self.len += 1;
    }

    /// Returns a slice of the vector's contents.
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Returns a mutable slice of the vector's contents.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Returns the length of the vector.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns whether the vector is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn reallocate(&mut self, new_capacity: usize) {
        let new_layout =
            Layout::from_size_align(new_capacity * std::mem::size_of::<T>(), self.alignment)
                .expect("Invalid layout");

        let new_ptr = unsafe { alloc(new_layout) as *mut T };
        if new_ptr.is_null() {
            panic!("Memory allocation failed");
        }

        if self.capacity > 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(self.ptr.as_ptr(), new_ptr, self.len);
                let old_layout = Layout::from_size_align(
                    self.capacity * std::mem::size_of::<T>(),
                    self.alignment,
                )
                .expect("Invalid layout");
                dealloc(self.ptr.as_ptr() as *mut u8, old_layout);
            }
        }

        self.ptr = NonNull::new(new_ptr).expect("Non-null pointer");
        self.capacity = new_capacity;
    }
}

impl<T> Drop for AlignedVec<T> {
    fn drop(&mut self) {
        if self.capacity > 0 {
            // Drop all elements first
            for i in 0..self.len {
                unsafe {
                    self.ptr.as_ptr().add(i).drop_in_place();
                }
            }

            // Deallocate memory
            unsafe {
                let layout = Layout::from_size_align(
                    self.capacity * std::mem::size_of::<T>(),
                    self.alignment,
                )
                .expect("Invalid layout");
                dealloc(self.ptr.as_ptr() as *mut u8, layout);
            }
        }
    }
}

unsafe impl<T: Send> Send for AlignedVec<T> {}
unsafe impl<T: Sync> Sync for AlignedVec<T> {}

/// Aligns a slice to the specified boundary for optimal access patterns.
///
/// This function can be used to ensure data is properly aligned for
/// vectorized operations.
pub fn align_slice<T>(slice: &[T], alignment: usize) -> &[T] {
    let ptr = slice.as_ptr() as usize;
    let aligned_ptr = (ptr + alignment - 1) & !(alignment - 1);
    let offset = (aligned_ptr - ptr) / std::mem::size_of::<T>();

    if offset >= slice.len() {
        &[]
    } else {
        &slice[offset..]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aligned_vec_basic_operations() {
        let mut vec = AlignedVec::<i32>::new_cache_aligned();
        assert!(vec.is_empty());

        vec.push(1);
        vec.push(2);
        vec.push(3);

        assert_eq!(vec.len(), 3);
        assert_eq!(vec.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn test_aligned_vec_reserve() {
        let mut vec = AlignedVec::<i32>::new_cache_aligned();
        vec.reserve(100);

        for i in 0..50 {
            vec.push(i);
        }

        assert_eq!(vec.len(), 50);
    }

    #[test]
    fn test_align_slice() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let aligned = align_slice(&data, 8);
        assert!(!aligned.is_empty());
    }
}
