//! # Geometry Module
//!
//! Provides geometric primitives and operations for 3D spatial calculations.
//! This module contains the core geometric types used throughout the interpolation
//! algorithms, with a focus on performance and numerical stability.

use std::ops::{Index, IndexMut};

/// A 3D point with generic coordinate type support
///
/// This structure represents a point in 3D space with coordinates stored
/// in a fixed-size array for optimal memory layout and cache performance.
/// The generic type parameter allows for both floating-point and integer
/// coordinate systems.
///
/// # Type Parameters
///
/// * `T` - The numeric type for coordinates (typically `f64` or `f32`)
///
/// # Examples
///
/// ```
/// use naturalneighbor3d::core::geometry::Point3D;
///
/// let point_f64 = Point3D::new(1.0, 2.0, 3.0);
/// let point_i32 = Point3D::new(1i32, 2i32, 3i32);
/// let origin = Point3D::<f64>::origin();
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point3D<T = f64> {
    /// Coordinate array [x, y, z] for efficient memory access
    pub coords: [T; 3],
}

impl<T> Point3D<T>
where
    T: Copy + Default,
{
    /// Creates a new 3D point with specified coordinates
    ///
    /// # Arguments
    ///
    /// * `x` - X-coordinate
    /// * `y` - Y-coordinate  
    /// * `z` - Z-coordinate
    ///
    /// # Returns
    ///
    /// A new `Point3D` instance with the given coordinates
    ///
    /// # Examples
    ///
    /// ```
    /// # use naturalneighbor3d::core::geometry::Point3D;
    /// let point = Point3D::new(1.0, 2.0, 3.0);
    /// assert_eq!(point.x(), 1.0);
    /// assert_eq!(point.y(), 2.0);
    /// assert_eq!(point.z(), 3.0);
    /// ```
    #[inline]
    pub fn new(x: T, y: T, z: T) -> Self {
        Self { coords: [x, y, z] }
    }

    /// Creates a point at the origin (0, 0, 0)
    ///
    /// # Returns
    ///
    /// A new `Point3D` instance at the origin
    ///
    /// # Examples
    ///
    /// ```
    /// # use naturalneighbor3d::core::geometry::Point3D;
    /// let origin = Point3D::<f64>::origin();
    /// assert_eq!(origin.x(), 0.0);
    /// assert_eq!(origin.y(), 0.0);
    /// assert_eq!(origin.z(), 0.0);
    /// ```
    #[inline]
    pub fn origin() -> Self
    where
        T: Default,
    {
        Self {
            coords: [T::default(), T::default(), T::default()],
        }
    }

    /// Returns the X-coordinate
    ///
    /// # Examples
    ///
    /// ```
    /// # use naturalneighbor3d::core::geometry::Point3D;
    /// let point = Point3D::new(1.0, 2.0, 3.0);
    /// assert_eq!(point.x(), 1.0);
    /// ```
    #[inline]
    pub fn x(&self) -> T {
        self.coords[0]
    }

    /// Returns the Y-coordinate
    ///
    /// # Examples
    ///
    /// ```
    /// # use naturalneighbor3d::core::geometry::Point3D;
    /// let point = Point3D::new(1.0, 2.0, 3.0);
    /// assert_eq!(point.y(), 2.0);
    /// ```
    #[inline]
    pub fn y(&self) -> T {
        self.coords[1]
    }

    /// Returns the Z-coordinate
    ///
    /// # Examples
    ///
    /// ```
    /// # use naturalneighbor3d::core::geometry::Point3D;
    /// let point = Point3D::new(1.0, 2.0, 3.0);
    /// assert_eq!(point.z(), 3.0);
    /// ```
    #[inline]
    pub fn z(&self) -> T {
        self.coords[2]
    }

    /// Returns a mutable reference to the X-coordinate
    ///
    /// # Examples
    ///
    /// ```
    /// # use naturalneighbor3d::core::geometry::Point3D;
    /// let mut point = Point3D::new(1.0, 2.0, 3.0);
    /// *point.x_mut() = 10.0;
    /// assert_eq!(point.x(), 10.0);
    /// ```
    #[inline]
    pub fn x_mut(&mut self) -> &mut T {
        &mut self.coords[0]
    }

    /// Returns a mutable reference to the Y-coordinate
    ///
    /// # Examples
    ///
    /// ```
    /// # use naturalneighbor3d::core::geometry::Point3D;
    /// let mut point = Point3D::new(1.0, 2.0, 3.0);
    /// *point.y_mut() = 20.0;
    /// assert_eq!(point.y(), 20.0);
    /// ```
    #[inline]
    pub fn y_mut(&mut self) -> &mut T {
        &mut self.coords[1]
    }

    /// Returns a mutable reference to the Z-coordinate
    ///
    /// # Examples
    ///
    /// ```
    /// # use naturalneighbor3d::core::geometry::Point3D;
    /// let mut point = Point3D::new(1.0, 2.0, 3.0);
    /// *point.z_mut() = 30.0;
    /// assert_eq!(point.z(), 30.0);
    /// ```
    #[inline]
    pub fn z_mut(&mut self) -> &mut T {
        &mut self.coords[2]
    }
}

impl<T> Point3D<T>
where
    T: Copy
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + PartialOrd
        + Default,
{
    /// Calculates the squared Euclidean distance to another point
    ///
    /// This method avoids the expensive square root operation for better
    /// performance when only relative distances are needed (e.g., for
    /// nearest neighbor searches).
    ///
    /// # Arguments
    ///
    /// * `other` - The target point to measure distance to
    ///
    /// # Returns
    ///
    /// The squared distance between the two points
    ///
    /// # Examples
    ///
    /// ```
    /// # use naturalneighbor3d::core::geometry::Point3D;
    /// let p1 = Point3D::new(0.0, 0.0, 0.0);
    /// let p2 = Point3D::new(3.0, 4.0, 0.0);
    /// let sq_dist = p1.squared_distance_to(&p2);
    /// assert_eq!(sq_dist, 25.0); // 3² + 4² + 0² = 25
    /// ```
    #[inline]
    pub fn squared_distance_to(&self, other: &Self) -> T {
        let dx = self.coords[0] - other.coords[0];
        let dy = self.coords[1] - other.coords[1];
        let dz = self.coords[2] - other.coords[2];
        dx * dx + dy * dy + dz * dz
    }

    /// Calculates the Euclidean distance to another point
    ///
    /// This method computes the true geometric distance between two points
    /// in 3D space using the Pythagorean theorem.
    ///
    /// # Arguments
    ///
    /// * `other` - The target point to measure distance to
    ///
    /// # Returns
    ///
    /// The Euclidean distance as a 64-bit floating-point number
    ///
    /// # Examples
    ///
    /// ```
    /// # use naturalneighbor3d::core::geometry::Point3D;
    /// let p1 = Point3D::new(0.0, 0.0, 0.0);
    /// let p2 = Point3D::new(3.0, 4.0, 0.0);
    /// let distance = p1.distance_to(&p2);
    /// assert!((distance - 5.0).abs() < f64::EPSILON);
    /// ```
    #[inline]
    pub fn distance_to(&self, other: &Self) -> f64
    where
        T: Into<f64>,
    {
        let sq_dist: f64 = self.squared_distance_to(other).into();
        sq_dist.sqrt()
    }
}

/// Vector operations for Point3D
impl<T> Point3D<T>
where
    T: Copy
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + Default,
{
    /// Adds two points component-wise (vector addition)
    ///
    /// # Arguments
    ///
    /// * `other` - The point to add
    ///
    /// # Returns
    ///
    /// A new point representing the vector sum
    ///
    /// # Examples
    ///
    /// ```
    /// # use naturalneighbor3d::core::geometry::Point3D;
    /// let p1 = Point3D::new(1.0, 2.0, 3.0);
    /// let p2 = Point3D::new(4.0, 5.0, 6.0);
    /// let sum = p1.add(&p2);
    /// assert_eq!(sum, Point3D::new(5.0, 7.0, 9.0));
    /// ```
    #[inline]
    pub fn add(&self, other: &Self) -> Self {
        Self::new(
            self.coords[0] + other.coords[0],
            self.coords[1] + other.coords[1],
            self.coords[2] + other.coords[2],
        )
    }

    /// Subtracts two points component-wise (vector subtraction)
    ///
    /// # Arguments
    ///
    /// * `other` - The point to subtract
    ///
    /// # Returns
    ///
    /// A new point representing the vector difference
    ///
    /// # Examples
    ///
    /// ```
    /// # use naturalneighbor3d::core::geometry::Point3D;
    /// let p1 = Point3D::new(5.0, 7.0, 9.0);
    /// let p2 = Point3D::new(1.0, 2.0, 3.0);
    /// let diff = p1.subtract(&p2);
    /// assert_eq!(diff, Point3D::new(4.0, 5.0, 6.0));
    /// ```
    #[inline]
    pub fn subtract(&self, other: &Self) -> Self {
        Self::new(
            self.coords[0] - other.coords[0],
            self.coords[1] - other.coords[1],
            self.coords[2] - other.coords[2],
        )
    }

    /// Multiplies the point by a scalar value
    ///
    /// # Arguments
    ///
    /// * `scalar` - The scalar value to multiply by
    ///
    /// # Returns
    ///
    /// A new point with all coordinates scaled
    ///
    /// # Examples
    ///
    /// ```
    /// # use naturalneighbor3d::core::geometry::Point3D;
    /// let point = Point3D::new(1.0, 2.0, 3.0);
    /// let scaled = point.scale(2.0);
    /// assert_eq!(scaled, Point3D::new(2.0, 4.0, 6.0));
    /// ```
    #[inline]
    pub fn scale(&self, scalar: T) -> Self {
        Self::new(
            self.coords[0] * scalar,
            self.coords[1] * scalar,
            self.coords[2] * scalar,
        )
    }
}

/// Computes the dot product with another point (treating as vectors)
impl<T> Point3D<T>
where
    T: Copy + std::ops::Mul<Output = T> + std::ops::Add<Output = T>,
{
    /// Calculates the dot product with another point
    ///
    /// # Arguments
    ///
    /// * `other` - The other point/vector
    ///
    /// # Returns
    ///
    /// The dot product as a scalar value
    ///
    /// # Examples
    ///
    /// ```
    /// # use naturalneighbor3d::core::geometry::Point3D;
    /// let p1 = Point3D::new(1.0, 2.0, 3.0);
    /// let p2 = Point3D::new(4.0, 5.0, 6.0);
    /// let dot = p1.dot(&p2);
    /// assert_eq!(dot, 32.0); // 1*4 + 2*5 + 3*6 = 32
    /// ```
    #[inline]
    pub fn dot(&self, other: &Self) -> T {
        self.coords[0] * other.coords[0]
            + self.coords[1] * other.coords[1]
            + self.coords[2] * other.coords[2]
    }
}

/// Indexing support for Point3D
impl<T> Index<usize> for Point3D<T> {
    type Output = T;

    /// Returns a reference to the coordinate at the given index
    ///
    /// # Arguments
    ///
    /// * `index` - The coordinate index (0=x, 1=y, 2=z)
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds (>= 3)
    ///
    /// # Examples
    ///
    /// ```
    /// # use naturalneighbor3d::core::geometry::Point3D;
    /// let point = Point3D::new(1.0, 2.0, 3.0);
    /// assert_eq!(point[0], 1.0);
    /// assert_eq!(point[1], 2.0);
    /// assert_eq!(point[2], 3.0);
    /// ```
    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.coords[index]
    }
}

/// Mutable indexing support for Point3D
impl<T> IndexMut<usize> for Point3D<T> {
    /// Returns a mutable reference to the coordinate at the given index
    ///
    /// # Arguments
    ///
    /// * `index` - The coordinate index (0=x, 1=y, 2=z)
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds (>= 3)
    ///
    /// # Examples
    ///
    /// ```
    /// # use naturalneighbor3d::core::geometry::Point3D;
    /// let mut point = Point3D::new(1.0, 2.0, 3.0);
    /// point[0] = 10.0;
    /// assert_eq!(point[0], 10.0);
    /// ```
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.coords[index]
    }
}

/// Conversion from array to Point3D
impl<T> From<[T; 3]> for Point3D<T> {
    /// Creates a Point3D from a 3-element array
    ///
    /// # Arguments
    ///
    /// * `coords` - Array of 3 coordinates [x, y, z]
    ///
    /// # Examples
    ///
    /// ```
    /// # use naturalneighbor3d::core::geometry::Point3D;
    /// let point = Point3D::from([1.0, 2.0, 3.0]);
    /// assert_eq!(point.x(), 1.0);
    /// assert_eq!(point.y(), 2.0);
    /// assert_eq!(point.z(), 3.0);
    /// ```
    #[inline]
    fn from(coords: [T; 3]) -> Self {
        Self { coords }
    }
}

/// Conversion from tuple to Point3D
impl<T> From<(T, T, T)> for Point3D<T>
where
    T: Copy,
{
    /// Creates a Point3D from a 3-tuple
    ///
    /// # Arguments
    ///
    /// * `(x, y, z)` - Tuple of 3 coordinates
    ///
    /// # Examples
    ///
    /// ```
    /// # use naturalneighbor3d::core::geometry::Point3D;
    /// let point = Point3D::from((1.0, 2.0, 3.0));
    /// assert_eq!(point.x(), 1.0);
    /// assert_eq!(point.y(), 2.0);
    /// assert_eq!(point.z(), 3.0);
    /// ```
    #[inline]
    fn from((x, y, z): (T, T, T)) -> Self {
        Self { coords: [x, y, z] }
    }
}

/// Conversion to array from Point3D
impl<T> From<Point3D<T>> for [T; 3] {
    /// Converts a Point3D to a 3-element array
    ///
    /// # Examples
    ///
    /// ```
    /// # use naturalneighbor3d::core::geometry::Point3D;
    /// let point = Point3D::new(1.0, 2.0, 3.0);
    /// let array: [f64; 3] = point.into();
    /// assert_eq!(array, [1.0, 2.0, 3.0]);
    /// ```
    #[inline]
    fn from(point: Point3D<T>) -> Self {
        point.coords
    }
}

/// Display formatting for Point3D
impl<T> std::fmt::Display for Point3D<T>
where
    T: std::fmt::Display,
{
    /// Formats the point for display
    ///
    /// # Examples
    ///
    /// ```
    /// # use naturalneighbor3d::core::geometry::Point3D;
    /// let point = Point3D::new(1.5, 2.7, 3.9);
    /// assert_eq!(format!("{}", point), "Point3D(1.5, 2.7, 3.9)");
    /// ```
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Point3D({}, {}, {})",
            self.coords[0], self.coords[1], self.coords[2]
        )
    }
}

/// Bounding box for a collection of 3D points
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingBox<T = f64> {
    /// Minimum coordinates
    pub min: Point3D<T>,
    /// Maximum coordinates
    pub max: Point3D<T>,
}

impl<T> BoundingBox<T>
where
    T: Copy + PartialOrd + Default,
{
    /// Creates a new bounding box with given min and max points
    ///
    /// # Arguments
    ///
    /// * `min` - Minimum coordinates
    /// * `max` - Maximum coordinates
    ///
    /// # Examples
    ///
    /// ```
    /// # use naturalneighbor3d::core::geometry::{Point3D, BoundingBox};
    /// let min = Point3D::new(0.0, 0.0, 0.0);
    /// let max = Point3D::new(10.0, 10.0, 10.0);
    /// let bbox = BoundingBox::new(min, max);
    /// ```
    #[inline]
    pub fn new(min: Point3D<T>, max: Point3D<T>) -> Self {
        Self { min, max }
    }

    /// Creates an empty bounding box
    ///
    /// # Examples
    ///
    /// ```
    /// # use naturalneighbor3d::core::geometry::BoundingBox;
    /// let bbox = BoundingBox::<f64>::empty();
    /// ```
    #[inline]
    pub fn empty() -> Self {
        Self {
            min: Point3D::origin(),
            max: Point3D::origin(),
        }
    }

    /// Checks if a point is contained within the bounding box
    ///
    /// # Arguments
    ///
    /// * `point` - The point to test
    ///
    /// # Returns
    ///
    /// `true` if the point is inside or on the boundary
    ///
    /// # Examples
    ///
    /// ```
    /// # use naturalneighbor3d::core::geometry::{Point3D, BoundingBox};
    /// let bbox = BoundingBox::new(
    ///     Point3D::new(0.0, 0.0, 0.0),
    ///     Point3D::new(10.0, 10.0, 10.0)
    /// );
    /// assert!(bbox.contains(&Point3D::new(5.0, 5.0, 5.0)));
    /// assert!(!bbox.contains(&Point3D::new(15.0, 5.0, 5.0)));
    /// ```
    #[inline]
    pub fn contains(&self, point: &Point3D<T>) -> bool {
        point.x() >= self.min.x()
            && point.x() <= self.max.x()
            && point.y() >= self.min.y()
            && point.y() <= self.max.y()
            && point.z() >= self.min.z()
            && point.z() <= self.max.z()
    }

    /// Expands the bounding box to include a new point
    ///
    /// # Arguments
    ///
    /// * `point` - The point to include
    ///
    /// # Examples
    ///
    /// ```
    /// # use naturalneighbor3d::core::geometry::{Point3D, BoundingBox};
    /// let mut bbox = BoundingBox::new(
    ///     Point3D::new(0.0, 0.0, 0.0),
    ///     Point3D::new(5.0, 5.0, 5.0)
    /// );
    /// bbox.expand(&Point3D::new(10.0, 3.0, 3.0));
    /// assert_eq!(bbox.max.x(), 10.0);
    /// ```
    pub fn expand(&mut self, point: &Point3D<T>) {
        if point.x() < self.min.x() {
            self.min.coords[0] = point.x();
        }
        if point.y() < self.min.y() {
            self.min.coords[1] = point.y();
        }
        if point.z() < self.min.z() {
            self.min.coords[2] = point.z();
        }

        if point.x() > self.max.x() {
            self.max.coords[0] = point.x();
        }
        if point.y() > self.max.y() {
            self.max.coords[1] = point.y();
        }
        if point.z() > self.max.z() {
            self.max.coords[2] = point.z();
        }
    }
}

impl<T> BoundingBox<T>
where
    T: Copy + std::ops::Sub<Output = T> + Default,
{
    /// Calculates the dimensions of the bounding box
    ///
    /// # Returns
    ///
    /// A Point3D representing the width, height, and depth
    ///
    /// # Examples
    ///
    /// ```
    /// # use naturalneighbor3d::core::geometry::{Point3D, BoundingBox};
    /// let bbox = BoundingBox::new(
    ///     Point3D::new(0.0, 0.0, 0.0),
    ///     Point3D::new(10.0, 5.0, 3.0)
    /// );
    /// let dimensions = bbox.dimensions();
    /// assert_eq!(dimensions.x(), 10.0);
    /// assert_eq!(dimensions.y(), 5.0);
    /// assert_eq!(dimensions.z(), 3.0);
    /// ```
    #[inline]
    pub fn dimensions(&self) -> Point3D<T> {
        Point3D::new(
            self.max.x() - self.min.x(),
            self.max.y() - self.min.y(),
            self.max.z() - self.min.z(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_creation() {
        let p1 = Point3D::new(1.0, 2.0, 3.0);
        assert_eq!(p1.x(), 1.0);
        assert_eq!(p1.y(), 2.0);
        assert_eq!(p1.z(), 3.0);

        let p2 = Point3D::from([1.0, 2.0, 3.0]);
        assert_eq!(p1, p2);

        let p3 = Point3D::from((1.0, 2.0, 3.0));
        assert_eq!(p1, p3);

        let origin = Point3D::<f64>::origin();
        assert_eq!(origin.x(), 0.0);
        assert_eq!(origin.y(), 0.0);
        assert_eq!(origin.z(), 0.0);
    }

    #[test]
    fn test_distance_calculations() {
        let p1 = Point3D::new(0.0, 0.0, 0.0);
        let p2 = Point3D::new(3.0, 4.0, 0.0);

        let squared_dist = p1.squared_distance_to(&p2);
        assert_eq!(squared_dist, 25.0);

        let dist = p1.distance_to(&p2);
        assert!((dist - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector_operations() {
        let p1 = Point3D::new(1.0, 2.0, 3.0);
        let p2 = Point3D::new(4.0, 5.0, 6.0);

        let sum = p1.add(&p2);
        assert_eq!(sum, Point3D::new(5.0, 7.0, 9.0));

        let diff = p2.subtract(&p1);
        assert_eq!(diff, Point3D::new(3.0, 3.0, 3.0));

        let scaled = p1.scale(2.0);
        assert_eq!(scaled, Point3D::new(2.0, 4.0, 6.0));

        let dot = p1.dot(&p2);
        assert_eq!(dot, 32.0); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_indexing() {
        let mut p = Point3D::new(1.0, 2.0, 3.0);
        assert_eq!(p[0], 1.0);
        assert_eq!(p[1], 2.0);
        assert_eq!(p[2], 3.0);

        p[0] = 10.0;
        assert_eq!(p[0], 10.0);
        assert_eq!(p.x(), 10.0);
    }

    #[test]
    fn test_conversions() {
        let point = Point3D::new(1.0, 2.0, 3.0);
        let array: [f64; 3] = point.into();
        assert_eq!(array, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_display() {
        let point = Point3D::new(1.5, 2.7, 3.9);
        let display_str = format!("{}", point);
        assert_eq!(display_str, "Point3D(1.5, 2.7, 3.9)");
    }

    #[test]
    fn test_bounding_box() {
        let mut bbox =
            BoundingBox::new(Point3D::new(0.0, 0.0, 0.0), Point3D::new(10.0, 10.0, 10.0));

        // Test containment
        assert!(bbox.contains(&Point3D::new(5.0, 5.0, 5.0)));
        assert!(bbox.contains(&Point3D::new(0.0, 0.0, 0.0)));
        assert!(bbox.contains(&Point3D::new(10.0, 10.0, 10.0)));
        assert!(!bbox.contains(&Point3D::new(15.0, 5.0, 5.0)));
        assert!(!bbox.contains(&Point3D::new(-1.0, 5.0, 5.0)));

        // Test expansion
        bbox.expand(&Point3D::new(15.0, 3.0, 3.0));
        assert_eq!(bbox.max.x(), 15.0);
        assert_eq!(bbox.max.y(), 10.0);

        bbox.expand(&Point3D::new(-5.0, 3.0, 3.0));
        assert_eq!(bbox.min.x(), -5.0);

        // Test dimensions
        let dimensions = bbox.dimensions();
        assert_eq!(dimensions.x(), 20.0); // 15.0 - (-5.0)
        assert_eq!(dimensions.y(), 10.0);
        assert_eq!(dimensions.z(), 10.0);
    }

    #[test]
    fn test_mutable_coordinates() {
        let mut point = Point3D::new(1.0, 2.0, 3.0);

        *point.x_mut() = 10.0;
        *point.y_mut() = 20.0;
        *point.z_mut() = 30.0;

        assert_eq!(point.x(), 10.0);
        assert_eq!(point.y(), 20.0);
        assert_eq!(point.z(), 30.0);
    }
}
