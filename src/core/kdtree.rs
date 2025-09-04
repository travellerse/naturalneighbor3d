//! # KD-Tree Module
//!
//! Provides an efficient k-dimensional tree implementation for fast spatial
//! queries in 3D space. This module is optimized for nearest neighbor searches
//! required by natural neighbor interpolation algorithms.
//!
//! The KD-tree implementation supports:
//! - Fast nearest neighbor queries in O(log n) average time
//! - Efficient construction from unsorted data
//! - Memory-efficient node storage
//! - Support for both exact and approximate queries

use super::geometry::Point3D;

/// Result of a nearest neighbor query
///
/// Contains both the value at the found point and the squared distance
/// to avoid unnecessary square root calculations for performance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QueryResult {
    /// The data value associated with the nearest point
    pub value: f64,
    /// The squared Euclidean distance to the query point
    pub distance: f64,
}

impl QueryResult {
    /// Creates a new query result
    ///
    /// # Arguments
    ///
    /// * `value` - The data value at the found point
    /// * `distance` - The squared distance to the query point
    ///
    /// # Examples
    ///
    /// ```
    /// # use naturalneighbor3d::core::QueryResult;
    /// let result = QueryResult::new(42.0, 25.0);
    /// assert_eq!(result.value, 42.0);
    /// assert_eq!(result.distance, 25.0);
    /// ```
    #[inline]
    pub fn new(value: f64, distance: f64) -> Self {
        Self { value, distance }
    }

    /// Returns the actual (non-squared) distance to the query point
    ///
    /// # Examples
    ///
    /// ```
    /// # use naturalneighbor3d::core::QueryResult;
    /// let result = QueryResult::new(42.0, 25.0);
    /// assert_eq!(result.euclidean_distance(), 5.0);
    /// ```
    #[inline]
    pub fn euclidean_distance(&self) -> f64 {
        self.distance.sqrt()
    }
}

/// Internal node structure for the KD-tree
///
/// Each node stores a 3D point, its associated value, the splitting axis,
/// and references to left and right children. The tree is balanced to
/// ensure optimal query performance.
#[derive(Debug)]
struct KdNode {
    /// The 3D point stored at this node
    point: Point3D<f64>,
    /// The data value associated with this point
    value: f64,
    /// The axis used for splitting at this node (0=x, 1=y, 2=z)
    axis: usize,
    /// Left child (points with smaller coordinate on splitting axis)
    left: Option<Box<KdNode>>,
    /// Right child (points with larger coordinate on splitting axis)
    right: Option<Box<KdNode>>,
}

impl KdNode {
    /// Creates a new KD-tree node
    ///
    /// # Arguments
    ///
    /// * `point` - The 3D point for this node
    /// * `value` - The data value associated with this point
    /// * `axis` - The splitting axis for this node
    ///
    /// # Returns
    ///
    /// A new KdNode instance
    #[inline]
    fn new(point: Point3D<f64>, value: f64, axis: usize) -> Self {
        Self {
            point,
            value,
            axis,
            left: None,
            right: None,
        }
    }

    /// Checks if this node is a leaf (has no children)
    ///
    /// # Returns
    ///
    /// `true` if the node has no children, `false` otherwise
    #[inline]
    #[allow(dead_code)]
    fn is_leaf(&self) -> bool {
        self.left.is_none() && self.right.is_none()
    }
}

/// A high-performance KD-tree for 3D spatial queries
///
/// This implementation provides efficient nearest neighbor searches
/// for 3D points with associated scalar values. The tree is automatically
/// balanced during construction for optimal query performance.
///
/// # Examples
///
/// ```
/// # use naturalneighbor3d::core::kdtree::KdTree;
/// # use naturalneighbor3d::core::geometry::Point3D;
/// let mut tree = KdTree::new();
///
/// // Add some points
/// tree.add(Point3D::new(1.0, 2.0, 3.0), 10.0);
/// tree.add(Point3D::new(4.0, 5.0, 6.0), 20.0);
/// tree.add(Point3D::new(0.0, 0.0, 0.0), 5.0);
///
/// // Build the tree (must be called after adding points)
/// tree.build();
///
/// // Query for nearest neighbor
/// let query_point = Point3D::new(0.1, 0.1, 0.1);
/// if let Some(result) = tree.nearest(&query_point) {
///     println!("Nearest value: {}, distance: {}", result.value, result.euclidean_distance());
/// }
/// ```
pub struct KdTree {
    /// Root node of the tree (None if empty)
    root: Option<Box<KdNode>>,
    /// Temporary storage for points before tree construction
    nodes: Vec<(Point3D<f64>, f64)>,
    /// Total number of nodes in the built tree
    node_count: usize,
}

impl KdTree {
    /// Creates a new empty KD-tree
    ///
    /// # Examples
    ///
    /// ```
    /// # use naturalneighbor3d::core::kdtree::KdTree;
    /// let tree = KdTree::new();
    /// assert!(tree.is_empty());
    /// ```
    pub fn new() -> Self {
        Self {
            root: None,
            nodes: Vec::new(),
            node_count: 0,
        }
    }

    /// Creates a new KD-tree with pre-allocated capacity
    ///
    /// This can improve performance when the approximate number of
    /// points is known in advance by avoiding memory reallocations.
    ///
    /// # Arguments
    ///
    /// * `capacity` - The expected number of points to be added
    ///
    /// # Examples
    ///
    /// ```
    /// # use naturalneighbor3d::core::kdtree::KdTree;
    /// let tree = KdTree::with_capacity(1000);
    /// assert!(tree.is_empty());
    /// ```
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            root: None,
            nodes: Vec::with_capacity(capacity),
            node_count: 0,
        }
    }

    /// Adds a point and its associated value to the tree
    ///
    /// Points are stored temporarily until `build()` is called.
    /// This allows for efficient bulk construction of the tree.
    ///
    /// # Arguments
    ///
    /// * `point` - The 3D point to add
    /// * `value` - The scalar value associated with this point
    ///
    /// # Examples
    ///
    /// ```
    /// # use naturalneighbor3d::core::kdtree::KdTree;
    /// # use naturalneighbor3d::core::geometry::Point3D;
    /// let mut tree = KdTree::new();
    /// tree.add(Point3D::new(1.0, 2.0, 3.0), 42.0);
    /// ```
    pub fn add(&mut self, point: Point3D<f64>, value: f64) {
        self.nodes.push((point, value));
    }

    /// Builds the KD-tree from all added points
    ///
    /// This method must be called after adding points and before
    /// performing queries. It constructs a balanced tree for
    /// optimal query performance.
    ///
    /// # Examples
    ///
    /// ```
    /// # use naturalneighbor3d::core::kdtree::KdTree;
    /// # use naturalneighbor3d::core::geometry::Point3D;
    /// let mut tree = KdTree::new();
    /// tree.add(Point3D::new(1.0, 2.0, 3.0), 42.0);
    /// tree.build();
    /// assert!(!tree.is_empty());
    /// ```
    pub fn build(&mut self) {
        if !self.nodes.is_empty() {
            let mut nodes = std::mem::take(&mut self.nodes);
            self.node_count = nodes.len();
            self.root = Self::build_recursive(&mut nodes, 0);
        }
    }

    /// Recursively builds a balanced KD-tree
    ///
    /// This is the core tree construction algorithm that recursively
    /// partitions points along alternating axes to create a balanced tree.
    ///
    /// # Arguments
    ///
    /// * `nodes` - Mutable slice of points to partition
    /// * `depth` - Current tree depth (determines splitting axis)
    ///
    /// # Returns
    ///
    /// An optional boxed node representing the subtree root
    fn build_recursive(nodes: &mut [(Point3D<f64>, f64)], depth: usize) -> Option<Box<KdNode>> {
        if nodes.is_empty() {
            return None;
        }

        let axis = depth % 3; // Cycle through x, y, z axes

        // Sort points by the current axis coordinate
        nodes.sort_by(|a, b| a.0[axis].partial_cmp(&b.0[axis]).unwrap());

        let median = nodes.len() / 2;
        let (point, value) = nodes[median];

        // Split the array around the median
        let (left_nodes, rest) = nodes.split_at_mut(median);
        let right_nodes = &mut rest[1..];

        // Create the node
        let mut node = KdNode::new(point, value, axis);

        // Recursively build left and right subtrees
        node.left = Self::build_recursive(left_nodes, depth + 1);
        node.right = Self::build_recursive(right_nodes, depth + 1);

        Some(Box::new(node))
    }

    /// Finds the nearest neighbor to a query point
    ///
    /// Performs an efficient search through the KD-tree to find the
    /// point closest to the given query point.
    ///
    /// # Arguments
    ///
    /// * `query` - The point to search near
    ///
    /// # Returns
    ///
    /// An optional `QueryResult` containing the nearest point's value
    /// and squared distance, or `None` if the tree is empty
    ///
    /// # Examples
    ///
    /// ```
    /// # use naturalneighbor3d::core::kdtree::KdTree;
    /// # use naturalneighbor3d::core::geometry::Point3D;
    /// let mut tree = KdTree::new();
    /// tree.add(Point3D::new(0.0, 0.0, 0.0), 1.0);
    /// tree.add(Point3D::new(10.0, 10.0, 10.0), 2.0);
    /// tree.build();
    ///
    /// let query = Point3D::new(1.0, 1.0, 1.0);
    /// if let Some(result) = tree.nearest(&query) {
    ///     assert_eq!(result.value, 1.0); // Closest to origin
    /// }
    /// ```
    pub fn nearest(&self, query: &Point3D<f64>) -> Option<QueryResult> {
        self.root
            .as_ref()
            .map(|root| self.nearest_recursive(root, query, f64::INFINITY))
    }

    /// Recursively searches for the nearest neighbor
    ///
    /// This implements the standard KD-tree nearest neighbor algorithm
    /// with pruning to avoid unnecessary subtree exploration.
    ///
    /// # Arguments
    ///
    /// * `node` - Current node being examined
    /// * `query` - The query point
    /// * `best_dist_sq` - Current best squared distance found
    ///
    /// # Returns
    ///
    /// The best `QueryResult` found in this subtree
    fn nearest_recursive(
        &self,
        node: &KdNode,
        query: &Point3D<f64>,
        mut best_dist_sq: f64,
    ) -> QueryResult {
        // Calculate distance to current node
        let dist_sq = query.squared_distance_to(&node.point);
        let mut best_result = QueryResult::new(node.value, dist_sq);

        if dist_sq < best_dist_sq {
            best_dist_sq = dist_sq;
        }

        let axis = node.axis;
        let diff = query[axis] - node.point[axis];

        // Determine which subtree to search first (closest side)
        let (primary, secondary) = if diff <= 0.0 {
            (&node.left, &node.right)
        } else {
            (&node.right, &node.left)
        };

        // Search the closer subtree first
        if let Some(primary_node) = primary {
            let result = self.nearest_recursive(primary_node, query, best_dist_sq);
            if result.distance < best_result.distance {
                best_result = result;
                best_dist_sq = result.distance;
            }
        }

        // Check if we need to search the farther subtree
        // Only search if the splitting plane is close enough to potentially contain a better point
        if diff * diff < best_dist_sq {
            if let Some(secondary_node) = secondary {
                let result = self.nearest_recursive(secondary_node, query, best_dist_sq);
                if result.distance < best_result.distance {
                    best_result = result;
                }
            }
        }

        best_result
    }

    /// Alias for `nearest()` to maintain backward compatibility
    ///
    /// This method provides the same functionality as `nearest()`
    /// but uses an iterative naming convention.
    ///
    /// # Arguments
    ///
    /// * `query` - The point to search near
    ///
    /// # Returns
    ///
    /// An optional `QueryResult` containing the nearest point's value
    /// and squared distance, or `None` if the tree is empty
    #[inline]
    pub fn nearest_iterative(&self, query: &Point3D<f64>) -> Option<QueryResult> {
        self.nearest(query)
    }

    /// Finds all neighbors within a specified radius
    ///
    /// Returns all points in the tree that are within the given
    /// squared distance from the query point.
    ///
    /// # Arguments
    ///
    /// * `query` - The center point for the search
    /// * `radius_sq` - The squared radius of the search area
    ///
    /// # Returns
    ///
    /// A vector of `QueryResult` objects for all points within the radius
    ///
    /// # Examples
    ///
    /// ```
    /// # use naturalneighbor3d::core::kdtree::KdTree;
    /// # use naturalneighbor3d::core::geometry::Point3D;
    /// let mut tree = KdTree::new();
    /// tree.add(Point3D::new(0.0, 0.0, 0.0), 1.0);
    /// tree.add(Point3D::new(1.0, 0.0, 0.0), 2.0);
    /// tree.add(Point3D::new(10.0, 0.0, 0.0), 3.0);
    /// tree.build();
    ///
    /// let query = Point3D::new(0.5, 0.0, 0.0);
    /// let neighbors = tree.neighbors_within_radius(&query, 2.0); // radius_sq = 4.0
    /// assert_eq!(neighbors.len(), 2); // First two points are within radius
    /// ```
    pub fn neighbors_within_radius(
        &self,
        query: &Point3D<f64>,
        radius_sq: f64,
    ) -> Vec<QueryResult> {
        let mut results = Vec::new();
        if let Some(root) = &self.root {
            self.radius_search_recursive(root, query, radius_sq, &mut results);
        }
        results
    }

    /// Recursively searches for neighbors within a radius
    ///
    /// # Arguments
    ///
    /// * `node` - Current node being examined
    /// * `query` - The query point
    /// * `radius_sq` - The squared radius of the search area
    /// * `results` - Vector to accumulate results
    fn radius_search_recursive(
        &self,
        node: &KdNode,
        query: &Point3D<f64>,
        radius_sq: f64,
        results: &mut Vec<QueryResult>,
    ) {
        let dist_sq = query.squared_distance_to(&node.point);

        // If this point is within radius, add it to results
        if dist_sq <= radius_sq {
            results.push(QueryResult::new(node.value, dist_sq));
        }

        let axis = node.axis;
        let diff = query[axis] - node.point[axis];

        // Search both subtrees if the splitting plane intersects the search radius
        if let Some(left) = &node.left {
            if diff - radius_sq.sqrt() <= 0.0 {
                self.radius_search_recursive(left, query, radius_sq, results);
            }
        }

        if let Some(right) = &node.right {
            if diff + radius_sq.sqrt() >= 0.0 {
                self.radius_search_recursive(right, query, radius_sq, results);
            }
        }
    }

    /// Checks if the tree is empty
    ///
    /// # Returns
    ///
    /// `true` if the tree contains no nodes, `false` otherwise
    ///
    /// # Examples
    ///
    /// ```
    /// # use naturalneighbor3d::core::kdtree::KdTree;
    /// let tree = KdTree::new();
    /// assert!(tree.is_empty());
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.root.is_none()
    }

    /// Returns the number of nodes in the tree
    ///
    /// # Returns
    ///
    /// The total number of points stored in the tree
    ///
    /// # Examples
    ///
    /// ```
    /// # use naturalneighbor3d::core::kdtree::KdTree;
    /// # use naturalneighbor3d::core::geometry::Point3D;
    /// let mut tree = KdTree::new();
    /// tree.add(Point3D::new(1.0, 2.0, 3.0), 42.0);
    /// tree.build();
    /// assert_eq!(tree.len(), 1);
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.node_count
    }

    /// Returns the depth (height) of the tree
    ///
    /// # Returns
    ///
    /// The maximum depth from root to any leaf node
    ///
    /// # Examples
    ///
    /// ```
    /// # use naturalneighbor3d::core::kdtree::KdTree;
    /// # use naturalneighbor3d::core::geometry::Point3D;
    /// let mut tree = KdTree::new();
    /// tree.add(Point3D::new(1.0, 2.0, 3.0), 42.0);
    /// tree.build();
    /// assert_eq!(tree.depth(), 1);
    /// ```
    pub fn depth(&self) -> usize {
        self.root.as_ref().map_or(0, |root| self.node_depth(root))
    }

    /// Recursively calculates the depth of a subtree
    ///
    /// # Arguments
    ///
    /// * `node` - The root node of the subtree
    ///
    /// # Returns
    ///
    /// The depth of the subtree rooted at the given node
    fn node_depth(&self, node: &KdNode) -> usize {
        let left_depth = node.left.as_ref().map_or(0, |n| self.node_depth(n));
        let right_depth = node.right.as_ref().map_or(0, |n| self.node_depth(n));
        1 + left_depth.max(right_depth)
    }

    /// Clears all nodes from the tree
    ///
    /// Removes all points and resets the tree to an empty state.
    /// This is more efficient than creating a new tree instance.
    ///
    /// # Examples
    ///
    /// ```
    /// # use naturalneighbor3d::core::kdtree::KdTree;
    /// # use naturalneighbor3d::core::geometry::Point3D;
    /// let mut tree = KdTree::new();
    /// tree.add(Point3D::new(1.0, 2.0, 3.0), 42.0);
    /// tree.build();
    /// assert!(!tree.is_empty());
    ///
    /// tree.clear();
    /// assert!(tree.is_empty());
    /// ```
    pub fn clear(&mut self) {
        self.root = None;
        self.nodes.clear();
        self.node_count = 0;
    }

    /// Returns statistics about the tree structure
    ///
    /// Provides information about tree balance and structure
    /// that can be useful for performance analysis.
    ///
    /// # Returns
    ///
    /// A `TreeStats` struct containing various tree metrics
    pub fn stats(&self) -> TreeStats {
        TreeStats {
            node_count: self.node_count,
            depth: self.depth(),
            is_balanced: self.is_balanced(),
        }
    }

    /// Checks if the tree is reasonably balanced
    ///
    /// A tree is considered balanced if its depth is within
    /// a reasonable bound of the optimal depth for the number of nodes.
    ///
    /// # Returns
    ///
    /// `true` if the tree is well-balanced, `false` otherwise
    fn is_balanced(&self) -> bool {
        if self.node_count == 0 {
            return true;
        }

        let optimal_depth = (self.node_count as f64).log2().ceil() as usize;
        let actual_depth = self.depth();

        // Consider balanced if actual depth is within 50% of optimal
        actual_depth <= optimal_depth + (optimal_depth / 2).max(1)
    }
}

/// Statistics about the KD-tree structure
#[derive(Debug, Clone, PartialEq)]
pub struct TreeStats {
    /// Total number of nodes in the tree
    pub node_count: usize,
    /// Maximum depth from root to any leaf
    pub depth: usize,
    /// Whether the tree is reasonably balanced
    pub is_balanced: bool,
}

impl Default for KdTree {
    /// Creates a new empty KD-tree
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kdtree_basic() {
        let mut tree = KdTree::new();
        tree.add(Point3D::new(1.0, 2.0, 3.0), 10.0);
        tree.add(Point3D::new(4.0, 5.0, 6.0), 20.0);
        tree.add(Point3D::new(0.0, 0.0, 0.0), 5.0);
        tree.build();

        assert!(!tree.is_empty());
        assert_eq!(tree.len(), 3);

        let query = Point3D::new(0.1, 0.1, 0.1);
        let result = tree.nearest(&query).unwrap();

        // Should find the nearest point (0,0,0) with value 5.0
        assert!((result.value - 5.0).abs() < f64::EPSILON);
        assert!(result.distance < 1.0); // Distance should be small
    }

    #[test]
    fn test_empty_tree() {
        let tree = KdTree::new();
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
        assert_eq!(tree.depth(), 0);

        let query = Point3D::new(1.0, 2.0, 3.0);
        assert!(tree.nearest(&query).is_none());
    }

    #[test]
    fn test_single_node() {
        let mut tree = KdTree::new();
        tree.add(Point3D::new(1.0, 2.0, 3.0), 42.0);
        tree.build();

        assert!(!tree.is_empty());
        assert_eq!(tree.len(), 1);
        assert_eq!(tree.depth(), 1);

        let query = Point3D::new(10.0, 20.0, 30.0);
        let result = tree.nearest(&query).unwrap();

        assert!((result.value - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_radius_search() {
        let mut tree = KdTree::new();
        tree.add(Point3D::new(0.0, 0.0, 0.0), 1.0);
        tree.add(Point3D::new(1.0, 0.0, 0.0), 2.0);
        tree.add(Point3D::new(2.0, 0.0, 0.0), 3.0);
        tree.add(Point3D::new(10.0, 0.0, 0.0), 4.0);
        tree.build();

        let query = Point3D::new(0.5, 0.0, 0.0);
        let neighbors = tree.neighbors_within_radius(&query, 4.0); // radius_sq = 4.0, so radius = 2.0

        // Should find the first 3 points (within radius 2.0)
        assert_eq!(neighbors.len(), 3);

        let values: Vec<f64> = neighbors.iter().map(|r| r.value).collect();
        assert!(values.contains(&1.0));
        assert!(values.contains(&2.0));
        assert!(values.contains(&3.0));
    }

    #[test]
    fn test_query_result() {
        let result = QueryResult::new(42.0, 25.0);
        assert_eq!(result.value, 42.0);
        assert_eq!(result.distance, 25.0);
        assert_eq!(result.euclidean_distance(), 5.0);
    }

    #[test]
    fn test_tree_stats() {
        let mut tree = KdTree::new();
        for i in 0..7 {
            tree.add(Point3D::new(i as f64, 0.0, 0.0), i as f64);
        }
        tree.build();

        let stats = tree.stats();
        assert_eq!(stats.node_count, 7);
        assert!(stats.depth >= 3); // Should be reasonably balanced
        assert!(stats.is_balanced);
    }

    #[test]
    fn test_clear() {
        let mut tree = KdTree::new();
        tree.add(Point3D::new(1.0, 2.0, 3.0), 42.0);
        tree.build();
        assert!(!tree.is_empty());

        tree.clear();
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
        assert_eq!(tree.depth(), 0);
    }

    #[test]
    fn test_with_capacity() {
        let tree = KdTree::with_capacity(1000);
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
    }

    #[test]
    fn test_nearest_iterative_alias() {
        let mut tree = KdTree::new();
        tree.add(Point3D::new(0.0, 0.0, 0.0), 1.0);
        tree.build();

        let query = Point3D::new(0.1, 0.1, 0.1);

        let result1 = tree.nearest(&query);
        let result2 = tree.nearest_iterative(&query);

        assert_eq!(result1, result2);
    }

    #[test]
    fn test_performance_characteristics() {
        // Test with a larger dataset to verify performance characteristics
        let mut tree = KdTree::with_capacity(1000);

        // Add points in a grid pattern
        for i in 0..10 {
            for j in 0..10 {
                for k in 0..10 {
                    tree.add(
                        Point3D::new(i as f64, j as f64, k as f64),
                        (i * 100 + j * 10 + k) as f64,
                    );
                }
            }
        }

        tree.build();
        assert_eq!(tree.len(), 1000);

        // Tree should be reasonably balanced
        let stats = tree.stats();
        assert!(stats.is_balanced);
        assert!(stats.depth <= 15); // Should not be too deep for 1000 nodes

        // Test query performance
        let query = Point3D::new(5.5, 5.5, 5.5);
        let result = tree.nearest(&query);
        assert!(result.is_some());
    }
}
