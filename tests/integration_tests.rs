//! Integration tests for the natural neighbor 3D interpolation library
//!
//! This file tests the integration between all modules and validates
//! the complete workflow from input validation to interpolation.

use naturalneighbor3d::core::geometry::Point3D;
use naturalneighbor3d::core::kdtree::{KdTree, QueryResult};
use naturalneighbor3d::utils::grid;

/// 测试辅助函数：根据数据和查询点创建 KD 树并查询
fn query_tree(data: &[(f64, f64, f64, f64)], query: Point3D<f64>) -> Option<QueryResult> {
    let mut tree = KdTree::new();

    for &(x, y, z, value) in data {
        tree.add(Point3D::new(x, y, z), value);
    }

    tree.build();
    tree.nearest(&query)
}

#[cfg(test)]
mod kdtree_tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_single_point() {
        let value = 5.0;
        let tree_data = &[(0.0, 0.0, 0.0, value)];

        // 测试查询同一点
        let result = query_tree(tree_data, Point3D::new(0.0, 0.0, 0.0)).unwrap();
        assert_relative_eq!(result.distance, 0.0, epsilon = 1e-10);
        assert_relative_eq!(result.value, value, epsilon = 1e-10);

        // 测试查询远处的点
        let result = query_tree(tree_data, Point3D::new(0.0, 0.0, 10.0)).unwrap();
        assert_relative_eq!(result.distance, 100.0, epsilon = 1e-10);
        assert_relative_eq!(result.value, value, epsilon = 1e-10);

        let result = query_tree(tree_data, Point3D::new(0.0, 0.0, -10.0)).unwrap();
        assert_relative_eq!(result.distance, 100.0, epsilon = 1e-10);
        assert_relative_eq!(result.value, value, epsilon = 1e-10);
    }

    #[test]
    fn test_two_points() {
        let value1 = 7.0;
        let value2 = 5.0;
        let tree_data = &[(0.0, 0.0, 0.0, value1), (0.0, 0.0, 10.0, value2)];

        // 测试查询第二个点
        let result = query_tree(tree_data, Point3D::new(0.0, 0.0, 10.0)).unwrap();
        assert_relative_eq!(result.distance, 0.0, epsilon = 1e-10);
        assert_relative_eq!(result.value, value2, epsilon = 1e-10);

        // 测试查询远处的点，应该返回第一个点
        let result = query_tree(tree_data, Point3D::new(-10.0, 0.0, 0.0)).unwrap();
        assert_relative_eq!(result.distance, 100.0, epsilon = 1e-10);
        assert_relative_eq!(result.value, value1, epsilon = 1e-10);

        // 测试查询第一个点
        let result = query_tree(tree_data, Point3D::new(0.0, 0.0, 0.0)).unwrap();
        assert_relative_eq!(result.distance, 0.0, epsilon = 1e-10);
        assert_relative_eq!(result.value, value1, epsilon = 1e-10);

        // 测试查询距离第一个点更近的位置
        let result = query_tree(tree_data, Point3D::new(0.0, 0.0, 4.0)).unwrap();
        assert_relative_eq!(result.distance, 16.0, epsilon = 1e-10);
        assert_relative_eq!(result.value, value1, epsilon = 1e-10);

        // 测试查询距离第二个点更近的位置
        let result = query_tree(tree_data, Point3D::new(0.0, 0.0, 6.0)).unwrap();
        assert_relative_eq!(result.distance, 16.0, epsilon = 1e-10);
        assert_relative_eq!(result.value, value2, epsilon = 1e-10);
    }

    #[test]
    fn test_middle_point() {
        let value1 = 7.0;
        let value2 = 5.0;
        let tree_data = &[(0.0, 0.0, 0.0, value1), (0.0, 0.0, 10.0, value2)];

        // 测试查询中点 - 应该返回距离相等的任意一个
        let result = query_tree(tree_data, Point3D::new(0.0, 0.0, 5.0)).unwrap();
        assert_relative_eq!(result.distance, 25.0, epsilon = 1e-10);
        // 中点距离两个点相等，可能返回任意一个值
        assert!(result.value == value1 || result.value == value2);

        // 测试其他位置的中点查询
        let result = query_tree(tree_data, Point3D::new(-100.0, 100.0, 5.0)).unwrap();
        assert!(result.value == value1 || result.value == value2);

        let result = query_tree(tree_data, Point3D::new(-100.0, 0.0, 5.0)).unwrap();
        assert!(result.value == value1 || result.value == value2);
    }

    #[test]
    fn test_cube() {
        let value000 = 1.0;
        let value100 = 2.0;
        let value010 = 3.0;
        let value001 = 4.0;
        let value110 = 5.0;
        let value011 = 6.0;
        let value101 = 7.0;
        let value111 = 8.0;

        let tree_data = &[
            (0.0, 0.0, 0.0, value000),
            (1.0, 0.0, 0.0, value100),
            (0.0, 1.0, 0.0, value010),
            (0.0, 0.0, 1.0, value001),
            (1.0, 1.0, 0.0, value110),
            (0.0, 1.0, 1.0, value011),
            (1.0, 0.0, 1.0, value101),
            (1.0, 1.0, 1.0, value111),
        ];

        // 测试查询立方体顶点
        let result = query_tree(tree_data, Point3D::new(0.0, 0.0, 0.0)).unwrap();
        assert_relative_eq!(result.value, value000, epsilon = 1e-10);

        let result = query_tree(tree_data, Point3D::new(1.0, 1.0, 1.0)).unwrap();
        assert_relative_eq!(result.value, value111, epsilon = 1e-10);

        // 测试查询接近各个顶点的位置
        let bump = 0.000000000001;
        let test_cases = [
            (0.5 - bump, 0.5 - bump, 0.5 - bump, value000),
            (0.5 + bump, 0.5 - bump, 0.5 - bump, value100),
            (0.5 - bump, 0.5 + bump, 0.5 - bump, value010),
            (0.5 - bump, 0.5 - bump, 0.5 + bump, value001),
            (0.5 + bump, 0.5 + bump, 0.5 - bump, value110),
            (0.5 - bump, 0.5 + bump, 0.5 + bump, value011),
            (0.5 + bump, 0.5 - bump, 0.5 + bump, value101),
            (0.5 + bump, 0.5 + bump, 0.5 + bump, value111),
        ];

        for &(x, y, z, expected_value) in &test_cases {
            let result = query_tree(tree_data, Point3D::new(x, y, z)).unwrap();
            assert_relative_eq!(result.value, expected_value, epsilon = 1e-10);
        }

        // 测试查询立方体中心点
        // 注意：由于这是最近邻搜索，不是插值，中心点会返回最近的一个顶点
        let result = query_tree(tree_data, Point3D::new(0.5, 0.5, 0.5)).unwrap();
        // 中心点到所有顶点距离相等，会返回其中一个
        let valid_values = [
            value000, value100, value010, value001, value110, value011, value101, value111,
        ];
        assert!(
            valid_values.contains(&result.value),
            "Center point returned unexpected value: {}",
            result.value
        );
    }

    #[test]
    fn test_empty_tree() {
        let tree = KdTree::new();
        let query = Point3D::new(1.0, 2.0, 3.0);
        assert!(tree.nearest(&query).is_none());
    }

    #[test]
    fn test_tree_properties() {
        let mut tree = KdTree::new();
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);

        // 添加一些点
        tree.add(Point3D::new(1.0, 2.0, 3.0), 10.0);
        tree.add(Point3D::new(4.0, 5.0, 6.0), 20.0);
        tree.add(Point3D::new(0.0, 0.0, 0.0), 5.0);
        tree.build();

        assert!(!tree.is_empty());
        assert_eq!(tree.len(), 3);
    }
}

#[cfg(test)]
mod geometry_tests {
    use super::*;
    use approx::assert_relative_eq;

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
    fn test_distance_calculation() {
        let p1 = Point3D::new(0.0, 0.0, 0.0);
        let p2 = Point3D::new(3.0, 4.0, 0.0);

        let squared_dist = p1.squared_distance_to(&p2);
        assert_relative_eq!(squared_dist, 25.0, epsilon = 1e-10);

        let dist = p1.distance_to(&p2);
        assert_relative_eq!(dist, 5.0, epsilon = 1e-10);

        // 测试3D距离
        let p3 = Point3D::new(1.0, 1.0, 1.0);
        let p4 = Point3D::new(4.0, 5.0, 1.0);
        let expected_squared_dist = 9.0 + 16.0; // (4-1)² + (5-1)² + (1-1)²
        assert_relative_eq!(
            p3.squared_distance_to(&p4),
            expected_squared_dist,
            epsilon = 1e-10
        );
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
    fn test_point_equality() {
        let p1 = Point3D::new(1.0, 2.0, 3.0);
        let p2 = Point3D::new(1.0, 2.0, 3.0);
        let p3 = Point3D::new(1.0, 2.0, 4.0);

        assert_eq!(p1, p2);
        assert_ne!(p1, p3);
    }
}

/// 新的模块化架构集成测试
#[cfg(test)]
mod refactored_integration_tests {
    use super::*;
    use naturalneighbor3d::{config, errors, utils::validation};
    use ndarray::{Array1, Array2};

    #[test]
    fn test_complete_workflow() {
        // 1. 准备测试数据
        let points = Array2::from_shape_vec(
            (4, 3),
            vec![
                0.0, 0.0, 0.0, // 单位立方体的角点
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0,
            ],
        )
        .expect("Failed to create points array");

        let values = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        let ranges = Array2::from_shape_vec(
            (3, 3),
            vec![
                0.0, 1.0, 0.5, // X 范围: 0 到 1，步长 0.5
                0.0, 1.0, 0.5, // Y 范围: 0 到 1，步长 0.5
                0.0, 1.0, 0.5, // Z 范围: 0 到 1，步长 0.5
            ],
        )
        .expect("Failed to create ranges array");

        // 2. 测试输入验证
        let validation_result =
            validation::validate_inputs(&points.view(), &values.view(), &ranges.view());
        assert!(
            validation_result.is_ok(),
            "Validation should pass for valid inputs"
        );

        // 3. 测试网格参数计算
        let grid_params = grid::compute_grid_params(&ranges.view());
        assert_eq!(grid_params.output_shape, vec![3, 3, 3]);
        assert_eq!(grid_params.starts, vec![0.0, 0.0, 0.0]);

        // 4. 测试坐标转换
        let points_ijk = grid::xyz_to_ijk(&points.view(), &grid_params);
        assert_eq!(points_ijk.shape(), [4, 3]);

        // 验证 (0,0,0) 映射到网格坐标 (0,0,0)
        assert_eq!(points_ijk[[0, 0]], 0.0);
        assert_eq!(points_ijk[[0, 1]], 0.0);
        assert_eq!(points_ijk[[0, 2]], 0.0);

        // 验证 (1,1,1) 映射到网格坐标 (2,2,2)
        assert_eq!(points_ijk[[3, 0]], 2.0);
        assert_eq!(points_ijk[[3, 1]], 2.0);
        assert_eq!(points_ijk[[3, 2]], 2.0);
    }

    #[test]
    fn test_interpolation_config() {
        let config = config::InterpolationConfig::new()
            .with_parallel_threshold(5000)
            .with_max_search_radius(15.0)
            .with_neighbor_limits(5, 100)
            .with_method(config::InterpolationMethod::NaturalNeighbor);

        assert_eq!(config.parallel_threshold, 5000);
        assert_eq!(config.max_search_radius, 15.0);
        assert_eq!(config.min_neighbors, 5);
        assert_eq!(config.max_neighbors, 100);
        assert_eq!(config.method, config::InterpolationMethod::NaturalNeighbor);
        assert!(config.validate());

        // 测试无效配置
        let invalid_config = config::InterpolationConfig::new().with_neighbor_limits(100, 50); // min > max
        assert!(!invalid_config.validate());
    }

    #[test]
    fn test_error_handling() {
        use errors::InterpolationError;

        // 测试数组长度不匹配
        let points = Array2::from_shape_vec((2, 3), vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let values = Array1::from_vec(vec![1.0]); // 错误的长度
        let ranges =
            Array2::from_shape_vec((3, 3), vec![0.0, 1.0, 0.1, 0.0, 1.0, 0.1, 0.0, 1.0, 0.1])
                .unwrap();

        let result = validation::validate_inputs(&points.view(), &values.view(), &ranges.view());
        assert!(matches!(
            result,
            Err(InterpolationError::MismatchedLength {
                points: 2,
                values: 1
            })
        ));

        // 测试无效的点数组形状
        let invalid_points = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        let values = Array1::from_vec(vec![1.0, 2.0]);

        let result =
            validation::validate_inputs(&invalid_points.view(), &values.view(), &ranges.view());
        assert!(matches!(
            result,
            Err(InterpolationError::InvalidPointsShape { .. })
        ));
    }

    #[test]
    fn test_grid_params_functionality() {
        let params =
            config::GridParams::new(vec![10, 20, 30], vec![0.1, 0.2, 0.3], vec![0.0, 1.0, 2.0]);

        // 测试基本功能
        assert_eq!(params.total_points(), 6000);
        assert!(params.validate());

        // 测试轴边界 - 使用近似比较避免浮点数精度问题
        let axis0_bounds = params.axis_bounds(0).unwrap();
        assert!((axis0_bounds.0 - 0.0).abs() < f64::EPSILON);
        assert!((axis0_bounds.1 - 0.9).abs() < f64::EPSILON);

        let axis1_bounds = params.axis_bounds(1).unwrap();
        assert!((axis1_bounds.0 - 1.0).abs() < f64::EPSILON);
        assert!((axis1_bounds.1 - 4.8).abs() < 1e-10);

        let axis2_bounds = params.axis_bounds(2).unwrap();
        assert!((axis2_bounds.0 - 2.0).abs() < f64::EPSILON);
        assert!((axis2_bounds.1 - 10.7).abs() < 1e-10);

        assert_eq!(params.axis_bounds(3), None);

        // 测试无效参数
        let invalid_params = config::GridParams::new(
            vec![0, 20, 30], // 无效：零大小
            vec![0.1, 0.2, 0.3],
            vec![0.0, 1.0, 2.0],
        );
        assert!(!invalid_params.validate());
    }

    #[test]
    fn test_geometry_and_kdtree_integration() {
        // 测试几何操作
        let point1 = Point3D::new(0.0, 0.0, 0.0);
        let point2 = Point3D::new(3.0, 4.0, 0.0);

        let distance = point1.distance_to(&point2);
        assert!((distance - 5.0).abs() < f64::EPSILON);

        let sq_distance = point1.squared_distance_to(&point2);
        assert_eq!(sq_distance, 25.0);

        // 测试 KD 树集成
        let mut tree = KdTree::new();
        tree.add(Point3D::new(0.0, 0.0, 0.0), 1.0);
        tree.add(Point3D::new(1.0, 0.0, 0.0), 2.0);
        tree.add(Point3D::new(0.0, 1.0, 0.0), 3.0);
        tree.add(Point3D::new(1.0, 1.0, 0.0), 4.0);
        tree.build();

        let query_point = Point3D::new(0.1, 0.1, 0.0);
        if let Some(result) = tree.nearest(&query_point) {
            assert_eq!(result.value, 1.0); // 应该找到原点
            assert!(result.distance < 1.0); // 应该很接近
        } else {
            panic!("最近邻查询应该返回结果");
        }

        // 测试半径搜索
        let neighbors = tree.neighbors_within_radius(&query_point, 2.0); // radius_sq = 4.0
        assert!(!neighbors.is_empty());
        assert!(neighbors.len() <= 4); // 最多找到所有点
    }

    #[test]
    fn test_interpolation_method_enum() {
        use config::InterpolationMethod;

        assert_eq!(
            InterpolationMethod::NaturalNeighbor.to_string(),
            "natural_neighbor"
        );
        assert_eq!(InterpolationMethod::Linear.to_string(), "linear");
        assert_eq!(InterpolationMethod::NearestNeighbor.to_string(), "nearest");

        // 测试默认值
        assert_eq!(
            InterpolationMethod::default(),
            InterpolationMethod::NaturalNeighbor
        );
    }

    #[test]
    fn test_comprehensive_error_scenarios() {
        use errors::InterpolationError;

        // 测试空数据
        let empty_points = Array2::zeros((0, 3));
        let empty_values = Array1::zeros(0);
        let ranges =
            Array2::from_shape_vec((3, 3), vec![0.0, 1.0, 0.1, 0.0, 1.0, 0.1, 0.0, 1.0, 0.1])
                .unwrap();

        let result =
            validation::validate_inputs(&empty_points.view(), &empty_values.view(), &ranges.view());
        assert!(matches!(result, Err(InterpolationError::EmptyData)));

        // 测试无效范围
        let points = Array2::from_shape_vec((2, 3), vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let values = Array1::from_vec(vec![1.0, 2.0]);
        let invalid_ranges = Array2::from_shape_vec(
            (3, 3),
            vec![1.0, 0.0, 0.1, 0.0, 1.0, 0.1, 0.0, 1.0, 0.1], // start > stop
        )
        .unwrap();

        let result =
            validation::validate_inputs(&points.view(), &values.view(), &invalid_ranges.view());
        assert!(matches!(
            result,
            Err(InterpolationError::InvalidRange { axis: 0, .. })
        ));

        // 测试无效步长
        let invalid_step_ranges = Array2::from_shape_vec(
            (3, 3),
            vec![0.0, 1.0, -0.1, 0.0, 1.0, 0.1, 0.0, 1.0, 0.1], // 负步长
        )
        .unwrap();

        let result = validation::validate_inputs(
            &points.view(),
            &values.view(),
            &invalid_step_ranges.view(),
        );
        assert!(matches!(
            result,
            Err(InterpolationError::InvalidStep { axis: 0, .. })
        ));
    }
}
