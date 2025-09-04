use approx::assert_relative_eq;
use naturalneighbor3d::core::geometry::Point3D;
use naturalneighbor3d::core::kdtree::KdTree;
/// 立方体测试
/// 仿照 cpptests 中的立方体测试，验证 3D 插值的正确性
use ndarray::{Array1, Array2};

/// 创建单位立方体的 8 个顶点和对应的值
fn create_unit_cube_data() -> (Array2<f64>, Array1<f64>) {
    let mut points = Array2::zeros((8, 3));
    let mut values = Array1::zeros(8);

    // 立方体 8 个顶点
    let cube_points = [
        (0.0, 0.0, 0.0, 1.0), // value000
        (1.0, 0.0, 0.0, 2.0), // value100
        (0.0, 1.0, 0.0, 3.0), // value010
        (0.0, 0.0, 1.0, 4.0), // value001
        (1.0, 1.0, 0.0, 5.0), // value110
        (0.0, 1.0, 1.0, 6.0), // value011
        (1.0, 0.0, 1.0, 7.0), // value101
        (1.0, 1.0, 1.0, 8.0), // value111
    ];

    for (i, &(x, y, z, val)) in cube_points.iter().enumerate() {
        points[[i, 0]] = x;
        points[[i, 1]] = y;
        points[[i, 2]] = z;
        values[i] = val;
    }

    (points, values)
}

/// 创建基于立方体数据的 KD 树
fn create_cube_kdtree() -> KdTree {
    let (points, values) = create_unit_cube_data();
    let mut tree = KdTree::new();

    for i in 0..8 {
        let point = Point3D::new(points[[i, 0]], points[[i, 1]], points[[i, 2]]);
        tree.add(point, values[i]);
    }

    tree.build();
    tree
}

#[cfg(test)]
mod cube_tests {
    use super::*;

    #[test]
    fn test_cube_vertices() {
        let tree = create_cube_kdtree();
        let (points, values) = create_unit_cube_data();

        // 测试每个顶点的查询结果
        for i in 0..8 {
            let query_point = Point3D::new(points[[i, 0]], points[[i, 1]], points[[i, 2]]);
            let result = tree.nearest(&query_point).unwrap();

            assert_relative_eq!(result.distance, 0.0, epsilon = 1e-10);
            assert_relative_eq!(result.value, values[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_cube_edges() {
        let tree = create_cube_kdtree();

        // 测试立方体边的中点
        let edge_tests = [
            // 底面边的中点
            ((0.5, 0.0, 0.0), vec![1.0, 2.0]), // 连接 (0,0,0) 和 (1,0,0)
            ((0.0, 0.5, 0.0), vec![1.0, 3.0]), // 连接 (0,0,0) 和 (0,1,0)
            ((1.0, 0.5, 0.0), vec![2.0, 5.0]), // 连接 (1,0,0) 和 (1,1,0)
            ((0.5, 1.0, 0.0), vec![3.0, 5.0]), // 连接 (0,1,0) 和 (1,1,0)
            // 顶面边的中点
            ((0.5, 0.0, 1.0), vec![4.0, 7.0]), // 连接 (0,0,1) 和 (1,0,1)
            ((0.0, 0.5, 1.0), vec![4.0, 6.0]), // 连接 (0,0,1) 和 (0,1,1)
            ((1.0, 0.5, 1.0), vec![7.0, 8.0]), // 连接 (1,0,1) 和 (1,1,1)
            ((0.5, 1.0, 1.0), vec![6.0, 8.0]), // 连接 (0,1,1) 和 (1,1,1)
            // 垂直边的中点
            ((0.0, 0.0, 0.5), vec![1.0, 4.0]), // 连接 (0,0,0) 和 (0,0,1)
            ((1.0, 0.0, 0.5), vec![2.0, 7.0]), // 连接 (1,0,0) 和 (1,0,1)
            ((0.0, 1.0, 0.5), vec![3.0, 6.0]), // 连接 (0,1,0) 和 (0,1,1)
            ((1.0, 1.0, 0.5), vec![5.0, 8.0]), // 连接 (1,1,0) 和 (1,1,1)
        ];

        for ((x, y, z), possible_values) in edge_tests {
            let query_point = Point3D::new(x, y, z);
            let result = tree.nearest(&query_point).unwrap();

            // 边的中点应该距离最近的顶点有固定的距离
            assert_relative_eq!(result.distance, 0.25, epsilon = 1e-10);

            // 值应该是连接该边的两个顶点值之一
            assert!(
                possible_values.contains(&result.value),
                "Value {} at edge midpoint ({}, {}, {}) should be one of {:?}",
                result.value,
                x,
                y,
                z,
                possible_values
            );
        }
    }

    #[test]
    fn test_cube_faces() {
        let tree = create_cube_kdtree();

        // 测试立方体面的中心点
        let face_tests = [
            // 底面中心 (z=0)
            ((0.5, 0.5, 0.0), vec![1.0, 2.0, 3.0, 5.0]),
            // 顶面中心 (z=1)
            ((0.5, 0.5, 1.0), vec![4.0, 6.0, 7.0, 8.0]),
            // 前面中心 (y=0)
            ((0.5, 0.0, 0.5), vec![1.0, 2.0, 4.0, 7.0]),
            // 后面中心 (y=1)
            ((0.5, 1.0, 0.5), vec![3.0, 5.0, 6.0, 8.0]),
            // 左面中心 (x=0)
            ((0.0, 0.5, 0.5), vec![1.0, 3.0, 4.0, 6.0]),
            // 右面中心 (x=1)
            ((1.0, 0.5, 0.5), vec![2.0, 5.0, 7.0, 8.0]),
        ];

        for ((x, y, z), possible_values) in face_tests {
            let query_point = Point3D::new(x, y, z);
            let result = tree.nearest(&query_point).unwrap();

            // 面中心到最近顶点的距离应该是 0.5
            assert_relative_eq!(result.distance, 0.5, epsilon = 1e-10);

            // 值应该是该面的四个顶点值之一
            assert!(
                possible_values.contains(&result.value),
                "Value {} at face center ({}, {}, {}) should be one of {:?}",
                result.value,
                x,
                y,
                z,
                possible_values
            );
        }
    }

    #[test]
    fn test_cube_center() {
        let tree = create_cube_kdtree();

        // 测试立方体中心点
        let query_point = Point3D::new(0.5, 0.5, 0.5);
        let result = tree.nearest(&query_point).unwrap();

        // 中心点到所有顶点的距离都相等
        let expected_distance = 0.75; // (0.5² + 0.5² + 0.5²)
        assert_relative_eq!(result.distance, expected_distance, epsilon = 1e-10);

        // 值应该是 8 个顶点值之一
        let all_values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert!(
            all_values.contains(&result.value),
            "Value {} at cube center should be one of {:?}",
            result.value,
            all_values
        );
    }

    #[test]
    fn test_cube_near_vertices() {
        let tree = create_cube_kdtree();
        let bump = 1e-12; // 非常小的偏移

        // 测试非常接近各个顶点的位置
        let vertex_tests = [
            ((-bump, -bump, -bump), 1.0),                // 接近 (0,0,0)
            ((1.0 + bump, -bump, -bump), 2.0),           // 接近 (1,0,0)
            ((-bump, 1.0 + bump, -bump), 3.0),           // 接近 (0,1,0)
            ((-bump, -bump, 1.0 + bump), 4.0),           // 接近 (0,0,1)
            ((1.0 + bump, 1.0 + bump, -bump), 5.0),      // 接近 (1,1,0)
            ((-bump, 1.0 + bump, 1.0 + bump), 6.0),      // 接近 (0,1,1)
            ((1.0 + bump, -bump, 1.0 + bump), 7.0),      // 接近 (1,0,1)
            ((1.0 + bump, 1.0 + bump, 1.0 + bump), 8.0), // 接近 (1,1,1)
        ];

        for ((x, y, z), expected_value) in vertex_tests {
            let query_point = Point3D::new(x, y, z);
            let result = tree.nearest(&query_point).unwrap();

            assert_relative_eq!(result.value, expected_value, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_outside_cube() {
        let tree = create_cube_kdtree();

        // 测试立方体外部的点
        let outside_tests = [
            ((-1.0, 0.5, 0.5), vec![1.0, 3.0, 4.0, 6.0]), // 左侧
            ((2.0, 0.5, 0.5), vec![2.0, 5.0, 7.0, 8.0]),  // 右侧
            ((0.5, -1.0, 0.5), vec![1.0, 2.0, 4.0, 7.0]), // 前侧
            ((0.5, 2.0, 0.5), vec![3.0, 5.0, 6.0, 8.0]),  // 后侧
            ((0.5, 0.5, -1.0), vec![1.0, 2.0, 3.0, 5.0]), // 下方
            ((0.5, 0.5, 2.0), vec![4.0, 6.0, 7.0, 8.0]),  // 上方
        ];

        for ((x, y, z), possible_values) in outside_tests {
            let query_point = Point3D::new(x, y, z);
            let result = tree.nearest(&query_point).unwrap();

            // 应该返回对应面上的某个顶点值
            assert!(
                possible_values.contains(&result.value),
                "Point ({}, {}, {}) should have value from {:?}, got {}",
                x,
                y,
                z,
                possible_values,
                result.value
            );
        }
    }

    #[test]
    fn test_performance_with_large_cube() {
        // 创建一个更大的立方体网格用于性能测试
        let grid_size = 5;
        let mut tree = KdTree::new();

        // 创建 5x5x5 的网格
        for i in 0..grid_size {
            for j in 0..grid_size {
                for k in 0..grid_size {
                    let x = i as f64;
                    let y = j as f64;
                    let z = k as f64;
                    let value = (i * grid_size * grid_size + j * grid_size + k) as f64;
                    tree.add(Point3D::new(x, y, z), value);
                }
            }
        }

        tree.build();

        // 进行多次查询以测试性能
        let num_queries = 1000;
        for i in 0..num_queries {
            let x = (i as f64 * 0.1) % (grid_size as f64);
            let y = (i as f64 * 0.2) % (grid_size as f64);
            let z = (i as f64 * 0.3) % (grid_size as f64);

            let query_point = Point3D::new(x, y, z);
            let result = tree.nearest(&query_point);
            assert!(result.is_some(), "Query {} should return a result", i);
        }
    }
}
