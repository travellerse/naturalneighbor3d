use criterion::{black_box, criterion_group, criterion_main, Criterion};
use naturalneighbor3d::core::geometry::Point3D;
use naturalneighbor3d::core::kdtree::KdTree;

fn bench_kdtree_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("kdtree_build");

    for size in [100, 1000, 10000].iter() {
        group.bench_with_input(format!("kdtree_build_{}", size), size, |b, &size| {
            b.iter(|| {
                let mut tree = KdTree::new();
                for i in 0..size {
                    let x = (i as f64) * 0.1;
                    let y = (i as f64) * 0.2;
                    let z = (i as f64) * 0.3;
                    tree.add(Point3D::new(x, y, z), i as f64);
                }
                tree.build();
                black_box(tree)
            });
        });
    }
    group.finish();
}

fn bench_kdtree_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("kdtree_query");

    for size in [100, 1000, 10000].iter() {
        // 预构建树
        let mut tree = KdTree::new();
        for i in 0..*size {
            let x = (i as f64) * 0.1;
            let y = (i as f64) * 0.2;
            let z = (i as f64) * 0.3;
            tree.add(Point3D::new(x, y, z), i as f64);
        }
        tree.build();

        group.bench_with_input(format!("kdtree_query_{}", size), &tree, |b, tree| {
            b.iter(|| {
                let query = Point3D::new(black_box(50.0), black_box(100.0), black_box(150.0));
                tree.nearest(&query)
            });
        });
    }
    group.finish();
}

fn bench_kdtree_batch_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("kdtree_batch_query");

    // 构建一个中等大小的树
    let mut tree = KdTree::new();
    for i in 0..1000 {
        let x = (i as f64) * 0.1;
        let y = (i as f64) * 0.2;
        let z = (i as f64) * 0.3;
        tree.add(Point3D::new(x, y, z), i as f64);
    }
    tree.build();

    for batch_size in [10, 100, 1000].iter() {
        // 预生成查询点
        let queries: Vec<Point3D<f64>> = (0..*batch_size)
            .map(|i| Point3D::new((i as f64) * 0.05, (i as f64) * 0.15, (i as f64) * 0.25))
            .collect();

        group.bench_with_input(
            format!("batch_query_{}", batch_size),
            &queries,
            |b, queries| {
                b.iter(|| {
                    for query in queries {
                        black_box(tree.nearest(query));
                    }
                });
            },
        );
    }
    group.finish();
}

fn bench_point_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("point_operations");

    let p1 = Point3D::new(1.0, 2.0, 3.0);
    let p2 = Point3D::new(4.0, 5.0, 6.0);

    group.bench_function("point_creation", |b| {
        b.iter(|| Point3D::new(black_box(1.0), black_box(2.0), black_box(3.0)));
    });

    group.bench_function("squared_distance", |b| {
        b.iter(|| black_box(p1.squared_distance_to(&p2)));
    });

    group.bench_function("distance", |b| {
        b.iter(|| black_box(p1.distance_to(&p2)));
    });

    group.finish();
}

fn bench_cube_interpolation(c: &mut Criterion) {
    let mut group = c.benchmark_group("cube_interpolation");

    // 创建不同大小的立方体网格
    for grid_size in [5, 10, 20].iter() {
        let mut tree = KdTree::new();

        // 创建 grid_size^3 的网格
        for i in 0..*grid_size {
            for j in 0..*grid_size {
                for k in 0..*grid_size {
                    let x = i as f64;
                    let y = j as f64;
                    let z = k as f64;
                    let value = (i * grid_size * grid_size + j * grid_size + k) as f64;
                    tree.add(Point3D::new(x, y, z), value);
                }
            }
        }
        tree.build();

        group.bench_with_input(
            format!(
                "cube_interpolation_{}x{}x{}",
                grid_size, grid_size, grid_size
            ),
            &tree,
            |b, tree| {
                b.iter(|| {
                    // 在立方体内随机查询多个点
                    for i in 0..100 {
                        let x = (i as f64 * 0.1) % (*grid_size as f64);
                        let y = (i as f64 * 0.2) % (*grid_size as f64);
                        let z = (i as f64 * 0.3) % (*grid_size as f64);

                        let query = Point3D::new(x, y, z);
                        black_box(tree.nearest(&query));
                    }
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_kdtree_build,
    bench_kdtree_query,
    bench_kdtree_batch_query,
    bench_point_operations,
    bench_cube_interpolation
);
criterion_main!(benches);
