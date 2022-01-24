use std::f32::consts::PI;

use nalgebra::{Point3, Rotation3, Translation3, Vector2, Vector3};

use crate::asset::model::{Model, Vertex};
use crate::scene::camera::Camera;
use crate::scene::light::PointLight;
use crate::Scene;

/// Unless you are crazy, you should probably use some other methods for importing models and scenes,
/// like converting them to an in-house format or using the glTF importer.
/// This unholy file only exists to prevent introducing even more dependencies.

/// Generate a scene of six cubes;
pub fn generate_scene() -> Scene {
    let translations = vec![
        Translation3::<f32>::new(-1.4625, 1.3897, 1.0551),
        Translation3::<f32>::new(-1.6013, 2.4059, -1.0525),
        Translation3::<f32>::new(1.6057, -1.8776, -1.8982),
        Translation3::<f32>::new(-0.9783, -0.3757, -1.5997),
        Translation3::<f32>::new(0.7279, -0.7224, 1.6402),
        Translation3::<f32>::new(0.8448, 1.7458, -0.3116),
    ];

    let rotations = vec![
        Rotation3::<f32>::from_euler_angles(-1.3779, 0.5408, -0.2679),
        Rotation3::<f32>::from_euler_angles(1.6480, -3.1284, -0.3431),
        Rotation3::<f32>::from_euler_angles(0.2602, 2.7593, -0.7464),
        Rotation3::<f32>::from_euler_angles(-1.6771, -1.6910, -1.7670),
        Rotation3::<f32>::from_euler_angles(-1.9735, 3.0947, 2.2616),
        Rotation3::<f32>::from_euler_angles(2.0737, 1.0701, -1.2355),
    ];

    let objects: Vec<_> = (0..6)
        .into_iter()
        .map(|i| {
            let transforms = nalgebra::convert(translations[i] * rotations[i]);
            (format!("{}", i + 1), transforms)
        })
        .collect();

    let point_lights = vec![
        (
            PointLight {
                luminance: Vector3::new(1.0, 0.5, 0.8),
            },
            Point3::from(Vector3::new(0.203887, 3.85083, 4.0084)),
        ),
        (
            PointLight {
                luminance: Vector3::new(0.5, 0.6, 1.0),
            },
            Point3::from(Vector3::new(-0.040132, -4.11927, 2.60012)),
        ),
    ];

    let camera = Camera {
        position: Point3::from(Vector3::new(-3.09931, -3.35153, 0.369489)),
        target: Point3::origin(),
        up: Vector3::new(0., 0., 1.),
        fov_y: PI / 2.,
        near: 1.,
        far: 100.,
    };

    Scene {
        objects,
        point_lights,
        camera,
    }
}

/// Generate a simple cube model.
pub fn cube(base_color: String) -> Model {
    fn v(n: u32, u: u32, v: u32, x: u32, y: u32, z: u32) -> Vertex {
        Vertex {
            position: Point3::new(x as f32 - 0.5, y as f32 - 0.5, z as f32 - 0.5),
            normal: match n {
                1 => Vector3::new(0., 0., 1.),
                2 => Vector3::new(-1., 0., 0.),
                3 => Vector3::new(0., -1., 0.),
                4 => Vector3::new(0., 1., 0.),
                5 => Vector3::new(1., 0., 0.),
                6 => Vector3::new(0., 0., -1.),
                _ => unreachable!(),
            },
            uv: Vector2::new(u as f32, v as f32),
        }
    }
    Model {
        vertices: vec![
            v(1, 0, 1, 0, 1, 1),
            v(1, 1, 1, 1, 1, 1),
            v(1, 0, 0, 0, 0, 1),
            v(1, 1, 0, 1, 0, 1),
            //
            v(6, 0, 1, 0, 0, 0),
            v(6, 1, 1, 1, 0, 0),
            v(6, 0, 0, 0, 1, 0),
            v(6, 1, 0, 1, 1, 0),
            //
            v(3, 0, 1, 0, 0, 1),
            v(3, 1, 1, 1, 0, 1),
            v(3, 0, 0, 0, 0, 0),
            v(3, 1, 0, 1, 0, 0),
            //
            v(4, 0, 1, 1, 1, 1),
            v(4, 1, 1, 0, 1, 1),
            v(4, 0, 0, 1, 1, 0),
            v(4, 1, 0, 0, 1, 0),
            //
            v(5, 0, 1, 1, 0, 1),
            v(5, 1, 1, 1, 1, 1),
            v(5, 0, 0, 1, 0, 0),
            v(5, 1, 0, 1, 1, 0),
            //
            v(2, 0, 1, 0, 1, 1),
            v(2, 1, 1, 0, 0, 1),
            v(2, 0, 0, 0, 1, 0),
            v(2, 1, 0, 0, 0, 0),
        ],
        indices: (0u32..6)
            .into_iter()
            .flat_map(|face| {
                vec![0, 2, 1, 1, 2, 3]
                    .into_iter()
                    .map(move |x| x + face * 4)
            })
            .collect(),
        base_color,
    }
}
