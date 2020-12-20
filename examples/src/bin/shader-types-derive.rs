// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

// This example demonstrates how to put derives onto generated Rust structs from
// the Shader types through the "types-meta" options of
// `vulkano_shaders::shader!` macro.

// Vulkano Shader macro is capable to generate Rust structs representing each
// type found in the shader source. These structs appear in the `ty` module
// generated in the same module where the macro was called.
//
// By default each type has only `Clone` and `Copy` implementations. For
// ergonomic purposes developer may want to implement more traits on top of each
// type. For example "standard" traits such as `Default` or `Debug`.
//
// One way to do so is implementing them manually, but it would be hard to do,
// and complicates code maintenances.
//
// Another way is to specify a macro option to automatically put derives and
// blanket impls onto each generated type by the Macro itself.
//
// The macro is capable to implement standard trait derives in smart way ignoring
// `_dummyX` fields whenever these fields make no sense. And in addition to that
// developer can also specify derives of traits from external modules/crates
// whenever such traits provide custom derive feature.

use ron::from_str;
use ron::ser::{to_string_pretty, PrettyConfig};
use std::fmt::{Debug, Display, Error, Formatter};

vulkano_shaders::shader! {
    ty: "compute",
    src: "
        #version 450

        struct Foo {
            float x;
            vec3 z;
        };

        struct Bar {
            vec2 y;
            Foo foo;
        };

        layout(push_constant) uniform PushConstantData {
            int multiple;
        } pc;

        layout(set = 0, binding = 1) buffer Bars {
            Bar bar[];
        };

        void main() {}
    ",
    types_meta: {
        use serde::{Deserialize, Serialize};

        #[derive(Clone, Copy, PartialEq, Debug, Default, Serialize, Deserialize)]

        impl Eq
    }
}

// In the example above the macro generated `Clone`, `Copy`, `PartialEq`,
// `Debug` and `Default` implementations for each declared
// type(`PushConstantData`, `Foo` and `Bar`) in the shader, and applied
// `impl Eq` for each of them too. And it also applied derives of
// `Serialize` and `Deserialize` traits from Serde crate, but it didn't apply
// these things to `Bars` since the `Bars` type does not have size known in
// compile time.
//
// The macro also didn't generate `Display` implementation since we didn't
// specify it. As such we still can implement this trait manually for some
// selected types.

impl Display for crate::ty::Foo {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> Result<(), Error> {
        Debug::fmt(self, formatter)
    }
}

fn main() {
    use crate::ty::*;

    // Prints "Foo { x: 0.0, z: [100.0, 200.0, 300.0] }" skipping "_dummyX" fields.
    println!(
        "{}",
        Foo {
            z: [100.0, 200.0, 300.0],

            ..Default::default()
        }
    );

    let mut bar = Bar {
        y: [5.1, 6.2],

        // Fills all fields with zeroes including "_dummyX" fields, so we don't
        // have to maintain them manually anymore.
        ..Default::default()
    };

    // The data inside "_dummyX" has no use, but we still can fill it with
    // something different from zeroes.
    bar._dummy0 = [5; 8];

    // Objects are equal since "_dummyX" fields ignoring during comparison
    assert_eq!(
        Bar {
            y: [5.1, 6.2],
            ..Default::default()
        },
        bar,
    );

    assert_ne!(Bar::default(), bar);

    bar.foo.x = 125.0;

    // Since we put `Serialize` and `Deserialize` traits to derives list we can
    // serialize and deserialize Shader data

    let serialized = to_string_pretty(&bar, PrettyConfig::default()).unwrap();

    println!("Serialized Bar: {}", serialized);

    let deserialized = from_str::<Bar>(&serialized).unwrap();

    assert_eq!(deserialized.foo.x, 125.0);
}
