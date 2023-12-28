#version 450
#extension GL_EXT_mesh_shader : require



// In mesh shaders you have to load all data manually from storage buffers, which are declared just like uniform
// buffers, but using the `buffer` keyword. You may not use:
// * `in`: Unlike vertex shaders, Mesh shaders do not have an input assembly (IA) stage that pulls data from buffers
//    and forwards them to the vertex shaders as `in` inputs.
// * `uniform`: Uniform buffers have to be of constant size, but as our buffers may have a varying amount of data,
//    they have to be storage buffers instead.
//
// The triangle vertex positions.
layout(set = 0, binding = 0) buffer VertexBuffer {
	vec2 position[];
} buffer_vertex;

// The per-instance data.
struct Instance {
	vec2 position_offset;
	float scale;
};

layout(set = 0, binding = 1) buffer InstanceBuffer {
	Instance instance[];
} buffer_instance;



// This declaration specifies the workgroup size of the mesh shader, similarly to compute shaders
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
// This declares the primitive you want to emit, typically triangles, as well as maximum amount of vertices and primitives you may emit.
layout(triangles, max_vertices = 3, max_primitives = 1) out;



const uint rows = 10;
const uint cols = 10;
const uint n_instances = rows * cols;



void main() {
	vec2 position_offset;
	float scale;

	const bool LOAD_FROM_INSTANCE_BUFFER = true;
	if (LOAD_FROM_INSTANCE_BUFFER) {
		// load instance data from buffers, similarly to doing an instanced draw
		Instance instance = buffer_instance.instance[gl_GlobalInvocationID.y * rows + gl_GlobalInvocationID.x];
		position_offset = instance.position_offset;
		scale = instance.scale;
	} else {
		// generate the geometry dynamically in the mesh shader
		uint c = gl_GlobalInvocationID.x;
		uint r = gl_GlobalInvocationID.y;

		// the same algo for generating the triangle data as in the instanced example
		float half_cell_w = 0.5 / float(cols);
		float half_cell_h = 0.5 / float(rows);
		float x = half_cell_w + (c / float(cols)) * 2.0 - 1.0;
		float y = half_cell_h + (r / float(rows)) * 2.0 - 1.0;
		position_offset = vec2(x, y);
		scale = (2.0 / float(rows)) * (c * float(rows) + r) / n_instances;
	}

	// dynamically set the amount of vertices and triangles that you would like to emit
	// * may not exceed the maximum specified in the out declaration above
	// * must be called before emitting any vertices or indices
	SetMeshOutputsEXT(
		3, // vertices
		1 // triangles = indices / 3
	);

	// emit vertex data
	for (uint i = 0; i < 3; i++) {
		// like setting gl_Position but for each vertex
		gl_MeshVerticesEXT[i].gl_Position = vec4(buffer_vertex.position[i] * scale + position_offset, 0.0, 1.0);
	}

	// emit triangle indices
	gl_PrimitiveTriangleIndicesEXT[0] = uvec3(0, 1, 2);
}
