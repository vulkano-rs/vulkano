#version 450

// The triangle vertex positions.
layout(location = 0) in vec2 position;

// The per-instance data.
layout(location = 1) in vec2 position_offset;
layout(location = 2) in float scale;

void main() {
	// Apply the scale and offset for the instance.
	gl_Position = vec4(position * scale + position_offset, 0.0, 1.0);
}
