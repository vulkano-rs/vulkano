#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadEXT vec3 hit_value;

layout(set = 0, binding = 0) uniform accelerationStructureEXT top_level_as;
layout(set = 0, binding = 1) uniform Camera {
    mat4 view_proj;    // Camera view * projection
    mat4 view_inverse; // Camera inverse view matrix
    mat4 proj_inverse; // Camera inverse projection matrix
} camera;
layout(set = 1, binding = 0, rgba32f) uniform image2D image;

void main() {
    const vec2 pixel_center = vec2(gl_LaunchIDEXT.xy) + vec2(0.5);
    const vec2 in_uv = pixel_center / vec2(gl_LaunchSizeEXT.xy);
    vec2 d = in_uv * 2.0 - 1.0;

    vec4 origin = camera.view_inverse * vec4(0, 0, 0, 1);
    vec4 target = camera.proj_inverse * vec4(d.x, d.y, 1, 1);
    vec4 direction = camera.view_inverse * vec4(normalize(target.xyz), 0);

    uint ray_flags = gl_RayFlagsOpaqueEXT;
    float t_min = 0.001;
    float t_max = 10000.0;

    traceRayEXT(
        top_level_as,  // acceleration structure
        ray_flags,     // rayFlags
        0xFF,          // cullMask
        0,             // sbtRecordOffset
        0,             // sbtRecordStride
        0,             // missIndex
        origin.xyz,    // ray origin
        t_min,         // ray min range
        direction.xyz, // ray direction
        t_max,         // ray max range
        0);            // payload (location = 0)

    imageStore(image, ivec2(gl_LaunchIDEXT.xy), vec4(hit_value, 1.0));
}
