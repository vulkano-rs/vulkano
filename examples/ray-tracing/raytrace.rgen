#version 460
#extension GL_EXT_ray_tracing : require

struct Camera {
  mat4 viewProj;     // Camera view * projection
  mat4 viewInverse;  // Camera inverse view matrix
  mat4 projInverse;  // Camera inverse projection matrix
};

layout(location = 0) rayPayloadEXT vec3 hitValue;

layout(set = 0, binding = 0) uniform accelerationStructureEXT topLevelAS;
layout(set = 0, binding = 1) uniform _Camera { Camera camera; };
layout(set = 1, binding = 0, rgba32f) uniform image2D image;

void main() {
  const vec2 pixelCenter = vec2(gl_LaunchIDEXT.xy) + vec2(0.5);
  const vec2 inUV = pixelCenter / vec2(gl_LaunchSizeEXT.xy);
  vec2 d = inUV * 2.0 - 1.0;

  vec4 origin = camera.viewInverse * vec4(0, 0, 0, 1);
  vec4 target = camera.projInverse * vec4(d.x, d.y, 1, 1);
  vec4 direction = camera.viewInverse * vec4(normalize(target.xyz), 0);

  uint rayFlags = gl_RayFlagsOpaqueEXT;
  float tMin = 0.001;
  float tMax = 10000.0;

  traceRayEXT(topLevelAS,     // acceleration structure
              rayFlags,       // rayFlags
              0xFF,           // cullMask
              0,              // sbtRecordOffset
              0,              // sbtRecordStride
              0,              // missIndex
              origin.xyz,     // ray origin
              tMin,           // ray min range
              direction.xyz,  // ray direction
              tMax,           // ray max range
              0               // payload (location = 0)
  );

  imageStore(image, ivec2(gl_LaunchIDEXT.xy), vec4(hitValue, 1.0));
}
