# Version 0.5.1 (TBA)

- Added support for `VK_EXT_full_screen_exclusive`
    + const `STRUCTURE_TYPE_SURFACE_FULL_SCREEN_EXCLUSIVE_INFO_EXT`
    + const `FullScreenExclusiveEXT`
    + const `FULL_SCREEN_EXCLUSIVE_DEFAUlT_EXT`
    + const `FULL_SCREEN_EXCLUSIVE_ALLOWED_EXT`
    + const `FULL_SCREEN_EXCLUSIVE_DISALLOWED_EXT`
    + const `FULL_SCREEN_EXCLUSIVE_APPLICATION_CONTROLLED_EXT`
    + const `FULL_SCREEN_EXCLUSIVE_MAX_ENUM_EXT`
    + struct `SurfaceFullScreenExclusiveInfoEXT`

# Version 0.5.0 (2019-11-01)

- Add const `STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES_KHR` and
  struct `PhysicalDevice16BitStorageFeaturesKHR` for `VK_KHR_16bit_storage`
  extension.
- Removed the following deprecated constants
    +   `STRUCTURE_TYPE_DEBUG_REPORT_CREATE_INFO_EXT`
    +   `COLORSPACE_SRGB_NONLINEAR_KHR`
    +   `COLOR_SPACE_DISPLAY_P3_LINEAR_EXT`
    +   `COLOR_SPACE_SCRGB_LINEAR_EXT`
    +   `COLOR_SPACE_SCRGB_NONLINEAR_EXT`
    +   `COLOR_SPACE_BT2020_NONLINEAR_EXT`
    +   `DEBUG_REPORT_OBJECT_TYPE_DEBUG_REPORT_EXT`

- Removed the deprecated constants and functions related to `VK_EXT_debug_report` and `VK_EXT_debug_marker` and adding the constants and functions related to `VK_EXT_debug_utils`

# Version 0.4.0 (2018-11-16)

- Removed MIR support

# Version 0.3.4 (2018-11-08) **yanked**

- Accidentally released with breaking change with the removal of MIR support.
