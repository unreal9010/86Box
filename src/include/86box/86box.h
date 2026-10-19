@@
 extern int      video_vk_device;            /* (C) video */
 extern double   video_gl_input_scale;       /* (C) OpenGL 3.x input scale */
 extern int      video_gl_input_scale_mode;  /* (C) OpenGL 3.x input stretch mode */
+extern double   video_vk_input_scale;       /* (C) Vulkan input scale for shader inputs */
+extern int      video_vk_input_scale_mode;  /* (C) Vulkan input stretch mode (reuse FULLSCR_SCALE_* enum) */
 extern int      gfxcard[GFXCARD_MAX];       /* (C) graphics/video card */
@@
