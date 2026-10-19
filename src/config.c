@@
     video_filter_method = ini_section_get_int(cat, "video_filter_method", 1);
@@
     video_gl_input_scale = ini_section_get_double(cat, "video_gl_input_scale", 1.0);
     video_gl_input_scale_mode = ini_section_get_int(cat, "video_gl_input_scale_mode", FULLSCR_SCALE_FULL);
+
+    /* Vulkan shader input scale */
+    video_vk_input_scale = ini_section_get_double(cat, "video_vk_input_scale", 1.0);
+    video_vk_input_scale_mode = ini_section_get_int(cat, "video_vk_input_scale_mode", FULLSCR_SCALE_FULL);
@@
-    if (video_gl_input_scale != 1.0) {
-        ini_section_set_double(cat, "video_gl_input_scale", video_gl_input_scale);
-    } else {
-        ini_section_delete_var(cat, "video_gl_input_scale");
-    }
-
-    if (video_gl_input_scale_mode != FULLSCR_SCALE_FULL) {
-        ini_section_set_int(cat, "video_gl_input_scale_mode", video_gl_input_scale_mode);
-    } else {
-        ini_section_delete_var(cat, "video_gl_input_scale_mode");
-    }
+    if (video_gl_input_scale != 1.0) {
+        ini_section_set_double(cat, "video_gl_input_scale", video_gl_input_scale);
+    } else {
+        ini_section_delete_var(cat, "video_gl_input_scale");
+    }
+
+    if (video_gl_input_scale_mode != FULLSCR_SCALE_FULL) {
+        ini_section_set_int(cat, "video_gl_input_scale_mode", video_gl_input_scale_mode);
+    } else {
+        ini_section_delete_var(cat, "video_gl_input_scale_mode");
+    }
+
+    /* Persist Vulkan shader input scale */
+    if (video_vk_input_scale != 1.0) {
+        ini_section_set_double(cat, "video_vk_input_scale", video_vk_input_scale);
+    } else {
+        ini_section_delete_var(cat, "video_vk_input_scale");
+    }
+
+    if (video_vk_input_scale_mode != FULLSCR_SCALE_FULL) {
+        ini_section_set_int(cat, "video_vk_input_scale_mode", video_vk_input_scale_mode);
+    } else {
+        ini_section_delete_var(cat, "video_vk_input_scale_mode");
+    }
@@
