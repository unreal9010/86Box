@@
-    if (noshadersloaded) {
-        m_devFuncs->vkCmdBlitImage(cmdBufs, src_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, swapchainImages[swapchain_image_index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bregion, cur_video_f[...]
-    } else {
-#ifdef LIBRA_RUNTIME_VULKAN
-        m_devFuncs->vkCmdCopyImage(cmdBufs, src_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, shaderSrcImages[swapchain_image_index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cregion);
-#endif
-    }
+    if (noshadersloaded) {
+        m_devFuncs->vkCmdBlitImage(cmdBufs, src_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, swapchainImages[swapchain_image_index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bregion, cur_video_filter_method ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
+    } else {
+#ifdef LIBRA_RUNTIME_VULKAN
+        // Determine shader input scaled size
+        uint32_t src_w = (uint32_t) source.width();
+        uint32_t src_h = (uint32_t) source.height();
+        double scale = std::max(0.5, video_vk_input_scale);
+        uint32_t shader_input_w = std::min<uint32_t>(max_texture_size, std::max<uint32_t>(1, (uint32_t)std::lround(src_w * scale)));
+        uint32_t shader_input_h = std::min<uint32_t>(max_texture_size, std::max<uint32_t>(1, (uint32_t)std::lround(src_h * scale)));
+
+        if (shader_input_w == src_w && shader_input_h == src_h) {
+            m_devFuncs->vkCmdCopyImage(cmdBufs, src_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, shaderSrcImages[swapchain_image_index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cregion);
+        } else {
+            VkImageBlit blitRegion{};
+            blitRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
+            blitRegion.srcSubresource.mipLevel = 0;
+            blitRegion.srcSubresource.baseArrayLayer = 0;
+            blitRegion.srcSubresource.layerCount = 1;
+            blitRegion.dstSubresource = blitRegion.srcSubresource;
+
+            blitRegion.srcOffsets[0].x = source.x();
+            blitRegion.srcOffsets[0].y = source.y();
+            blitRegion.srcOffsets[0].z = 0;
+            blitRegion.srcOffsets[1].x = source.x() + source.width();
+            blitRegion.srcOffsets[1].y = source.y() + source.height();
+            blitRegion.srcOffsets[1].z = 1;
+
+            blitRegion.dstOffsets[0].x = 0;
+            blitRegion.dstOffsets[0].y = 0;
+            blitRegion.dstOffsets[0].z = 0;
+            blitRegion.dstOffsets[1].x = (int32_t) shader_input_w;
+            blitRegion.dstOffsets[1].y = (int32_t) shader_input_h;
+            blitRegion.dstOffsets[1].z = 1;
+
+            m_devFuncs->vkCmdBlitImage(cmdBufs,
+                src_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
+                shaderSrcImages[swapchain_image_index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
+                1, &blitRegion,
+                VK_FILTER_NEAREST);
+        }
+#endif
+    }
@@
-        if (i == 0) {
-            vport.x = destination.x();
-            vport.y = destination.y();
-            vport.width = destination.width();
-            vport.height= destination.height();
-#ifndef LIBRASHADER_STATIC
-            error = librashader_inst.vk_filter_chain_frame(&shaderFilterChains[swapchain_image_index][i].chain, cmdBufs, current_frame_shader, { shader_img_src, VK_FORMAT_B8G8R8A8_UNORM, (unsigne[...]
-#else
-            error = libra_vk_filter_chain_frame(&shaderFilterChains[swapchain_image_index][i].chain, cmdBufs, current_frame_shader, { shader_img_src, VK_FORMAT_B8G8R8A8_UNORM, (unsigned int)source.width(), (unsigned int)source.height()}, { shader_img_dst, VK_FORMAT_B8G8R8A8_UNORM, (unsigned int)curEx[...]
-#endif
-        } else {
-            vport.x = 0;
-            vport.y = 0;
-            vport.width = curExtent.width;
-            vport.height = curExtent.height;
-#ifndef LIBRASHADER_STATIC
+        // compute shader input size for librashader call
+        uint32_t src_w = (uint32_t) source.width();
+        uint32_t src_h = (uint32_t) source.height();
+        double scale = std::max(0.5, video_vk_input_scale);
+        uint32_t shader_input_w = std::min<uint32_t>(max_texture_size, std::max<uint32_t>(1, (uint32_t)std::lround(src_w * scale)));
+        uint32_t shader_input_h = std::min<uint32_t>(max_texture_size, std::max<uint32_t>(1, (uint32_t)std::lround(src_h * scale)));
+
+        if (i == 0) {
+            // For first pass, set viewport to destination mapped into shader input space
+            vport.x = (int) std::lround(((double) destination.x() / (double) src_w) * (double) shader_input_w);
+            vport.y = (int) std::lround(((double) destination.y() / (double) src_h) * (double) shader_input_h);
+            vport.width = shader_input_w == src_w ? destination.width() : shader_input_w;
+            vport.height= shader_input_h == src_h ? destination.height() : shader_input_h;
+#ifndef LIBRASHADER_STATIC
+            error = librashader_inst.vk_filter_chain_frame(&shaderFilterChains[swapchain_image_index][i].chain, cmdBufs, current_frame_shader, { shader_img_src, VK_FORMAT_B8G8R8A8_UNORM, (unsigned int)shader_input_w, (unsigned int)shader_input_h}, { shader_img_dst, VK_FORMAT_B8G8R8A8_UNORM, (unsigned int)curExtent.width, (unsigned int)curExtent.height }, &vport, nullptr, nullptr);
+#else
+            error = libra_vk_filter_chain_frame(&shaderFilterChains[swapchain_image_index][i].chain, cmdBufs, current_frame_shader, { shader_img_src, VK_FORMAT_B8G8R8A8_UNORM, (unsigned int)shader_input_w, (unsigned int)shader_input_h}, { shader_img_dst, VK_FORMAT_B8G8R8A8_UNORM, (unsigned int)curExtent.width, (unsigned int)curExtent.height }, &vport, nullptr, nullptr);
+#endif
+        } else {
+            vport.x = 0;
+            vport.y = 0;
+            vport.width = curExtent.width;
+            vport.height = curExtent.height;
+#ifndef LIBRASHADER_STATIC
