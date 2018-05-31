clx_sources = \
	kernel_csc.clx                \
	kernel_demo.clx               \
	kernel_defog_dcp.clx          \
	kernel_min_filter.clx         \
	kernel_bi_filter.clx          \
	kernel_tnr.clx                \
	kernel_bayer_pipe.clx         \
	kernel_bayer_basic.clx        \
	kernel_fisheye.clx            \
	kernel_rgb_pipe.clx           \
	kernel_yuv_pipe.clx           \
	kernel_tonemapping.clx        \
	kernel_newtonemapping.clx     \
	kernel_image_scaler.clx       \
	kernel_retinex.clx            \
	kernel_gauss.clx              \
	kernel_gauss_lap_pyramid.clx  \
	kernel_geo_map.clx            \
	kernel_wavelet_denoise.clx    \
	kernel_wavelet_haar.clx       \
	kernel_wavelet_coeff.clx      \
	kernel_wire_frame.clx         \
	kernel_3d_denoise.clx         \
	kernel_3d_denoise_slm.clx     \
	kernel_image_warp.clx         \
	$(NULL)

add_quotation_marks_sh = \
	$(top_srcdir)/tools/add-quotation-marks.sh

cl_dir = $(top_srcdir)/shaders/cl

all-local: $(clx_sources)

$(clx_sources): %.clx: $(cl_dir)/%.cl
	@$(add_quotation_marks_sh) $< $@

CLEANFILES = $(clx_sources)
