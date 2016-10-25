clx_kernel_sources = \
	kernel_3a_stats.clx           \
	kernel_blc.clx                \
	kernel_csc_rgba64torgba.clx   \
	kernel_csc_rgbatolab.clx      \
	kernel_csc_rgbatonv12.clx     \
	kernel_csc_nv12torgba.clx     \
	kernel_csc_yuyvtorgba.clx     \
	kernel_demo.clx               \
	kernel_demosaic.clx           \
	kernel_denoise.clx            \
	kernel_defog_dcp.clx          \
	kernel_gamma.clx              \
	kernel_hdr_lab.clx            \
	kernel_hdr_rgb.clx            \
	kernel_macc.clx               \
	kernel_min_filter.clx         \
	kernel_bi_filter.clx          \
	kernel_snr.clx                \
	kernel_tnr_rgb.clx            \
	kernel_tnr_yuv.clx            \
	kernel_bayer_pipe.clx         \
	kernel_bayer_basic.clx        \
	kernel_wb.clx                 \
	kernel_ee.clx                 \
	kernel_dpc.clx                \
	kernel_rgb_pipe.clx           \
	kernel_bnr.clx                \
	kernel_yuv_pipe.clx           \
	kernel_tonemapping.clx        \
	kernel_newtonemapping.clx     \
	kernel_biyuv.clx              \
	kernel_image_scaler.clx       \
	kernel_retinex.clx            \
	kernel_gauss.clx              \
	kernel_gauss_lap_pyramid.clx  \
	kernel_geo_map.clx            \
	kernel_wavelet_denoise.clx    \
	kernel_wavelet_haar_decomposition.clx       \
	kernel_wavelet_haar_reconstruction.clx  \
	kernel_wavelet_coeff_variance.clx \
	kernel_wavelet_coeff_thresholding.clx \
	kernel_wire_frame.clx         \
	kernel_3d_denoise.clx         \
	kernel_3d_denoise_slm.clx         \
	kernel_image_warp.clx         \
	$(NULL)

cl_quotation_sh = \
	$(top_srcdir)/tools/cl-double-quotation.sh

cl_kernel_dir = $(top_srcdir)/cl_kernel

all-local: $(clx_kernel_sources)

$(clx_kernel_sources): %.clx: $(cl_kernel_dir)/%.cl
	@$(cl_quotation_sh) $< $@

CLEANFILES = $(clx_kernel_sources)
