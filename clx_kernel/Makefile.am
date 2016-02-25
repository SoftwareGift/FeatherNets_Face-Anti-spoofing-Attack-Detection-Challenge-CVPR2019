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
	kernel_gamma.clx              \
	kernel_hdr_lab.clx            \
	kernel_hdr_rgb.clx            \
	kernel_macc.clx               \
	kernel_snr.clx                \
	kernel_tnr_rgb.clx            \
	kernel_tnr_yuv.clx            \
	kernel_bayer_pipe.clx         \
	kernel_bayer_basic.clx         \
	kernel_wb.clx                 \
	kernel_ee.clx                 \
	kernel_dpc.clx                \
	kernel_rgb_pipe.clx           \
	kernel_bnr.clx                \
	kernel_yuv_pipe.clx           \
	kernel_tonemapping.clx        \
	kernel_newtonemapping.clx        \
	kernel_biyuv.clx              \
	kernel_image_scaler.clx       \
	kernel_retinex.clx       \
	kernel_gauss.clx       \
	$(NULL)

cl_quote_sh = \
	$(top_srcdir)/tools/cl_double_quotation.sh

cl_kernel_dir = $(top_srcdir)/cl_kernel

all-local: $(clx_kernel_sources)

$(clx_kernel_sources): %.clx: $(cl_kernel_dir)/%.cl
	@$(cl_quote_sh) $< $@

CLEANFILES = $(clx_kernel_sources)
