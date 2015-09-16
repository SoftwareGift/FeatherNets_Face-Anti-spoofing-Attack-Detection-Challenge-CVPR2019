clx_bin_sources = \
	kernel_demo.clx.bin               \
	$(NULL)

convert_binary_to_text_sh = \
	$(top_srcdir)/tools/convert-binary-to-text.sh

cl_bin_dir = $(top_srcdir)/cl_bin

all-local: $(clx_bin_sources)

$(clx_bin_sources): %.clx.bin: $(cl_bin_dir)/%.cl.bin
	@$(convert_binary_to_text_sh) $< $@

CLEANFILES = $(clx_bin_sources)
