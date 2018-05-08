glx_shader_sources = \
	shader_pyramid_copy.comp.glx        \
	$(NULL)

add_quotation_marks_sh = \
	$(top_srcdir)/tools/add-quotation-marks.sh

gl_shader_dir = $(top_srcdir)/gl_shader

all-local: $(glx_shader_sources)

$(glx_shader_sources): %.glx: $(gl_shader_dir)/%.gl
	@$(add_quotation_marks_sh) $< $@

CLEANFILES = $(glx_shader_sources)
