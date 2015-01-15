# Copyright (c) 2007-2011 Intel Corporation. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sub license, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice (including the
# next paragraph) shall be included in all copies or substantial portions
# of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
# IN NO EVENT SHALL PRECISION INSIGHT AND/OR ITS SUPPLIERS BE LIABLE FOR
# ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

all: html
install-data-local: install-html

EXTRA_DIST = \
	Doxyfile			\
	$(NULL)

XCAM_HEADER_DIR   = $(top_srcdir)/src
XCAM_HEADER_FILES = \
	$(XCAM_HEADER_DIR)/xcam_3a.h		\
	$(XCAM_HEADER_DIR)/xcam_3athread.h	\
	$(XCAM_HEADER_DIR)/xcam_3a_types.h	\
	$(XCAM_HEADER_DIR)/xcam_v4l2subdev.h	\
	$(top_srcdir)/interface/gstxcaminterface.h \
	$(NULL)

XCAM_HTML_FOOTER		= xcam_footer.html
XCAM_HTML_FRAGMENTS	= $(XCAM_HTML_FOOTER)

export XCAM_HEADER_DIR
export XCAM_HEADER_FILES
export XCAM_HTML_FOOTER
html-out/index.html: Doxyfile $(XCAM_HEADER_FILES) $(XCAM_HTML_FRAGMENTS)
	$(DOXYGEN) $<

if ENABLE_DOCS
html: html-out/index.html
install-html-local:
	install -d $(DESTDIR)$(docdir)/html
	for file in `ls html-out/` ; do \
	    if test -f html-out/$$file ; then \
		install -m 0644 html-out/$$file $(DESTDIR)$(docdir)/html ; \
	    else \
		install -d $(DESTDIR)$(docdir)/html/$$file ; \
		install -m 0644 html-out/$$file/* $(DESTDIR)$(docdir)/html/$$file; \
	    fi ; \
	done
uninstall-local:
	rm -rf $(DESTDIR)$(docdir)/html
endif

# Extra clean files so that maintainer-clean removes *everything*
MAINTAINERCLEANFILES = Makefile.in
